"""Phased web chat UI and offline actual-LaTeX SVG renderer for SemanticMathEngine."""

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import sys
import threading
from typing import Any, Optional
from urllib.parse import urlsplit

if __package__ in (None, ""):
    _SRC_DIR = Path(__file__).resolve().parents[1]
    if str(_SRC_DIR) not in sys.path:
        sys.path.insert(0, str(_SRC_DIR))

import matplotlib

matplotlib.use("Agg")
from matplotlib import mathtext  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402

from semantic_math.engine import SemanticMathEngine  # noqa: E402
from semantic_math.graph import MathGraphError  # noqa: E402
from semantic_math.interpreter import LLMCallerError  # noqa: E402

_MAX_INPUT_LENGTH = 1000
_MATHTEXT_LOCK = threading.Lock()
_MATHTEXT_FONT = FontProperties(size=18)


class ChatInputError(ValueError):
    """Raised when user chat text fails Phase-i input validation."""


def validate_chat_input(text: Any) -> str:
    """Phase i: Validate and normalize raw user chat text input."""
    if not isinstance(text, str):
        raise ChatInputError("Input message must be a text string.")
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        raise ChatInputError("Please enter a math problem before sending.")
    if len(cleaned) > _MAX_INPUT_LENGTH:
        raise ChatInputError(
            f"Input exceeds maximum length of {_MAX_INPUT_LENGTH} characters."
        )
    return cleaned


def render_latex_svg(latex_expr: str) -> Optional[str]:
    """Phase iii: Render a LaTeX math string to an inline SVG using matplotlib mathtext."""
    if not isinstance(latex_expr, str) or not latex_expr.strip():
        return None

    stripped = latex_expr.strip()
    wrapped = stripped if (stripped.startswith("$") and stripped.endswith("$")) else f"${stripped}$"

    buf = io.BytesIO()
    try:
        with _MATHTEXT_LOCK:
            mathtext.math_to_image(
                wrapped,
                buf,
                prop=_MATHTEXT_FONT,
                dpi=150,
                format="svg",
            )
    except Exception:
        return None

    svg_text = buf.getvalue().decode("utf-8", errors="replace")
    svg_start = svg_text.find("<svg")
    if svg_start == -1:
        return None
    return svg_text[svg_start:]


class ChatSession:
    """Connects Phase-i validation, Phase-ii engine solving, and Phase-iii LaTeX rendering."""

    def __init__(self, engine: Optional[SemanticMathEngine] = None) -> None:
        self.engine = engine if engine is not None else SemanticMathEngine()
        self.history: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def send(self, user_text: Any) -> dict[str, Any]:
        """Validate input, run the math engine, render LaTeX SVG, and store the turn."""
        cleaned = validate_chat_input(user_text)
        output = self.engine.solve(cleaned)
        equation_svg = render_latex_svg(output.equation_latex)
        expr_svg = render_latex_svg(output.latex)

        turn: dict[str, Any] = {
            "ok": True,
            "user_input": cleaned,
            "prompt": output.prompt,
            "raw_json": output.raw_json,
            "result": output.result,
            "latex": output.latex,
            "equation_latex": output.equation_latex,
            "latex_svg": equation_svg or expr_svg,
            "mermaid": output.mermaid,
        }
        with self._lock:
            self.history.append(turn)
        return turn

    def get_history(self) -> list[dict[str, Any]]:
        """Return a shallow copy of recorded chat turns."""
        with self._lock:
            return list(self.history)

    def clear(self) -> None:
        """Reset recorded chat turns."""
        with self._lock:
            self.history.clear()


_CHAT_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Semantic Math Engine Chat</title>
  <style>
    :root {
      --bg: #0f172a;
      --panel: #1e293b;
      --panel-2: #273449;
      --text: #f8fafc;
      --muted: #94a3b8;
      --accent: #38bdf8;
      --accent-dark: #0284c7;
      --user-bg: #1d4ed8;
      --border: #334155;
      --error: #f87171;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      height: 100vh;
      display: flex;
      flex-direction: column;
    }
    header {
      padding: 14px 20px;
      background: var(--panel);
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }
    header h1 { margin: 0; font-size: 1.05rem; font-weight: 600; }
    header .meta { font-size: 0.8rem; color: var(--muted); }
    .btn-secondary {
      background: transparent;
      color: var(--muted);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 6px 12px;
      cursor: pointer;
      font-size: 0.85rem;
    }
    .btn-secondary:hover { color: var(--text); border-color: var(--accent); }
    .chips {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      padding: 10px 20px;
      background: #131c31;
      border-bottom: 1px solid var(--border);
    }
    .chip {
      background: var(--panel);
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 999px;
      padding: 5px 12px;
      font-size: 0.8rem;
      cursor: pointer;
    }
    .chip:hover { border-color: var(--accent); color: var(--accent); }
    #messages {
      flex: 1;
      overflow-y: auto;
      padding: 20px;
      display: flex;
      flex-direction: column;
      gap: 14px;
      max-width: 920px;
      width: 100%;
      margin: 0 auto;
    }
    .bubble {
      max-width: 85%;
      padding: 12px 16px;
      border-radius: 14px;
      line-height: 1.45;
    }
    .bubble.user {
      align-self: flex-end;
      background: var(--user-bg);
      color: #fff;
      border-bottom-right-radius: 4px;
    }
    .bubble.assistant {
      align-self: flex-start;
      background: var(--panel);
      border: 1px solid var(--border);
      border-bottom-left-radius: 4px;
      width: min(760px, 95%);
    }
    .bubble.error {
      border-color: var(--error);
      color: #fecaca;
    }
    .latex-card {
      background: #ffffff;
      color: #0f172a;
      border-radius: 10px;
      padding: 14px 18px;
      margin: 6px 0 10px;
      overflow-x: auto;
      display: flex;
      align-items: center;
      min-height: 54px;
    }
    .latex-card svg {
      max-width: 100%;
      height: auto;
      display: block;
    }
    .answer-row {
      display: flex;
      align-items: center;
      gap: 10px;
      font-size: 0.95rem;
      margin-bottom: 8px;
    }
    .badge {
      background: #0c4a6e;
      color: #bae6fd;
      border: 1px solid #0284c7;
      border-radius: 6px;
      padding: 2px 8px;
      font-weight: 600;
    }
    details {
      margin-top: 6px;
      background: var(--panel-2);
      border-radius: 8px;
      padding: 6px 10px;
      font-size: 0.82rem;
    }
    summary { cursor: pointer; color: var(--muted); font-weight: 500; }
    pre {
      margin: 8px 0 2px;
      white-space: pre-wrap;
      word-break: break-word;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      color: #e2e8f0;
    }
    form {
      background: var(--panel);
      border-top: 1px solid var(--border);
      padding: 14px 20px;
    }
    .input-wrap {
      max-width: 920px;
      margin: 0 auto;
      display: flex;
      gap: 10px;
    }
    input[type="text"] {
      flex: 1;
      padding: 12px 14px;
      border-radius: 10px;
      border: 1px solid var(--border);
      background: var(--bg);
      color: var(--text);
      font-size: 0.95rem;
      outline: none;
    }
    input[type="text"]:focus { border-color: var(--accent); }
    button[type="submit"] {
      background: var(--accent-dark);
      color: #fff;
      border: none;
      border-radius: 10px;
      padding: 0 20px;
      font-weight: 600;
      cursor: pointer;
    }
    button[type="submit"]:disabled { opacity: 0.6; cursor: wait; }
    .empty-state {
      text-align: center;
      color: var(--muted);
      margin: auto;
      max-width: 520px;
      font-size: 0.92rem;
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Semantic Math Text Engine</h1>
      <div class="meta">Ollama (llama3.2:1b) &rarr; Pydantic Schema &rarr; C++ MathGraph &rarr; Rendered LaTeX</div>
    </div>
    <button class="btn-secondary" id="clearBtn" type="button">Clear Chat</button>
  </header>
  <div class="chips">
    <button class="chip" type="button">Add 1 to 3</button>
    <button class="chip" type="button">Can you please add 5 and 4 and 1000</button>
    <button class="chip" type="button">Can you subtract 10 from sum of 20 and 33</button>
    <button class="chip" type="button">Can you find the product of 6, 7, and 2</button>
    <button class="chip" type="button">Divide the sum of 10 and 20 by 5</button>
  </div>
  <main id="messages">
    <div class="empty-state" id="emptyState">
      Enter a math word problem below (or click an example above) to parse it into a graph and render the equation in LaTeX.
    </div>
  </main>
  <form id="chatForm" action="javascript:void(0);">
    <div class="input-wrap">
      <input id="promptInput" type="text" placeholder="e.g. Can you subtract 10 from sum of 20 and 33" autocomplete="off" />
      <button id="sendBtn" type="submit">Solve</button>
    </div>
  </form>
  <script>
    const messagesEl = document.getElementById("messages");
    const emptyStateEl = document.getElementById("emptyState");
    const formEl = document.getElementById("chatForm");
    const inputEl = document.getElementById("promptInput");
    const sendBtn = document.getElementById("sendBtn");
    const clearBtn = document.getElementById("clearBtn");

    function appendUserBubble(text) {
      if (emptyStateEl) emptyStateEl.style.display = "none";
      const div = document.createElement("div");
      div.className = "bubble user";
      div.textContent = text;
      messagesEl.appendChild(div);
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }

    function appendAssistantBubble(data) {
      const div = document.createElement("div");
      div.className = "bubble assistant";

      const answerRow = document.createElement("div");
      answerRow.className = "answer-row";
      const label = document.createElement("span");
      label.textContent = "Answer:";
      const badge = document.createElement("span");
      badge.className = "badge";
      badge.textContent = String(data.result);
      answerRow.append(label, badge);
      div.appendChild(answerRow);

      const card = document.createElement("div");
      card.className = "latex-card";
      if (data.latex_svg) {
        card.innerHTML = data.latex_svg;
      } else {
        const pre = document.createElement("pre");
        pre.style.color = "#0f172a";
        pre.textContent = data.equation_latex || data.latex || "";
        card.appendChild(pre);
      }
      div.appendChild(card);

      const details = document.createElement("details");
      const summary = document.createElement("summary");
      summary.textContent = "LaTeX source, JSON AST & Mermaid graph";
      const pre = document.createElement("pre");
      pre.textContent =
        "LaTeX: " + data.equation_latex + "\n\n" +
        "JSON AST:\n" + JSON.stringify(data.raw_json, null, 2) + "\n\n" +
        "Mermaid:\n" + data.mermaid;
      details.append(summary, pre);
      div.appendChild(details);

      messagesEl.appendChild(div);
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }

    function appendErrorBubble(msg) {
      if (emptyStateEl) emptyStateEl.style.display = "none";
      const div = document.createElement("div");
      div.className = "bubble assistant error";
      div.textContent = "Error: " + msg;
      messagesEl.appendChild(div);
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }

    async function submitPrompt(text) {
      appendUserBubble(text);
      sendBtn.disabled = true;
      inputEl.disabled = true;
      try {
        const res = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: text }),
        });
        const data = await res.json();
        if (!res.ok || !data.ok) {
          appendErrorBubble(data.error || "Failed to solve prompt.");
        } else {
          appendAssistantBubble(data);
        }
      } catch (err) {
        appendErrorBubble(err.message || String(err));
      } finally {
        sendBtn.disabled = false;
        inputEl.disabled = false;
        inputEl.focus();
      }
    }

    formEl.addEventListener("submit", (e) => {
      e.preventDefault();
      const text = inputEl.value.trim();
      if (!text) return;
      inputEl.value = "";
      submitPrompt(text);
    });

    document.querySelectorAll(".chip").forEach((chip) => {
      chip.addEventListener("click", () => {
        inputEl.value = chip.textContent;
        inputEl.focus();
      });
    });

    clearBtn.addEventListener("click", async () => {
      await fetch("/api/clear", { method: "POST" });
      messagesEl.innerHTML = "";
      if (emptyStateEl) {
        emptyStateEl.style.display = "block";
        messagesEl.appendChild(emptyStateEl);
      }
    });
  </script>
</body>
</html>
"""


def _make_handler(session: ChatSession) -> type[BaseHTTPRequestHandler]:
    """Build an HTTP request handler bound to the given ChatSession."""

    class ChatHTTPHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return

        def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json_body(self) -> dict[str, Any]:
            length_header = self.headers.get("Content-Length", "0")
            try:
                length = int(length_header)
            except ValueError as exc:
                raise ChatInputError("Invalid Content-Length header.") from exc
            raw = self.rfile.read(length) if length > 0 else b"{}"
            try:
                parsed = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ChatInputError("Request body must be valid JSON.") from exc
            if not isinstance(parsed, dict):
                raise ChatInputError("JSON payload must be an object.")
            return parsed

        def do_GET(self) -> None:
            path = urlsplit(self.path).path
            if path in ("/", "", "/index.html"):
                html = _CHAT_HTML.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html)))
                self.end_headers()
                self.wfile.write(html)
                return

            if path == "/favicon.ico":
                self.send_response(HTTPStatus.NO_CONTENT)
                self.end_headers()
                return

            if path == "/api/history":
                self._send_json(HTTPStatus.OK, {"ok": True, "history": session.get_history()})
                return

            self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "Not found."})

        def do_POST(self) -> None:
            path = urlsplit(self.path).path
            if path == "/api/clear":
                session.clear()
                self._send_json(HTTPStatus.OK, {"ok": True})
                return

            if path not in ("/api/validate", "/api/chat"):
                self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "Not found."})
                return

            try:
                body = self._read_json_body()
                message = body.get("message")
                if path == "/api/validate":
                    cleaned = validate_chat_input(message)
                    self._send_json(HTTPStatus.OK, {"ok": True, "cleaned": cleaned})
                    return

                turn = session.send(message)
                self._send_json(HTTPStatus.OK, turn)
            except ChatInputError as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
            except MathGraphError as exc:
                self._send_json(
                    HTTPStatus.UNPROCESSABLE_ENTITY, {"ok": False, "error": str(exc)}
                )
            except LLMCallerError as exc:
                self._send_json(HTTPStatus.BAD_GATEWAY, {"ok": False, "error": str(exc)})

    return ChatHTTPHandler


def create_chat_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    engine: Optional[SemanticMathEngine] = None,
) -> tuple[ThreadingHTTPServer, ChatSession]:
    """Create a ThreadingHTTPServer serving the chat window and LaTeX API."""
    session = ChatSession(engine=engine)
    handler_cls = _make_handler(session)
    server = ThreadingHTTPServer((host, port), handler_cls)
    return server, session


def main() -> int:
    """CLI entrypoint to launch the local chat web UI."""
    parser = argparse.ArgumentParser(description="Semantic Math Engine Chat UI")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    args = parser.parse_args()

    server, _session = create_chat_server(host=args.host, port=args.port)
    actual_host, actual_port = server.server_address
    print(f"Semantic Math Chat UI running at http://{actual_host}:{actual_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down chat UI...")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

