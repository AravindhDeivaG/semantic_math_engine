"""Unit and HTTP integration tests for semantic_math.app (chat UI + LaTeX SVG)."""

import json
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

_SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from semantic_math.app import (  # noqa: E402
    ChatInputError,
    ChatSession,
    create_chat_server,
    render_latex_svg,
    validate_chat_input,
)
from semantic_math.engine import SemanticMathEngine  # noqa: E402
from semantic_math.interpreter import LLMCaller  # noqa: E402


class _FakeClient:
    """Fake Ollama client returning mapped JSON expressions by user prompt."""

    def __init__(self, mapping: dict[str, Any], exc: Exception | None = None) -> None:
        self.mapping = mapping
        self.exc = exc

    def chat(self, **kwargs: Any) -> Any:
        if self.exc is not None:
            raise self.exc
        user_msg = kwargs["messages"][-1]["content"]
        for needle, payload in self.mapping.items():
            if needle in user_msg:
                return SimpleNamespace(
                    message=SimpleNamespace(content=json.dumps(payload))
                )
        raise AssertionError(f"Unmapped prompt: {user_msg!r}")


def _http_request(
    url: str, method: str = "GET", payload: dict[str, Any] | None = None
) -> tuple[int, dict[str, Any] | str]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    req = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(req, timeout=5) as resp:
            raw = resp.read().decode("utf-8")
            ctype = resp.headers.get("Content-Type", "")
            return resp.status, json.loads(raw) if "application/json" in ctype else raw
    except HTTPError as err:
        raw = err.read().decode("utf-8")
        return err.code, json.loads(raw)


class TestChatUIAndLatexRenderer(unittest.TestCase):
    """Tests covering Phase i validation, Phase ii solving, and Phase iii LaTeX SVG."""

    def test_phase_i_input_validation(self) -> None:
        self.assertEqual(validate_chat_input("  Add   1 to 3 "), "Add 1 to 3")
        for bad in ("", "   ", None, 42, "x" * 1001):
            with self.subTest(bad=bad):
                with self.assertRaises(ChatInputError):
                    validate_chat_input(bad)

    def test_phase_iii_render_latex_svg(self) -> None:
        expressions = [
            "1 + 3 = 4",
            r"\frac{12}{4} = 3",
            r"(-2)^{3} = -8",
            r"\sqrt{16} = 4",
            r"(10 - 3) \cdot \frac{12}{4} \cdot 2 + (\sqrt[3]{27})^{2} + 5.5 = 56.5",
        ]
        for tex in expressions:
            with self.subTest(tex=tex):
                svg = render_latex_svg(tex)
                self.assertIsNotNone(svg)
                assert svg is not None
                self.assertTrue(svg.lstrip().startswith("<svg"))
                self.assertIn("</svg>", svg)

        self.assertIsNone(render_latex_svg("   "))

    def test_http_chat_endpoints_across_all_phases(self) -> None:
        fake = _FakeClient(
            {
                "subtract 10 from sum of 20 and 33": {
                    "expression": {
                        "op": "subtract",
                        "left": {"op": "add", "operands": [20, 33]},
                        "right": 10,
                    }
                },
                "divide by zero": {
                    "expression": {
                        "op": "divide",
                        "numerator": 5,
                        "denominator": 0,
                    }
                },
                "sine of 30": {"expression": None},
            }
        )
        engine = SemanticMathEngine(llm_caller=LLMCaller(client=fake))
        server, session = create_chat_server(host="127.0.0.1", port=0, engine=engine)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{port}"

        try:
            # Phase i: HTML UI served cleanly (including query strings & index.html)
            status, html = _http_request(f"{base}/")
            self.assertEqual(status, 200)
            self.assertIsInstance(html, str)
            self.assertIn("Semantic Math Text Engine", html)
            self.assertIn("/api/chat", html)
            # Ensure JS escaped newlines are preserved (not expanded into broken JS literals)
            self.assertIn(r'"\n\n"', html)

            status_q, html_q = _http_request(f"{base}/?preview=1")
            self.assertEqual(status_q, 200)
            self.assertIn("Semantic Math Text Engine", html_q)

            status_idx, _ = _http_request(f"{base}/index.html")
            self.assertEqual(status_idx, 200)

            with urlopen(f"{base}/favicon.ico", timeout=5) as fav_resp:
                self.assertEqual(fav_resp.status, 204)

            # Phase i: /api/validate endpoint
            status, res = _http_request(
                f"{base}/api/validate", method="POST", payload={"message": "  Add 1 to 3 "}
            )
            self.assertEqual(status, 200)
            self.assertEqual(res, {"ok": True, "cleaned": "Add 1 to 3"})

            status, err_res = _http_request(
                f"{base}/api/validate", method="POST", payload={"message": "   "}
            )
            self.assertEqual(status, 400)
            self.assertFalse(err_res["ok"])  # type: ignore[index]

            # Phase ii & iii: /api/chat solves and attaches rendered LaTeX SVG
            status, chat_res = _http_request(
                f"{base}/api/chat",
                method="POST",
                payload={"message": "Can you subtract 10 from sum of 20 and 33"},
            )
            self.assertEqual(status, 200)
            assert isinstance(chat_res, dict)
            self.assertTrue(chat_res["ok"])
            self.assertEqual(chat_res["result"], 43.0)
            self.assertEqual(chat_res["equation_latex"], "20 + 33 - 10 = 43")
            self.assertTrue(chat_res["latex_svg"].lstrip().startswith("<svg"))
            self.assertTrue(chat_res["mermaid"].startswith("flowchart TD"))

            # History recorded & clearable
            status, hist_res = _http_request(f"{base}/api/history")
            self.assertEqual(status, 200)
            self.assertEqual(len(hist_res["history"]), 1)  # type: ignore[index]

            # Domain / unsupported intent errors return 422
            status, div_err = _http_request(
                f"{base}/api/chat",
                method="POST",
                payload={"message": "divide by zero"},
            )
            self.assertEqual(status, 422)
            self.assertFalse(div_err["ok"])  # type: ignore[index]

            status_unsup, unsup_err = _http_request(
                f"{base}/api/chat",
                method="POST",
                payload={"message": "sine of 30"},
            )
            self.assertEqual(status_unsup, 422)
            self.assertFalse(unsup_err["ok"])  # type: ignore[index]

            # Clear history
            status, clr_res = _http_request(f"{base}/api/clear", method="POST", payload={})
            self.assertEqual(status, 200)
            self.assertEqual(session.get_history(), [])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main(verbosity=2)

