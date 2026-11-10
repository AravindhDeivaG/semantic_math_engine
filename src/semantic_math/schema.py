"""Pydantic schemas for constrained, nested mathematical operation trees."""

import math
from typing import Annotated, Any, Literal, Union

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    StrictFloat,
    StrictInt,
)
from typing_extensions import TypeAliasType


def _require_finite_number(value: Union[int, float]) -> Union[int, float]:
    """Ensure numeric operands are finite (reject NaN and +/-Infinity)."""
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Numeric operand must be finite.")
    return value


NumberValue = Annotated[
    Union[StrictInt, StrictFloat],
    AfterValidator(_require_finite_number),
]


class _BaseOperation(BaseModel):
    """Base model disallowing unexpected fields on operation nodes."""

    model_config = ConfigDict(extra="forbid")


def _variadic_operands_field() -> Any:
    """Shared Field definition for variadic operations (2+ operands)."""
    return Field(
        ...,
        min_length=2,
        description="List of 2 or more operands (numbers or nested operations).",
    )


class AddOperation(_BaseOperation):
    """Sum of two or more operands."""

    op: Literal["add"] = Field(..., description="Addition operation discriminator.")
    operands: list["Operand"] = _variadic_operands_field()


class MultiplyOperation(_BaseOperation):
    """Product of two or more operands."""

    op: Literal["multiply"] = Field(
        ..., description="Multiplication operation discriminator."
    )
    operands: list["Operand"] = _variadic_operands_field()


class SubtractOperation(_BaseOperation):
    """Difference of strictly two operands: left - right."""

    op: Literal["subtract"] = Field(
        ..., description="Subtraction operation discriminator."
    )
    left: "Operand" = Field(..., description="Minuend (value to subtract from).")
    right: "Operand" = Field(..., description="Subtrahend (value to subtract).")


class DivideOperation(_BaseOperation):
    """Quotient of strictly two operands: numerator / denominator."""

    op: Literal["divide"] = Field(..., description="Division operation discriminator.")
    numerator: "Operand" = Field(..., description="Dividend (numerator).")
    denominator: "Operand" = Field(..., description="Divisor (denominator).")


class PowerOperation(_BaseOperation):
    """Exponentiation of strictly two operands: base ** exponent."""

    op: Literal["power"] = Field(..., description="Power operation discriminator.")
    base: "Operand" = Field(..., description="Base expression.")
    exponent: "Operand" = Field(..., description="Exponent expression.")


class NthRootOperation(_BaseOperation):
    """Principal nth root of strictly two operands: radicand ** (1 / n)."""

    op: Literal["nth_root"] = Field(
        ..., description="Nth-root operation discriminator."
    )
    radicand: "Operand" = Field(
        ..., description="Value inside the radical to take the root of."
    )
    n: "Operand" = Field(
        ...,
        description="Root degree (e.g., 2 for square root, 3 for cube root).",
    )


OperationExpr = Annotated[
    Union[
        AddOperation,
        SubtractOperation,
        MultiplyOperation,
        DivideOperation,
        PowerOperation,
        NthRootOperation,
    ],
    Field(discriminator="op"),
]

Operand = TypeAliasType(
    "Operand",
    Annotated[
        Union[NumberValue, OperationExpr],
        Field(description="Either a finite number or a nested operation expression."),
    ],
)


class MathExpression(RootModel[Operand]):
    """Root expression schema validating a numeric literal or nested operation tree."""


for _model in (
    AddOperation,
    MultiplyOperation,
    SubtractOperation,
    DivideOperation,
    PowerOperation,
    NthRootOperation,
    MathExpression,
):
    _model.model_rebuild()


def get_expression_json_schema() -> dict[str, Any]:
    """Return the complete JSON schema dictionary for a nested math expression."""
    return MathExpression.model_json_schema()
