from typing import Literal as _Literal

from polars import (
    Expr as _Expr,
    Float64 as _Float64,
    Int32 as _Int32,
    String as _String,
    col as _col,
    lit as _lit,
    when as _when,
)


def format_compact(
    column: str,
    decimals: int = 2,
    system: _Literal['financial', 'engineering'] = 'financial',
) -> _Expr:
    """
    Formats numeric values into compact notation using native Polars expressions.

    Supported Systems:
      - "financial":   K (Thousand), M (Million), B (Billion), T (Trillion)
      - "engineering": k (kilo),     M (Mega),    G (Giga),    T (Tera)
    """
    val = _col(column)
    abs_val = val.abs()

    # 1. Compute the dynamic scale factor floor(log10(|x|) / 3) * 3
    log10_expr = _when(abs_val > 0).then(abs_val.log10()).otherwise(_lit(0.0))
    thousands_exponent = (log10_expr.floor() // 3 * 3).cast(_Int32)

    # 2. Define the suffix mappings based on the chosen system
    if system == 'engineering':
        suffix_chain = (
            _when(thousands_exponent == 3)
            .then(_lit('k'))  # lowercase k for kilo
            .when(thousands_exponent == 6)
            .then(_lit('M'))  # Mega
            .when(thousands_exponent == 9)
            .then(_lit('G'))  # Giga
            .when(thousands_exponent == 12)
            .then(_lit('T'))  # Tera
            .otherwise(_lit(''))
        )
    else:  # default to "financial"
        suffix_chain = (
            _when(thousands_exponent == 3)
            .then(_lit('K'))  # Thousand
            .when(thousands_exponent == 6)
            .then(_lit('M'))  # Million
            .when(thousands_exponent == 9)
            .then(_lit('B'))  # Billion
            .when(thousands_exponent == 12)
            .then(_lit('T'))  # Trillion
            .otherwise(_lit(''))
        )

    # 3. Scale the value down: scaled = val / (10 ** thousands_exponent)
    divisor = _lit(10.0).pow(thousands_exponent.cast(_Float64))
    scaled_val = val / divisor

    # 4. Round and stitch into the final string
    return scaled_val.round(decimals).cast(_String) + suffix_chain
