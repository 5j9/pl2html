from typing import Literal as _Literal

import polars as pl
from polars import (
    Expr as _Expr,
    col as _col,
    lit as _lit,
    when as _when,
)


def fmt_number(
    column: str,
    decimals: int = 2,
    scale_by: float = 1.0,
    compact: bool = False,
    compact_system: _Literal['financial', 'engineering'] = 'financial',
    use_seps: bool = True,
    accounting: bool = False,
    pattern: str = '{x}',
) -> _Expr:
    """
    Highly optimized, native Polars numeric formatter matching great_tables features.
    Runs entirely in the parallel Rust engine without dropping into Python row loops.
    """
    val = _col(column)
    if scale_by != 1.0:
        val = val * scale_by

    abs_val = val.abs()

    # 1. Extract compact scaling suffix chain if enabled
    if compact:
        log10_expr = (
            _when(abs_val > 0).then(abs_val.log10()).otherwise(_lit(0.0))
        )
        thousands_exponent = (log10_expr / 3.0).floor().cast(pl.Int32) * 3

        if compact_system == 'engineering':
            suffix_chain = (
                _when(thousands_exponent == 3)
                .then(_lit('k'))
                .when(thousands_exponent == 6)
                .then(_lit('M'))
                .when(thousands_exponent == 9)
                .then(_lit('G'))
                .when(thousands_exponent == 12)
                .then(_lit('T'))
                .otherwise(_lit(''))
            )
        else:
            suffix_chain = (
                _when(thousands_exponent == 3)
                .then(_lit('K'))
                .when(thousands_exponent == 6)
                .then(_lit('M'))
                .when(thousands_exponent == 9)
                .then(_lit('B'))
                .when(thousands_exponent == 12)
                .then(_lit('T'))
                .otherwise(_lit(''))
            )
        divisor = _lit(10.0).pow(thousands_exponent.cast(pl.Float64))
        val_scaled = val / divisor
    else:
        val_scaled = val
        suffix_chain = _lit('')

    # 2. Handle precision rounding and extract base components natively
    # Add a tiny epsilon away from zero to guarantee consistent standard rounding for halves (.5)
    epsilon = _when(val_scaled >= 0).then(_lit(1e-9)).otherwise(_lit(-1e-9))
    rounded = (val_scaled + epsilon).round(decimals)

    if decimals > 0:
        int_part = rounded.cast(pl.Int64).abs().cast(pl.String)

        # Extract fractional part cleanly from full string cast
        full_str = rounded.abs().cast(pl.String)
        frac_part = (
            _when(full_str.str.contains(r'\.'))
            .then(full_str.str.split('.').list.get(1).str.slice(0, decimals))
            .otherwise(_lit(''))
        ).str.pad_end(
            decimals, fill_char='0'
        )  # Right-pad trailing zeroes cleanly (e.g., .5 -> .50)

        if use_seps:
            int_part = (
                int_part.str.reverse()
                .str.replace_all(r'(\d{3})', r'${1},')
                .str.strip_suffix(',')
                .str.reverse()
            )

        base_num_str = int_part + _lit('.') + frac_part
    else:
        base_num_str = rounded.round(0).cast(pl.Int64).abs().cast(pl.String)
        if use_seps:
            base_num_str = (
                base_num_str.str.reverse()
                .str.replace_all(r'(\d{3})', r'${1},')
                .str.strip_suffix(',')
                .str.reverse()
            )

    base_num_str = base_num_str + suffix_chain
    # 3. Extract pattern pieces ahead of time to support proper financial wrapping
    # e.g., pattern="${x}" -> prefix="$", suffix=""
    prefix, suffix = '', ''
    if pattern != '{x}':
        parts = pattern.split('{x}')
        if len(parts) == 2:
            prefix, suffix = parts[0], parts[1]

    # 4. Handle Negative Layouts and cleanly inject pattern prefixes/suffixes
    if accounting:
        formatted_expr = (
            _when(val < 0)
            .then(
                _lit('(')
                + _lit(prefix)
                + base_num_str
                + _lit(suffix)
                + _lit(')')
            )
            .otherwise(_lit(prefix) + base_num_str + _lit(suffix))
        )
    else:
        formatted_expr = (
            _when(val < 0)
            .then(_lit('-') + _lit(prefix) + base_num_str + _lit(suffix))
            .otherwise(_lit(prefix) + base_num_str + _lit(suffix))
        )

    return formatted_expr


def fmt_percent(
    column: str, decimals: int = 2, use_seps: bool = True
) -> _Expr:
    """Formats a column as a percentage, multiplying by 100 automatically."""
    return fmt_number(
        column=column,
        decimals=decimals,
        scale_by=100.0,
        use_seps=use_seps,
        pattern='{x}%',
    )


def fmt_currency(
    column: str,
    symbol: str = '$',
    decimals: int = 2,
    use_seps: bool = True,
    accounting: bool = True,
) -> _Expr:
    """Formats a column as a localized currency value."""
    return fmt_number(
        column=column,
        decimals=decimals,
        use_seps=use_seps,
        accounting=accounting,
        pattern=f'{symbol}{{x}}',
    )
