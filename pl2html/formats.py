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


def fmt_scientific(
    column: str,
    decimals: int = 2,
    scale_by: float = 1.0,
    exp_style: _Literal['x10n', 'e', 'E'] = 'x10n',
    pattern: str = '{x}',
    force_sign_m: bool = False,
    force_sign_n: bool = False,
) -> _Expr:
    """
    Highly optimized, native Polars scientific notation formatter matching great_tables features.
    Runs entirely in the parallel Rust engine without dropping into Python row loops.
    """
    val = _col(column)
    if scale_by != 1.0:
        val = val * scale_by

    abs_val = val.abs()

    # 1. Calculate the exponent and mantissa using log10
    log10_expr = _when(abs_val > 0).then(abs_val.log10()).otherwise(_lit(0.0))
    # Floor to get the correct integer exponent
    exponent = log10_expr.floor().cast(pl.Int32)

    # Handle the exact zero edge case gracefully
    exponent = _when(val == 0).then(_lit(0)).otherwise(exponent)

    divisor = _lit(10.0).pow(exponent.cast(pl.Float64))
    mantissa = val / divisor

    # 2. Format the Mantissa (m) using your decimal padding logic
    epsilon = _when(mantissa >= 0).then(_lit(1e-9)).otherwise(_lit(-1e-9))
    rounded_m = (mantissa + epsilon).round(decimals)

    if decimals > 0:
        int_m = rounded_m.cast(pl.Int64).abs().cast(pl.String)
        full_str_m = rounded_m.abs().cast(pl.String)
        frac_m = (
            _when(full_str_m.str.contains(r'\.'))
            .then(full_str_m.str.split('.').list.get(1).str.slice(0, decimals))
            .otherwise(_lit(''))
        ).str.pad_end(decimals, fill_char='0')

        m_str = int_m + _lit('.') + frac_m
    else:
        m_str = rounded_m.round(0).cast(pl.Int64).abs().cast(pl.String)

    # Apply Mantissa signs
    m_str = (
        _when(val < 0)
        .then(_lit('-') + m_str)
        .when((_lit(force_sign_m)) & (val > 0))
        .then(_lit('+') + m_str)
        .otherwise(m_str)
    )

    # 3. Format the Exponent (n)
    n_str = exponent.abs().cast(pl.String)
    n_str = (
        _when(exponent < 0)
        .then(_lit('-') + n_str)
        .when((_lit(force_sign_n)) & (exponent > 0))
        .then(_lit('+') + n_str)
        .otherwise(
            _when(_lit(force_sign_n)).then(_lit('+') + n_str).otherwise(n_str)
        )
    )

    # 4. Combine based on exp_style
    if exp_style == 'x10n':
        # Matching Great Tables standard: ' × 10^n'
        combined = m_str + _lit(' × 10^') + n_str
    elif exp_style == 'E':
        combined = m_str + _lit('E') + n_str
    else:
        combined = m_str + _lit('e') + n_str

    # 5. Extract and apply pattern decorations
    prefix, suffix = '', ''
    if pattern != '{x}':
        parts = pattern.split('{x}')
        if len(parts) == 2:
            prefix, suffix = parts[0], parts[1]

    return _lit(prefix) + combined + _lit(suffix)


def fmt_bytes(
    column: str,
    standard: _Literal['decimal', 'binary'] = 'decimal',
    decimals: int = 1,
    use_seps: bool = True,
    pattern: str = '{x}',
    force_sign: bool = False,
    incl_space: bool = True,
) -> _Expr:
    """
    Highly optimized, native Polars byte formatter matching great_tables features.
    Runs entirely in the parallel Rust engine without dropping into Python row loops.
    """
    val = _col(column)
    abs_val = val.abs()

    # 1. Determine base, step, and suffixes based on standard
    if standard == 'binary':
        base = 1024.0
        # Calculate log2 range divided by 10 to find step index (0=B, 1=KiB, 2=MiB...)
        log_expr = (
            _when(abs_val > 0).then(abs_val.log(2.0)).otherwise(_lit(0.0))
        )
        exponent_idx = (log_expr / 10.0).floor().cast(pl.Int32)
        # Prevent index out of bounds for values smaller than 1 byte
        exponent_idx = (
            _when(exponent_idx < 0).then(_lit(0)).otherwise(exponent_idx)
        )

        suffix_chain = (
            _when(exponent_idx == 0)
            .then(_lit('B'))
            .when(exponent_idx == 1)
            .then(_lit('KiB'))
            .when(exponent_idx == 2)
            .then(_lit('MiB'))
            .when(exponent_idx == 3)
            .then(_lit('GiB'))
            .when(exponent_idx == 4)
            .then(_lit('TiB'))
            .when(exponent_idx == 5)
            .then(_lit('PiB'))
            .otherwise(_lit('B'))
        )
    else:  # decimal standard
        base = 1000.0
        log_expr = (
            _when(abs_val > 0).then(abs_val.log10()).otherwise(_lit(0.0))
        )
        exponent_idx = (log_expr / 3.0).floor().cast(pl.Int32)
        exponent_idx = (
            _when(exponent_idx < 0).then(_lit(0)).otherwise(exponent_idx)
        )

        suffix_chain = (
            _when(exponent_idx == 0)
            .then(_lit('B'))
            .when(exponent_idx == 1)
            .then(_lit('kB'))
            .when(exponent_idx == 2)
            .then(_lit('MB'))
            .when(exponent_idx == 3)
            .then(_lit('GB'))
            .when(exponent_idx == 4)
            .then(_lit('TB'))
            .when(exponent_idx == 5)
            .then(_lit('PB'))
            .otherwise(_lit('B'))
        )

    # 2. Scale the value natively
    divisor = _lit(base).pow(exponent_idx.cast(pl.Float64))
    val_scaled = val / divisor

    # 3. Format precision and pad trailing decimals
    epsilon = _when(val_scaled >= 0).then(_lit(1e-9)).otherwise(_lit(-1e-9))
    rounded = (val_scaled + epsilon).round(decimals)

    if decimals > 0:
        int_part = rounded.cast(pl.Int64).abs().cast(pl.String)
        full_str = rounded.abs().cast(pl.String)
        frac_part = (
            _when(full_str.str.contains(r'\.'))
            .then(full_str.str.split('.').list.get(1).str.slice(0, decimals))
            .otherwise(_lit(''))
        ).str.pad_end(decimals, fill_char='0')

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

    # 4. Attach sign flags
    base_num_str = (
        _when(val < 0)
        .then(_lit('-') + base_num_str)
        .when((_lit(force_sign)) & (val > 0))
        .then(_lit('+') + base_num_str)
        .otherwise(base_num_str)
    )

    # 5. Append unit string with or without separation space
    space = _lit(' ') if incl_space else _lit('')
    combined = base_num_str + space + suffix_chain

    # 6. Apply pattern mask
    prefix, suffix = '', ''
    if pattern != '{x}':
        parts = pattern.split('{x}')
        if len(parts) == 2:
            prefix, suffix = parts[0], parts[1]

    return _lit(prefix) + combined + _lit(suffix)


def fmt_tf(
    column: str,
    tf_style: _Literal[
        'true-false',
        'yes-no',
        'up-down',
        'check-mark',
        'circles',
        'squares',
        'diamonds',
        'arrows',
        'triangles',
        'triangles-lr',
    ] = 'true-false',
    pattern: str = '{x}',
    true_val: str | None = None,
    false_val: str | None = None,
    na_val: str | None = None,
) -> _Expr:
    """
    Highly optimized, native Polars boolean formatter matching great_tables features.
    Maps booleans to text/symbols, overrides via custom labels, and handles null values safely.
    """
    # 1. Map tf_style presets
    style_map = {
        'true-false': ('true', 'false'),
        'yes-no': ('yes', 'no'),
        'up-down': ('up', 'down'),
        'check-mark': ('✓', '✗'),
        'circles': ('●', '○'),
        'squares': ('■', '□'),
        'diamonds': ('◆', '◇'),
        'arrows': ('↑', '↓'),
        'triangles': ('▲', '▼'),
        'triangles-lr': ('▶', '◀'),
    }

    preset_true, preset_false = style_map.get(tf_style, ('true', 'false'))

    # 2. Apply explicit user overrides if provided
    final_true = true_val if true_val is not None else preset_true
    final_false = false_val if false_val is not None else preset_false

    # 3. Form conditional mapping expression
    base_expr = (
        _when(_col(column)).then(_lit(final_true)).otherwise(_lit(final_false))
    )

    # 4. Handle pattern mapping
    prefix, suffix = '', ''
    if pattern != '{x}':
        parts = pattern.split('{x}')
        if len(parts) == 2:
            prefix, suffix = parts[0], parts[1]

    formatted_expr = _lit(prefix) + base_expr + _lit(suffix)

    # 5. Handle missing value overrides safely (nulls shouldn't get pattern prefixes/suffixes)
    if na_val is not None:
        return (
            _when(_col(column).is_null())
            .then(_lit(na_val))
            .otherwise(formatted_expr)
        )
    else:
        return (
            _when(_col(column).is_null())
            .then(_lit(None))
            .otherwise(formatted_expr)
        )


def sub_missing(column: str, missing_text: str = '—') -> _Expr:
    return _col(column).fill_null(_lit(missing_text))


def sub_zero(column: str, zero_text: str = '—') -> _Expr:
    return (
        _when(_col(column) == 0).then(_lit(zero_text)).otherwise(_col(column))
    )


def fmt_integer(
    column: str,
    scale_by: float = 1.0,
    compact: bool = False,
    compact_system: _Literal['financial', 'engineering'] = 'financial',
    use_seps: bool = True,
    accounting: bool = False,
    pattern: str = '{x}',
) -> _Expr:
    """
    Highly optimized, native Polars integer formatter.
    Wraps fmt_number forcing decimals=0 to keep rendering purely in Rust.
    """
    return fmt_number(
        column=column,
        decimals=0,
        scale_by=scale_by,
        compact=compact,
        compact_system=compact_system,
        use_seps=use_seps,
        accounting=accounting,
        pattern=pattern,
    )
