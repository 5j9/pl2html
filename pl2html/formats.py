from collections.abc import Iterable as _Iterable
from functools import wraps as _wraps
from typing import Literal as _Literal

from polars import (
    Expr as _Expr,
    Float64 as _Float64,
    Int32 as _Int32,
    Int64 as _Int64,
    String as _String,
    col as _col,
    lit as _lit,
    when as _when,
)


# --- 0. Decorator to cleanly support single or multiple columns ---
def _multicolumn(func):
    @_wraps(func)
    def wrapper(*, columns: str | _Iterable[str], **kwargs):
        if isinstance(columns, str):
            return func(columns=columns, **kwargs)
        return [func(columns=col, **kwargs).alias(col) for col in columns]

    return wrapper


# --- 1. Formatter Implementation Block ---

_Columns = str | list[str]


@_multicolumn
def fmt_number(
    *,
    columns: _Columns,
    decimals: int = 2,
    scale_by: float = 1.0,
    compact: bool = False,
    compact_system: _Literal['financial', 'engineering'] = 'financial',
    use_seps: bool = True,
    accounting: bool = False,
    pattern: str = '{x}',
    n_sigfig: int | None = None,
) -> _Expr:
    """
    Highly optimized, native Polars numeric formatter matching great_tables features.
    Runs entirely in the parallel Rust engine without dropping into Python row loops.
    """
    val = _col(columns)
    if scale_by != 1.0:
        val = val * scale_by

    abs_val = val.abs()

    # 1. Extract compact scaling suffix chain if enabled
    if compact:
        log10_expr = (
            _when(abs_val > 0).then(abs_val.log10()).otherwise(_lit(0.0))
        )
        thousands_exponent = (log10_expr / 3.0).floor().cast(_Int32) * 3

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
        divisor = _lit(10.0).pow(thousands_exponent.cast(_Float64))
        val_scaled = val / divisor
    else:
        val_scaled = val
        suffix_chain = _lit('')

    # 2. Dynamic Precision Handling (Significant Figures vs Fixed Decimals)
    if n_sigfig is not None:
        if n_sigfig < 1:
            raise ValueError('n_sigfig must be a positive integer >= 1')

        # Round natively using Polars' built-in significant figures feature
        rounded = val_scaled.round_sig_figs(n_sigfig)
        abs_rounded = rounded.abs()

        # Determine the number of dynamic decimal places needed for padding per row
        log10_expr = (
            _when(abs_rounded > 0)
            .then(abs_rounded.log10())
            .otherwise(_lit(0.0))
        )
        # decimals required = n_sigfig - 1 - floor(log10(x))
        dynamic_decimals = (_lit(n_sigfig - 1) - log10_expr.floor()).cast(
            _Int32
        )
        # Ensure we don't try to pad negative decimal places for large numbers
        dynamic_decimals = (
            _when(dynamic_decimals < 0)
            .then(_lit(0))
            .otherwise(dynamic_decimals)
        )
    else:
        # Fallback to standard fixed decimal logic
        epsilon = (
            _when(val_scaled >= 0).then(_lit(1e-9)).otherwise(_lit(-1e-9))
        )
        rounded = (val_scaled + epsilon).round(decimals)
        dynamic_decimals = _lit(decimals)

    # 3. Component Extraction & Alignment
    int_part = rounded.cast(_Int64).abs().cast(_String)
    full_str = rounded.abs().cast(_String)

    raw_frac = (
        _when(full_str.str.contains(r'\.'))
        .then(full_str.str.split('.').list.get(1))
        .otherwise(_lit(''))
    )

    # Use native Python string multiplication inside _lit()
    pad_len = n_sigfig if n_sigfig is not None else decimals
    frac_part = (
        _when(dynamic_decimals > 0)
        .then((raw_frac + _lit('0' * pad_len)).str.slice(0, dynamic_decimals))
        .otherwise(_lit(''))
    )

    # 4. Constructing Base Number String
    if use_seps:
        int_part = (
            int_part.str.reverse()
            .str.replace_all(r'(\d{3})', r'${1},')
            .str.strip_suffix(',')
            .str.reverse()
        )

    base_num_str = (
        _when(dynamic_decimals > 0)
        .then(int_part + _lit('.') + frac_part)
        .otherwise(int_part)
    )
    base_num_str = base_num_str + suffix_chain

    prefix, suffix = '', ''
    if pattern != '{x}':
        parts = pattern.split('{x}')
        if len(parts) == 2:
            prefix, suffix = parts[0], parts[1]

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


@_multicolumn
def fmt_percent(
    *, columns: _Columns, decimals: int = 2, use_seps: bool = True
) -> _Expr:
    """Formats columns as percentages, multiplying by 100 automatically."""
    return fmt_number(
        columns=columns,
        decimals=decimals,
        scale_by=100.0,
        use_seps=use_seps,
        pattern='{x}%',
    )


@_multicolumn
def fmt_currency(
    *,
    columns: _Columns,
    symbol: str = '$',
    decimals: int = 2,
    use_seps: bool = True,
    accounting: bool = True,
) -> _Expr:
    """Formats columns as localized currency values."""
    return fmt_number(
        columns=columns,
        decimals=decimals,
        use_seps=use_seps,
        accounting=accounting,
        pattern=f'{symbol}{{x}}',
    )


@_multicolumn
def fmt_scientific(
    *,
    columns: _Columns,
    decimals: int = 2,
    scale_by: float = 1.0,
    exp_style: _Literal['x10n', 'e', 'E'] = 'x10n',
    pattern: str = '{x}',
    force_sign_m: bool = False,
    force_sign_n: bool = False,
) -> _Expr:
    """Highly optimized scientific notation formatter."""
    val = _col(columns)
    if scale_by != 1.0:
        val = val * scale_by

    abs_val = val.abs()
    log10_expr = _when(abs_val > 0).then(abs_val.log10()).otherwise(_lit(0.0))
    exponent = log10_expr.floor().cast(_Int32)
    exponent = _when(val == 0).then(_lit(0)).otherwise(exponent)

    divisor = _lit(10.0).pow(exponent.cast(_Float64))
    mantissa = val / divisor

    epsilon = _when(mantissa >= 0).then(_lit(1e-9)).otherwise(_lit(-1e-9))
    rounded_m = (mantissa + epsilon).round(decimals)

    if decimals > 0:
        int_m = rounded_m.cast(_Int64).abs().cast(_String)
        full_str_m = rounded_m.abs().cast(_String)
        frac_m = (
            _when(full_str_m.str.contains(r'\.'))
            .then(full_str_m.str.split('.').list.get(1).str.slice(0, decimals))
            .otherwise(_lit(''))
        ).str.pad_end(decimals, fill_char='0')
        m_str = int_m + _lit('.') + frac_m
    else:
        m_str = rounded_m.round(0).cast(_Int64).abs().cast(_String)

    m_str = (
        _when(val < 0)
        .then(_lit('-') + m_str)
        .when((_lit(force_sign_m)) & (val > 0))
        .then(_lit('+') + m_str)
        .otherwise(m_str)
    )

    n_str = exponent.abs().cast(_String)
    n_str = (
        _when(exponent < 0)
        .then(_lit('-') + n_str)
        .when((_lit(force_sign_n)) & (exponent > 0))
        .then(_lit('+') + n_str)
        .otherwise(
            _when(_lit(force_sign_n)).then(_lit('+') + n_str).otherwise(n_str)
        )
    )

    if exp_style == 'x10n':
        combined = m_str + _lit(' × 10^') + n_str
    elif exp_style == 'E':
        combined = m_str + _lit('E') + n_str
    else:
        combined = m_str + _lit('e') + n_str

    prefix, suffix = '', ''
    if pattern != '{x}':
        parts = pattern.split('{x}')
        if len(parts) == 2:
            prefix, suffix = parts[0], parts[1]

    return _lit(prefix) + combined + _lit(suffix)


@_multicolumn
def fmt_bytes(
    *,
    columns: _Columns,
    standard: _Literal['decimal', 'binary'] = 'decimal',
    decimals: int = 1,
    use_seps: bool = True,
    pattern: str = '{x}',
    force_sign: bool = False,
    incl_space: bool = True,
) -> _Expr:
    """Highly optimized native bytes formatter."""
    val = _col(columns)
    abs_val = val.abs()

    if standard == 'binary':
        base = 1024.0
        log_expr = (
            _when(abs_val > 0).then(abs_val.log(2.0)).otherwise(_lit(0.0))
        )
        exponent_idx = (log_expr / 10.0).floor().cast(_Int32)
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
    else:
        base = 1000.0
        log_expr = (
            _when(abs_val > 0).then(abs_val.log10()).otherwise(_lit(0.0))
        )
        exponent_idx = (log_expr / 3.0).floor().cast(_Int32)
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

    divisor = _lit(base).pow(exponent_idx.cast(_Float64))
    val_scaled = val / divisor

    epsilon = _when(val_scaled >= 0).then(_lit(1e-9)).otherwise(_lit(-1e-9))
    rounded = (val_scaled + epsilon).round(decimals)

    if decimals > 0:
        int_part = rounded.cast(_Int64).abs().cast(_String)
        full_str = rounded.abs().cast(_String)
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
        base_num_str = rounded.round(0).cast(_Int64).abs().cast(_String)
        if use_seps:
            base_num_str = (
                base_num_str.str.reverse()
                .str.replace_all(r'(\d{3})', r'${1},')
                .str.strip_suffix(',')
                .str.reverse()
            )

    base_num_str = (
        _when(val < 0)
        .then(_lit('-') + base_num_str)
        .when((_lit(force_sign)) & (val > 0))
        .then(_lit('+') + base_num_str)
        .otherwise(base_num_str)
    )

    space = _lit(' ') if incl_space else _lit('')
    combined = base_num_str + space + suffix_chain

    prefix, suffix = '', ''
    if pattern != '{x}':
        parts = pattern.split('{x}')
        if len(parts) == 2:
            prefix, suffix = parts[0], parts[1]

    return _lit(prefix) + combined + _lit(suffix)


@_multicolumn
def fmt_tf(
    *,
    columns: _Columns,
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
    """Highly optimized native boolean layout mapper."""
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
    final_true = true_val if true_val is not None else preset_true
    final_false = false_val if false_val is not None else preset_false

    base_expr = (
        _when(_col(columns))
        .then(_lit(final_true))
        .otherwise(_lit(final_false))
    )

    prefix, suffix = '', ''
    if pattern != '{x}':
        parts = pattern.split('{x}')
        if len(parts) == 2:
            prefix, suffix = parts[0], parts[1]

    formatted_expr = _lit(prefix) + base_expr + _lit(suffix)

    if na_val is not None:
        return (
            _when(_col(columns).is_null())
            .then(_lit(na_val))
            .otherwise(formatted_expr)
        )
    return (
        _when(_col(columns).is_null())
        .then(_lit(None))
        .otherwise(formatted_expr)
    )


@_multicolumn
def sub_missing(*, columns: _Columns, missing_text: str = '—') -> _Expr:
    return _col(columns).fill_null(_lit(missing_text))


@_multicolumn
def sub_zero(*, columns: _Columns, zero_text: str = '—') -> _Expr:
    return (
        _when(_col(columns) == 0)
        .then(_lit(zero_text))
        .otherwise(_col(columns))
    )


@_multicolumn
def fmt_integer(
    *,
    columns: _Columns,
    scale_by: float = 1.0,
    compact: bool = False,
    compact_system: _Literal['financial', 'engineering'] = 'financial',
    use_seps: bool = True,
    accounting: bool = False,
    pattern: str = '{x}',
) -> _Expr:
    """Highly optimized native Polars integer formatter wrapper."""
    return fmt_number(
        columns=columns,
        decimals=0,
        scale_by=scale_by,
        compact=compact,
        compact_system=compact_system,
        use_seps=use_seps,
        accounting=accounting,
        pattern=pattern,
    )
