from polars import (
    DataFrame as _DataFrame,
    DataType as _DataType,
    Expr as _Expr,
    LazyFrame as _LazyFrame,
    String as _String,
    col as _col,
    concat_str as _concat_str,
    lit as _lit,
    when as _when,
)


def _escape_expr(expr: _Expr) -> _Expr:
    """
    Escapes an expression's string representations for secure HTML compatibility.
    Using literal=True is critical; otherwise, the '&' character in the replacement
    string acts as a regex match expansion macro and corrupts the entities.
    """
    return (
        expr.cast(_String)
        .str.replace_all('&', '&amp;', literal=True)
        .str.replace_all('<', '&lt;', literal=True)
        .str.replace_all('>', '&gt;', literal=True)
        .str.replace_all('"', '&quot;', literal=True)
        .str.replace_all("'", '&#x27;', literal=True)
    )


def _format_integer(col_name: str) -> _Expr:
    """Adds thousands separators to integers using look-around-free logic."""
    return (
        _col(col_name)
        .fill_null(0)
        .cast(_String)
        # Polars regex engine doesn't support look-behinds (e.g., (?<=\d)(?=(\d{3})+$)).
        # We reverse the string, match chunks of 3 digits from the right, add commas,
        # and then reverse it back to normal.
        .str.reverse()
        .str.replace_all(r'\d{3}', '$0,')
        .str.reverse()
        .str.replace(r'^,', '')
        .str.replace(r'^-,', '-')
    )


def _format_float(col_name: str) -> _Expr:
    """Formats floats with 3 decimals and thousands separators."""
    rounded_str = _col(col_name).round(3).cast(_String)

    parts = rounded_str.str.split_exact('.', 1)
    int_part = parts.struct.field('field_0')
    dec_part = parts.struct.field('field_1')

    formatted_int = (
        int_part.str.reverse()
        .str.replace_all(r'\d{3}', '$0,')
        .str.reverse()
        .str.replace(r'^,', '')
        .str.replace(r'^-,', '-')
    )

    return (
        _when(dec_part.is_not_null() & (dec_part != ''))
        .then(formatted_int + _lit('.') + dec_part)
        .otherwise(formatted_int)
    )


def _build_cell_expr(
    col_name: str,
    dtype: _DataType,
    attrs: dict[str, dict[str, _Expr]],
) -> _Expr:
    """Resolves attribute expressions before user formatters are applied."""
    if dtype.is_integer():
        fmt_expr = _format_integer(col_name)
    elif dtype.is_float():
        fmt_expr = _format_float(col_name)
    else:
        fmt_expr = _escape_expr(_col(col_name))

    fmt_expr = fmt_expr.fill_null('')

    attr_expr_list = []
    if col_name in attrs:
        for attr_name, val_expr in attrs[col_name].items():
            val_str_expr = val_expr.cast(_String)

            compiled_attr = (
                _when(val_str_expr.is_not_null() & (val_str_expr != ''))
                .then(_lit(f' {attr_name}="') + val_str_expr + _lit('"'))
                .otherwise(_lit(''))
            )
            attr_expr_list.append(compiled_attr)

    if attr_expr_list:
        opening_td = _lit('<td') + _concat_str(attr_expr_list) + _lit('>')
    else:
        opening_td = _lit('<td>')

    return opening_td + fmt_expr + _lit('</td>')


def _build_html_skeleton(visible_columns: list[str]) -> tuple[_Expr, _Expr]:
    header_parts = ['<table>', '  <thead>\n    <tr>']
    for c in visible_columns:
        header_parts.append(f'      <th>{c}</th>')
    header_parts.append('    </tr>\n  </thead>\n  <tbody>')

    html_header = _lit('\n'.join(header_parts) + '\n')
    html_footer = _lit('\n  </tbody>\n</table>')
    return html_header, html_footer


def _resolve_attributes(
    lf: _LazyFrame,
    df: _DataFrame,
    attrs: dict[str, dict[str, _Expr]],
    visible_columns: list[str],
) -> tuple[_LazyFrame, dict[str, dict[str, _Expr]]]:
    """Compiles dynamic attribute expressions using the raw, unformatted numeric data."""
    attr_exprs = []
    attr_aliases = {}

    for col_name, attr_map in attrs.items():
        if col_name in visible_columns:
            for attr_name, expr in attr_map.items():
                alias_key = f'__attr_{col_name}_{attr_name}'
                attr_exprs.append(_escape_expr(expr).alias(alias_key))
                attr_aliases[(col_name, attr_name)] = alias_key

    if not attr_exprs:
        return lf, attrs

    # Evaluate dynamic style contexts on the unformatted DataFrame
    resolved_attrs_df = df.select(attr_exprs)

    # Re-inject styles as true native structural data streams
    lf = lf.with_columns(resolved_attrs_df)

    # Remap the attributes dictionary to look at the new column references
    new_attrs = {col: attr_map.copy() for col, attr_map in attrs.items()}

    for (col_name, attr_name), alias in attr_aliases.items():
        new_attrs[col_name][attr_name] = _col(alias)

    return lf, new_attrs


def _apply_user_formatters(
    lf: _LazyFrame, formatters: _Expr | list[_Expr] | None
) -> _LazyFrame:
    """Applies layout transformation modifications/conversions to columns."""
    if formatters is None:
        return lf
    if isinstance(formatters, _Expr):
        formatters = [formatters]
    return lf.with_columns(formatters)


def _compile_html(
    lf: _LazyFrame,
    visible_columns: list[str],
    attrs: dict[str, dict[str, _Expr]],
) -> str:
    """Constructs the raw structural HTML string from the resolved LazyFrame pipeline."""
    # NOTE: Recompute the schema after applying user formatters. Formatters may
    # cast numeric columns to String, and _build_cell_expr uses the current dtype
    # to decide whether default numeric formatting (e.g. .round()) should be
    # applied. Using the original schema would incorrectly attempt numeric
    # formatting on already-formatted string columns.
    current_schema = lf.collect_schema()

    cell_expressions = [
        _build_cell_expr(c, current_schema[c], attrs) for c in visible_columns
    ]

    row_expr = _lit('    <tr>') + _concat_str(cell_expressions) + _lit('</tr>')
    html_header, html_footer = _build_html_skeleton(visible_columns)

    return (
        lf.select(row_expr.alias('html_row'))
        .select(
            (
                html_header + _col('html_row').str.join('\n') + html_footer
            ).alias('html_table')
        )
        .collect()
        .item()
    )


def to_html(
    df: _DataFrame | _LazyFrame,
    *,
    attrs: dict[str, dict[str, _Expr]] | None = None,
    exclude_columns: list[str] | None = None,
    formatters: _Expr | list[_Expr] | None = None,
) -> str:
    """Compiles a Polars DataFrame safely into an HTML string layout.

    Accepts structural custom attribute mappings to handle layout modifications
    and cell styling natively.

    Args:
        df: The source Polars DataFrame or LazyFrame containing the data.
        attrs: A dictionary mapping column names to cell attributes (e.g., style,
            class). The inner dictionary values can be raw Polars expressions that
            evaluate dynamically based on the column values.
        exclude_columns: A list of column names to omit from the final rendered
            HTML table. See note below on structural execution context.
        formatters: A single Polars expression or a list of expressions used to
            format column values into display strings (e.g., using `fmt_number`).

    Note on Execution Order (formatters and exclude_columns):
        Both `formatters` and `exclude_columns` serve critical roles in preserving
        the numeric evaluation state of the DataFrame before HTML compilation:

        1. formatters: Usually, formatting expressions can be applied directly to
           the DataFrame via `.with_columns()` before calling `to_html`. However,
           doing so converts numeric columns into `String` types too early, causing
           downstream style calculations (e.g., string division errors) to fail.
           Passing them here guarantees formatting runs *after* style resolution.

        2. exclude_columns: Dropping intermediate or helper columns before passing
           the DataFrame to `to_html` removes the data context required by your
           `attrs` expressions. By using `exclude_columns`, helper columns remain
           fully accessible during the mathematical style resolution phase (Step 1),
           but are cleanly filtered out before generating the HTML structural matrix.

    Example:
        >>> import polars as pl
        >>> from pl2html import to_html
        >>>
        >>> # 'max_threshold' is a helper column required for styling but not rendering
        >>> df = pl.DataFrame({
        ...     "sprd": [10.0, 50.0, 100.0],
        ...     "max_threshold": [120.0, 120.0, 120.0]
        ... })
        >>>
        >>> # Style depends on the unrendered 'max_threshold' column
        >>> attrs = {"sprd": {"style": pl.col("sprd") / pl.col("max_threshold")}}
        >>>
        >>> # Safe execution: 'max_threshold' is preserved for math, then omitted from output
        >>> html = to_html(
        ...     df,
        ...     attrs=attrs,
        ...     exclude_columns=["max_threshold"]
        ... )
    """
    lf = df.lazy() if isinstance(df, _DataFrame) else df
    exclude_columns = exclude_columns or []
    attrs = attrs or {}

    # Setup the evaluation context data
    df_eager = lf.collect()
    lf = df_eager.lazy()

    visible_columns = [
        c for c in df_eager.schema.names() if c not in exclude_columns
    ]

    lf, attrs = _resolve_attributes(lf, df_eager, attrs, visible_columns)
    lf = _apply_user_formatters(lf, formatters)

    return _compile_html(lf, visible_columns, attrs)
