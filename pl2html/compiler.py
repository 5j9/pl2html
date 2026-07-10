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


def _escape_polars_string(col_name: str) -> _Expr:
    """
    Escapes a column's string representations for secure HTML compatibility.
    Because standard HTML entities use look-arounds or loops, we can fall back to
    a native character loop or custom mapper if extensive characters are used,
    but basic safety replaces characters sequentially.
    """
    return (
        _col(col_name)
        .cast(_String)
        .str.replace_all('&', '&amp;')
        .str.replace_all('<', '&lt;')
        .str.replace_all('>', '&gt;')
        .str.replace_all('"', '&quot;')
        .str.replace_all("'", '&#x27;')
    )


def _format_integer(col_name: str) -> _Expr:
    """Adds thousands separators to integers using look-around-free logic."""
    return (
        _col(col_name)
        .fill_null(0)
        .cast(_String)
        .str.reverse()
        .str.replace_all(r'\d{3}', '$0,')
        .str.reverse()
        .str.replace(r'^,', '')
        .str.replace(r'^-,', '-')
    )


def _format_float(col_name: str) -> _Expr:
    """Formats floats with 3 decimals and thousands separators."""
    rounded_str = _col(col_name).round(3).cast(_String)
    int_part = rounded_str.str.split_exact('.', 1).struct.field('field_0')
    dec_part = rounded_str.str.split_exact('.', 1).struct.field('field_1')

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
    """
    Applies secure auto-escaping or data formatting overrides, then dynamically
    compiles HTML attributes into a valid single opening <td> tag block.
    """
    # 1. Base Data Evaluation (Formatted and HTML Escaped safely)
    if dtype.is_integer():
        fmt_expr = _format_integer(col_name)
    elif dtype.is_float():
        fmt_expr = _format_float(col_name)
    else:
        fmt_expr = _escape_polars_string(col_name)

    fmt_expr = fmt_expr.fill_null('')

    # 2. Build Attributes Expression Natively (e.g., style="...", class="...")
    attr_expr_list = []
    if col_name in attrs:
        for attr_name, val_expr in attrs[col_name].items():
            # If the expression returns null or empty for a specific row, skip writing the attribute
            compiled_attr = (
                _when(val_expr.is_not_null() & (val_expr != ''))
                .then(
                    _lit(f' {attr_name}="')
                    + val_expr.cast(_String)
                    + _lit('"')
                )
                .otherwise(_lit(''))
            )
            attr_expr_list.append(compiled_attr)

    # Combine attribute strings together
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

    schema = lf.collect_schema()
    visible_columns = [c for c in schema.names() if c not in exclude_columns]

    # === STEP 1: RESOLVE MATH/STYLE EXPRESSIONS ON NUMERIC DATA FIRST ===
    style_selects = []
    style_maps = {}

    for col_name, styles in attrs.items():
        if col_name in visible_columns and 'style' in styles:
            expr_key = f'__style_{col_name}'
            style_selects.append(styles['style'].alias(expr_key))
            style_maps[col_name] = expr_key

    if style_selects:
        # Collect using the full numeric dataframe context (including excluded columns)
        df = lf.collect()
        lf = df.lazy()
        resolved_styles_df = df.select(style_selects)

        new_attrs = {}
        for col_name, styles in attrs.items():
            new_attrs[col_name] = styles.copy()
            if col_name in style_maps:
                new_attrs[col_name]['style'] = _lit(
                    resolved_styles_df.get_column(style_maps[col_name])
                )
        attrs = new_attrs

    # === STEP 2: APPLY FORMATTERS TO CONVERT COLUMNS TO STRINGS ===
    if formatters is not None:
        if isinstance(formatters, _Expr):
            formatters = [formatters]
        lf = lf.with_columns(formatters)

    # === STEP 3: BUILD HTML CELLS ===
    cell_expressions = []
    for c in visible_columns:
        cell_expressions.append(
            _build_cell_expr(c, lf.collect_schema()[c], attrs)
        )

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
