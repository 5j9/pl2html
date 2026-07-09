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

__version__ = '0.2.1.dev1'


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
) -> _LazyFrame:
    """
    Compiles a Polars DataFrame safely into an HTML string layout.
    Accepts structural custom attributes mappings to handle layout modifications natively.
    """
    lf = df.lazy() if isinstance(df, _DataFrame) else df

    exclude_columns = exclude_columns or []
    attrs = attrs or {}

    schema = lf.collect_schema()
    visible_columns = [c for c in schema.names() if c not in exclude_columns]

    cell_expressions = []

    # 2. Map structural cell loops
    for c in visible_columns:
        cell_expressions.append(_build_cell_expr(c, schema[c], attrs))

    # 3. Concatenate columns horizontally into rows
    row_expr = _lit('    <tr>') + _concat_str(cell_expressions) + _lit('</tr>')

    # 4. Generate wrappers and frame the query graph
    html_header, html_footer = _build_html_skeleton(visible_columns)

    return lf.select(row_expr.alias('html_row')).select(
        (html_header + _col('html_row').str.join('\n') + html_footer).alias(
            'html_table'
        )
    )
