import polars as pl

__version__ = '0.1.0'


def _escape_polars_string(col_name: str) -> pl.Expr:
    """
    Escapes a column's string representations for secure HTML compatibility.
    Because standard HTML entities use look-arounds or loops, we can fall back to
    a native character loop or custom mapper if extensive characters are used,
    but basic safety replaces characters sequentially.
    """
    return (
        pl.col(col_name)
        .cast(pl.String)
        .str.replace_all('&', '&amp;')
        .str.replace_all('<', '&lt;')
        .str.replace_all('>', '&gt;')
        .str.replace_all('"', '&quot;')
        .str.replace_all("'", '&#x27;')
    )


def _format_integer(col_name: str) -> pl.Expr:
    """Adds thousands separators to integers using look-around-free logic."""
    return (
        pl.col(col_name)
        .fill_null(0)
        .cast(pl.String)
        .str.reverse()
        .str.replace_all(r'\d{3}', '$0,')
        .str.reverse()
        .str.replace(r'^,', '')
        .str.replace(r'^-,', '-')
    )


def _format_float(col_name: str) -> pl.Expr:
    """Formats floats with 3 decimals and thousands separators."""
    rounded_str = pl.col(col_name).round(3).cast(pl.String)
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
        pl.when(dec_part.is_not_null() & (dec_part != ''))
        .then(formatted_int + pl.lit('.') + dec_part)
        .otherwise(formatted_int)
    )


def _build_cell_expr(
    col_name: str,
    dtype: pl.DataType,
    attrs: dict[str, dict[str, pl.Expr]],
) -> pl.Expr:
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
                pl.when(val_expr.is_not_null() & (val_expr != ''))
                .then(
                    pl.lit(f' {attr_name}="')
                    + val_expr.cast(pl.String)
                    + pl.lit('"')
                )
                .otherwise(pl.lit(''))
            )
            attr_expr_list.append(compiled_attr)

    # Combine attribute strings together
    if attr_expr_list:
        opening_td = (
            pl.lit('<td') + pl.concat_str(attr_expr_list) + pl.lit('>')
        )
    else:
        opening_td = pl.lit('<td>')

    return opening_td + fmt_expr + pl.lit('</td>')


def _build_html_skeleton(
    visible_columns: list[str], add_row_no: bool
) -> tuple[pl.Expr, pl.Expr]:
    header_parts = ['<table>', '  <thead>\n    <tr>']
    if add_row_no:
        header_parts.append('      <th>#</th>')
    for c in visible_columns:
        header_parts.append(f'      <th>{c}</th>')
    header_parts.append('    </tr>\n  </thead>\n  <tbody>')

    html_header = pl.lit('\n'.join(header_parts) + '\n')
    html_footer = pl.lit('\n  </tbody>\n</table>')
    return html_header, html_footer


def to_html(
    df: pl.DataFrame | pl.LazyFrame,
    *,
    attrs: dict[str, dict[str, pl.Expr]] | None = None,
    exclude_columns: list[str] | None = None,
    add_row_no: bool = True,
) -> pl.LazyFrame:
    """
    Compiles a Polars DataFrame safely into an HTML string layout.
    Accepts structural custom attributes mappings to handle layout modifications natively.
    """
    lf = df.lazy() if isinstance(df, pl.DataFrame) else df

    exclude_columns = exclude_columns or []
    attrs = attrs or {}

    schema = lf.collect_schema()
    visible_columns = [c for c in schema.names() if c not in exclude_columns]

    cell_expressions = []

    # 1. Row Index Element Processing
    if add_row_no:
        index_expr = (
            pl.lit('<td>')
            + (pl.arange(1, pl.len() + 1).cast(pl.String))
            + pl.lit('</td>')
        )
        cell_expressions.append(index_expr)

    # 2. Map structural cell loops
    for c in visible_columns:
        cell_expressions.append(_build_cell_expr(c, schema[c], attrs))

    # 3. Concatenate columns horizontally into rows
    row_expr = (
        pl.lit('    <tr>') + pl.concat_str(cell_expressions) + pl.lit('</tr>')
    )

    # 4. Generate wrappers and frame the query graph
    html_header, html_footer = _build_html_skeleton(
        visible_columns, add_row_no
    )

    return lf.select(row_expr.alias('html_row')).select(
        (html_header + pl.col('html_row').str.join('\n') + html_footer).alias(
            'html_table'
        )
    )
