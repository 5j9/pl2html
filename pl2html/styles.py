from matplotlib import colormaps as _colormaps
from matplotlib.colors import to_hex as _to_hex
from polars import (
    Expr as _Expr,
    Int32 as _Int32,
    String as _String,
    col as _col,
    lit as _lit,
    when as _when,
)


def data_color(
    column: str,
    palette: list[str],
    domain: tuple[float, float] | None = None,
    auto_contrast: bool = True,  # <-- Added parameter
) -> dict[str, dict[str, _Expr]]:
    """
    Generates a style attribute expression mapping values in a column to a hex color palette.
    If domain is None, it dynamically calculates min/max from the column at evaluation time.
    """
    if len(palette) < 2:
        raise ValueError(
            'Palette must contain at least 2 colors for interpolation.'
        )

    min_expr = _lit(domain[0]) if domain else _col(column).min()
    max_expr = _lit(domain[1]) if domain else _col(column).max()

    range_expr = max_expr - min_expr
    norm_val = (
        _when(range_expr == 0)
        .then(_lit(0.0))
        .otherwise((_col(column) - min_expr) / range_expr)
    )

    num_segments = len(palette) - 1
    segment_width = 1.0 / num_segments

    # --- Helper to turn RGB expressions into a safe CSS string ---
    def _build_css(r: _Expr, g: _Expr, b: _Expr) -> _Expr:
        base_style = (
            _lit('background-color: rgb(')
            + r.cast(_String)
            + _lit(',')
            + g.cast(_String)
            + _lit(',')
            + b.cast(_String)
            + _lit(');')
        )
        if auto_contrast:
            # Convert 0-255 RGB integers back to 0.0-1.0 scale for your formula
            r_norm = r / 255.0
            g_norm = g / 255.0
            b_norm = b / 255.0
            luminance = 0.2126 * r_norm + 0.7152 * g_norm + 0.0722 * b_norm

            # Append custom text color directly to the style string natively
            fg_color = (
                _when(luminance < 0.45)
                .then(_lit(' color: #FFFFFF;'))
                .otherwise(_lit(' color: #000000;'))
            )
            return base_style + fg_color
        return base_style

    # Initialize first segment
    first_r, first_g, first_b = (
        int(palette[0][1:3], 16),
        int(palette[0][3:5], 16),
        int(palette[0][5:7], 16),
    )
    first_local_factor = (norm_val - 0.0) / segment_width

    r_expr = (
        (
            _lit(first_r)
            + (_lit(int(palette[1][1:3], 16) - first_r) * first_local_factor)
        )
        .round(0)
        .cast(_Int32)
    )
    g_expr = (
        (
            _lit(first_g)
            + (_lit(int(palette[1][3:5], 16) - first_g) * first_local_factor)
        )
        .round(0)
        .cast(_Int32)
    )
    b_expr = (
        (
            _lit(first_b)
            + (_lit(int(palette[1][5:7], 16) - first_b) * first_local_factor)
        )
        .round(0)
        .cast(_Int32)
    )

    chain = _when(norm_val <= (1.0 / num_segments)).then(
        _build_css(r_expr, g_expr, b_expr)
    )

    # Loop over remaining segments
    for i in range(1, num_segments):
        low_color = palette[i]
        high_color = palette[i + 1]

        seg_start = i * segment_width
        seg_end = (i + 1) * segment_width

        r1, g1, b1 = (
            int(low_color[1:3], 16),
            int(low_color[3:5], 16),
            int(low_color[5:7], 16),
        )
        r2, g2, b2 = (
            int(high_color[1:3], 16),
            int(high_color[3:5], 16),
            int(high_color[5:7], 16),
        )

        local_factor = (norm_val - seg_start) / segment_width

        r_expr = (
            (_lit(r1) + (_lit(r2 - r1) * local_factor)).round(0).cast(_Int32)
        )
        g_expr = (
            (_lit(g1) + (_lit(g2 - g1) * local_factor)).round(0).cast(_Int32)
        )
        b_expr = (
            (_lit(b1) + (_lit(b2 - b1) * local_factor)).round(0).cast(_Int32)
        )

        chain = chain.when(norm_val <= seg_end).then(
            _build_css(r_expr, g_expr, b_expr)
        )

    style_expr = chain.otherwise(_lit(''))
    return {column: {'style': style_expr}}


def data_color_cmap(
    column: str,
    cmap_name: str,
    num_colors: int = 10,
    domain: tuple[float, float] | None = None,
    auto_contrast: bool = True,  # <-- Pass through
) -> dict[str, dict[str, _Expr]]:
    """Optional helper using matplotlib to sample named colormaps securely."""
    cmap = _colormaps[cmap_name]
    palette = [_to_hex(cmap(i / (num_colors - 1))) for i in range(num_colors)]

    return data_color(
        column, palette, domain=domain, auto_contrast=auto_contrast
    )
