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


def _interpolate_segment(
    norm_val: _Expr,
    low_hex: str,
    high_hex: str,
    seg_start: float,
    segment_width: float,
) -> tuple[_Expr, _Expr, _Expr]:
    """Calculates the dynamic RGB expressions for a specific color segment."""
    # Parse hex strings into base integer components
    r1, g1, b1 = (
        int(low_hex[1:3], 16),
        int(low_hex[3:5], 16),
        int(low_hex[5:7], 16),
    )
    r2, g2, b2 = (
        int(high_hex[1:3], 16),
        int(high_hex[3:5], 16),
        int(high_hex[5:7], 16),
    )

    # Compute local advancement factor within this specific segment channel
    local_factor = (norm_val - seg_start) / segment_width

    # Linear interpolation formula applied directly to Polars Expressions
    r_expr = (_lit(r1) + (_lit(r2 - r1) * local_factor)).round(0).cast(_Int32)
    g_expr = (_lit(g1) + (_lit(g2 - g1) * local_factor)).round(0).cast(_Int32)
    b_expr = (_lit(b1) + (_lit(b2 - b1) * local_factor)).round(0).cast(_Int32)

    return r_expr, g_expr, b_expr


def _build_css_expression(
    r: _Expr, g: _Expr, b: _Expr, auto_contrast: bool
) -> _Expr:
    """Stitches RGB expressions into a complete background and foreground style string."""
    base_style = (
        _lit('background-color: rgb(')
        + r.cast(_String)
        + _lit(',')
        + g.cast(_String)
        + _lit(',')
        + b.cast(_String)
        + _lit(');')
    )

    if not auto_contrast:
        return base_style

    # WCAG Relative Luminance Formula matching standard text accessibility bounds
    luminance = (
        (0.2126 * (r / 255.0))
        + (0.7152 * (g / 255.0))
        + (0.0722 * (b / 255.0))
    )

    fg_color = (
        _when(luminance < 0.45)
        .then(_lit(' color: #FFFFFF;'))
        .otherwise(_lit(' color: #000000;'))
    )
    return base_style + fg_color


def data_color(
    column: str,
    palette: list[str],
    domain: tuple[float, float] | None = None,
    auto_contrast: bool = True,
) -> dict[str, dict[str, _Expr]]:
    """
    Generates a style attribute expression mapping values in a column to a hex color palette.
    If domain is None, it dynamically calculates min/max from the column at evaluation time.
    """
    if len(palette) < 2:
        raise ValueError(
            'Palette must contain at least 2 colors for interpolation.'
        )

    # 1. Evaluate Data Domain and Normalize Boundaries
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

    # 2. Build the Initial Segment Boundary Condition
    r, g, b = _interpolate_segment(
        norm_val, palette[0], palette[1], 0.0, segment_width
    )
    chain = _when(norm_val <= segment_width).then(
        _build_css_expression(r, g, b, auto_contrast)
    )

    # 3. Chain Remaining Segments Sequentially
    for i in range(1, num_segments):
        seg_start = i * segment_width
        seg_end = (i + 1) * segment_width

        r, g, b = _interpolate_segment(
            norm_val, palette[i], palette[i + 1], seg_start, segment_width
        )
        chain = chain.when(norm_val <= seg_end).then(
            _build_css_expression(r, g, b, auto_contrast)
        )

    return {column: {'style': chain.otherwise(_lit(''))}}


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
