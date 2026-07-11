from polars import DataFrame

from pl2html.compiler import to_html
from pl2html.styles import data_color, rank_color
from tests import normalize_html


def test_styled_output_generation(expected_html):
    df = DataFrame(
        {'amount': [10.0, 50.0, 100.0], 'category': ['A', 'B', 'C']}
    )

    # Generate style configurations using the new helpers
    amount_styles = data_color(
        column='amount',
        palette=['#ff7675', '#fdcb6e', '#00b894'],  # Red -> Yellow -> Green
    )

    # If they want matplotlib wrappers instead:
    # amount_styles = data_color_cmap(column="amount", cmap_name="RdYlGn")

    # Combine configurations cleanly
    attrs_config = {}
    attrs_config.update(amount_styles)

    actual_html = to_html(df, attrs=attrs_config)
    assert normalize_html(actual_html) == expected_html


def test_data_color_auto_contrast_switching(expected_html):
    """
    Verifies that auto_contrast correctly injects 'color: #FFFFFF;' for dark
    backgrounds and 'color: #000000;' for light backgrounds natively.
    """
    # Arrange: Use a polar dark-to-light palette (Black to White)
    df = DataFrame({'value': [0.0, 100.0]})

    # 0.0 should hit rgb(0,0,0) -> low luminance -> white text
    # 100.0 should hit rgb(255,255,255) -> high luminance -> black text
    style_config = data_color(
        column='value', palette=['#000000', '#FFFFFF'], auto_contrast=True
    )

    # Act
    actual_html = to_html(df, attrs=style_config)

    # Assert
    assert normalize_html(actual_html) == expected_html


def test_rank_color_interpolation(expected_html):
    """
    Verifies that rank_color maps styles based on relative ordinal rank
    rather than absolute value distance, utilizing continuous interpolation.
    """
    # Arrange: 10, 11, and 9000 are unevenly distributed,
    # but their ordinal ranks are evenly spaced: 0, 1, 2.
    df = DataFrame({'metric': [10.0, 11.0, 9000.0]})

    # Rank 0 (10.0) -> rgb(0,0,0) [Black]
    # Rank 1 (11.0) -> rgb(128,128,128) [Gray / Midpoint]
    # Rank 2 (9000.0) -> rgb(255,255,255) [White]
    style_config = rank_color(
        column='metric', palette=['#000000', '#FFFFFF'], auto_contrast=True
    )

    # Act
    actual_html = to_html(df, attrs=style_config)

    # Assert
    assert normalize_html(actual_html) == expected_html


def test_data_color_nan_values(expected_html):
    """
    NaN values should not cause rendering to fail.
    Cells containing NaN simply receive no background style.
    """
    df = DataFrame(
        {
            'metric': [1.0, float('nan'), 2.0, float('nan'), 3.0],
        }
    )

    style_config = data_color(
        column='metric',
        palette=['#94B894', '#FDCB6E', '#FDBF75'],
    )

    actual_html = to_html(df, attrs=style_config)

    assert normalize_html(actual_html) == expected_html
