from polars import DataFrame

from pl2html import to_html
from pl2html.styles import data_color
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

    actual_html = to_html(df, attrs=attrs_config).collect().item()
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
    actual_html = to_html(df, attrs=style_config).collect().item().strip()

    # Assert
    assert normalize_html(actual_html) == expected_html
