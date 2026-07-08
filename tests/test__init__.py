import polars as pl

from pl2html import to_html
from tests import normalize_html


def test_basic_table_compilation(expected_html):
    # This automatically loads: html_fixtures/test_basic_table_compilation.html
    df = pl.DataFrame({'id': [1, 2], 'name': ['Alice', 'Bob']})

    actual_html = to_html(df, add_row_no=False).collect().item().strip()
    assert normalize_html(actual_html) == expected_html


def test_integer_thousands_separator(expected_html):
    # This automatically loads: html_fixtures/test_integer_thousands_separator.html
    df = pl.DataFrame({'large_numbers': [1000, 1000000]})

    actual_html = to_html(df, add_row_no=False).collect().item().strip()
    assert normalize_html(actual_html) == expected_html


def test_float_rounding_and_formatting(expected_html):
    """
    Verifies that float columns are natively rounded to 3 decimal places
    and nulls within float columns are safely converted to empty cells.
    """
    # Arrange: Setup a mix of varied decimal lengths, whole floats, and null values
    df = pl.DataFrame(
        {
            'metric_a': [1.123456, 2.5, None],
            'metric_b': [0.0001, -12.34567, 100.0],
        }
    )

    # Act: Generate the lazy HTML tree and collect the string result
    actual_html = to_html(df, add_row_no=False).collect().item().strip()

    assert normalize_html(actual_html) == expected_html


def test_large_float_thousands_separator(expected_html):
    """
    Verifies that float columns containing huge numbers are formatted with
    thousands separators for their integer parts while preserving exactly
    3 decimal places.
    """
    # Arrange: Setup giant floats, negative giant floats, and clean whole numbers
    df = pl.DataFrame(
        {
            'big_metrics': [
                1234567.89123,  # Should become 1,234,567.891
                -987654321.0,  # Should become -987,654,321.0
                1000.4,  # Should become 1,000.4
            ]
        }
    )

    # Act
    actual_html = to_html(df, add_row_no=False).collect().item().strip()
    assert normalize_html(actual_html) == expected_html


def test_format_overrides_custom_expressions(expected_html):
    """
    Verifies that formatting values and injecting custom HTML attributes (like styles/classes)
    are handled independently using the format_overrides and attrs parameters.
    """
    # Arrange: Setup a basic dataset with a score and a trend column
    df = pl.DataFrame(
        {
            'symbol': ['AAPL', 'TSLA'],
            'score': [85, 42],
            'status': ['Up', 'Down'],
        }
    )

    # 2. attrs ONLY handles HTML tag customizations (clean, zero-tag boilerplate)
    custom_attrs = {
        'score': {
            'class': pl.lit('score-cell'),
            'style': (
                pl.when(pl.col('score') >= 50)
                .then(pl.lit('background-color: #00FF00;'))
                .otherwise(pl.lit('background-color: #FF0000;'))
            ),
        }
    }

    # Act
    actual_html = (
        to_html(
            df,
            attrs=custom_attrs,
            add_row_no=False,
        )
        .collect()
        .item()
        .strip()
    )

    assert normalize_html(actual_html) == expected_html
