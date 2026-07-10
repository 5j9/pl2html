from datetime import date, datetime

from polars import DataFrame, Float64, String, col, lit, when

from pl2html.compiler import to_html
from pl2html.formats import fmt_number
from tests import normalize_html


def test_basic_table_compilation(expected_html):
    # This automatically loads: html_fixtures/test_basic_table_compilation.html
    df = DataFrame({'id': [1, 2], 'name': ['Alice', 'Bob']})

    actual_html = to_html(df)
    assert normalize_html(actual_html) == expected_html


def test_integer_thousands_separator(expected_html):
    # This automatically loads: html_fixtures/test_integer_thousands_separator.html
    df = DataFrame({'large_numbers': [1000, 1000000]})

    actual_html = to_html(df)
    assert normalize_html(actual_html) == expected_html


def test_float_rounding_and_formatting(expected_html):
    """
    Verifies that float columns are natively rounded to 3 decimal places
    and nulls within float columns are safely converted to empty cells.
    """
    # Arrange: Setup a mix of varied decimal lengths, whole floats, and null values
    df = DataFrame(
        {
            'metric_a': [1.123456, 2.5, None],
            'metric_b': [0.0001, -12.34567, 100.0],
        }
    )

    # Act: Generate the lazy HTML tree and collect the string result
    actual_html = to_html(df)

    assert normalize_html(actual_html) == expected_html


def test_large_float_thousands_separator(expected_html):
    """
    Verifies that float columns containing huge numbers are formatted with
    thousands separators for their integer parts while preserving exactly
    3 decimal places.
    """
    # Arrange: Setup giant floats, negative giant floats, and clean whole numbers
    df = DataFrame(
        {
            'big_metrics': [
                1234567.89123,  # Should become 1,234,567.891
                -987654321.0,  # Should become -987,654,321.0
                1000.4,  # Should become 1,000.4
            ]
        }
    )

    # Act
    actual_html = to_html(df)
    assert normalize_html(actual_html) == expected_html


def test_format_overrides_custom_expressions(expected_html):
    """
    Verifies that formatting values and injecting custom HTML attributes (like styles/classes)
    are handled independently using the format_overrides and attrs parameters.
    """
    # Arrange: Setup a basic dataset with a score and a trend column
    df = DataFrame(
        {
            'symbol': ['AAPL', 'TSLA'],
            'score': [85, 42],
            'status': ['Up', 'Down'],
        }
    )

    # 2. attrs ONLY handles HTML tag customizations (clean, zero-tag boilerplate)
    custom_attrs = {
        'score': {
            'class': lit('score-cell'),
            'style': (
                when(col('score') >= 50)
                .then(lit('background-color: #00FF00;'))
                .otherwise(lit('background-color: #FF0000;'))
            ),
        }
    }

    # Act
    actual_html = to_html(
        df,
        attrs=custom_attrs,
    )

    assert normalize_html(actual_html) == expected_html


def test_temporal_columns_handling(expected_html):
    """
    Verifies that Date and Datetime columns are cleanly handled,
    even when pre-formatted as strings or when left to the default
    string conversion fallback.
    """
    # Arrange: Create a DataFrame with native Date, Datetime, and pre-formatted strings
    df = DataFrame(
        {
            'event_date': [date(2026, 1, 1), date(2026, 12, 31)],
            'timestamp': [
                datetime(2026, 7, 8, 12, 0, 0),
                datetime(2026, 7, 8, 23, 59, 59),
            ],
            # Simulating how a user might pre-format custom date formats themselves
            'custom_format': [
                datetime(2026, 5, 15, 14, 30).strftime('%B %d, %Y'),
                datetime(2026, 10, 31, 18, 0).strftime('%B %d, %Y'),
            ],
        }
    )

    # Act
    actual_html = to_html(df)

    assert normalize_html(actual_html) == expected_html


def test_to_html_evaluates_styles_before_formatters(expected_html):
    # 1. Setup a numeric dataset where computing a rank ratio
    # requires dividing float values.
    df = DataFrame(
        {'sprd': [10.0, 50.0, 100.0], 'label': ['A', 'B', 'C']},
        schema={'sprd': Float64, 'label': String},
    )

    # 2. Simulate what rank_color does under the hood:
    # It builds an expression that performs math/division on the column.
    # If the column becomes a String too early, this division will throw an InvalidOperationError.
    mock_rank_style_expr = lit('font-size: ') + (
        col('sprd') / col('sprd').max()
    ).cast(
        String
    )  # Convert resulting scalar or weight to a dummy style string

    attrs = {'sprd': {'style': mock_rank_style_expr}}

    # 3. Generate the formatting expression that converts the float to a string
    formatting_exprs = fmt_number(columns=['sprd'], decimals=2)
    formatting_exprs = fmt_number(columns=['sprd'], decimals=2)

    # Act
    # This should process the math inside `attrs` first using the numeric data,
    # then apply the formatting expressions, and finally output valid HTML.
    assert (
        normalize_html(to_html(df, attrs=attrs, formatters=formatting_exprs))
        == expected_html
    )


def test_dynamic_attributes_preserve_numeric_context(expected_html):
    """
    Ensures ALL attribute expressions (not just 'style') are resolved against
    the raw numeric data context before text formatters distort the types.
    """
    # 1. Create a dataset where an attribute relies on math operations
    df = DataFrame(
        {'price': [100.0, 250.0, 50.0], 'benchmark': [150.0, 150.0, 150.0]}
    )

    # 2. Map expressions to non-style attributes ('class' and 'data-sort')
    # that depend on un-formatted, numeric relationships
    attrs = {
        'price': {
            'class': (
                when(col('price') > col('benchmark'))
                .then(lit('expensive'))
                .otherwise(lit('normal'))
            ),
            'data-sort': col('price') / col('benchmark'),
        }
    }

    # 3. Apply a formatter that converts 'price' to a String.
    # On the old codebase, this causes Step 1 to skip evaluation (since it's not "style"),
    # and Step 3 will panic attempting to divide String / Float64.
    actual_html = to_html(
        df,
        attrs=attrs,
        exclude_columns=['benchmark'],
        formatters=fmt_number(columns=['price'], decimals=2),
    )

    assert normalize_html(actual_html) == expected_html
