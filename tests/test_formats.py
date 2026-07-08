import polars as pl

from pl2html.formats import format_compact


def test_format_compact_systems():
    """
    Verifies that format_compact correctly applies metric scales
    and splits suffixes between financial (K, M, B) and engineering (k, M, G).
    """
    # Arrange
    df = pl.DataFrame(
        {'value': [0.0, -950.0, 12500.0, 4800000.0, 3100000000.0]}
    )

    # Act
    result = df.select(
        [
            format_compact('value', decimals=1, system='financial').alias(
                'financial'
            ),
            format_compact('value', decimals=1, system='engineering').alias(
                'engineering'
            ),
        ]
    )

    # Assert
    expected_financial = ['0.0', '-950.0', '12.5K', '4.8M', '3.1B']
    expected_engineering = ['0.0', '-950.0', '12.5k', '4.8M', '3.1G']

    assert result['financial'].to_list() == expected_financial
    assert result['engineering'].to_list() == expected_engineering
