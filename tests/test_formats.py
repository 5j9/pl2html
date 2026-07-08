import polars as pl

from pl2html.formats import fmt_currency, fmt_number, fmt_percent


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
            fmt_number(
                'value', decimals=1, compact=True, compact_system='financial'
            ).alias('financial'),
            fmt_number(
                'value', decimals=1, compact=True, compact_system='engineering'
            ).alias('engineering'),
        ]
    )

    # Assert
    expected_financial = ['0.0', '-950.0', '12.5K', '4.8M', '3.1B']
    expected_engineering = ['0.0', '-950.0', '12.5k', '4.8M', '3.1G']

    assert result['financial'].to_list() == expected_financial
    assert result['engineering'].to_list() == expected_engineering


def test_fmt_percent():
    """Verifies that fmt_percent scales values by 100 and appends % suffix."""
    # Arrange
    df = pl.DataFrame({'rates': [0.0523, -0.12, 1.0]})

    # Act
    result = df.select(
        [fmt_percent('rates', decimals=2, use_seps=False).alias('pct')]
    )

    # Assert
    expected = ['5.23%', '-12.00%', '100.00%']
    assert result['pct'].to_list() == expected


def test_fmt_currency():
    """Verifies currency symbols and accounting style formatting for negatives."""
    # Arrange
    df = pl.DataFrame({'amount': [1250.5, -450.0, 0.0]})

    # Act
    result = df.select(
        [
            fmt_currency(
                'amount',
                symbol='$',
                decimals=2,
                use_seps=True,
                accounting=True,
            ).alias('usd'),
            fmt_currency(
                'amount',
                symbol='€',
                decimals=0,
                use_seps=True,
                accounting=False,
            ).alias('eur'),
        ]
    )

    # Assert
    expected_usd = ['$1,250.50', '($450.00)', '$0.00']
    expected_eur = ['€1,251', '-€450', '€0']

    assert result['usd'].to_list() == expected_usd
    assert result['eur'].to_list() == expected_eur
