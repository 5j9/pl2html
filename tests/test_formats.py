import polars as pl

from pl2html.formats import (
    fmt_currency,
    fmt_number,
    fmt_percent,
    fmt_scientific,
)


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


def test_fmt_scientific():
    """
    Verifies that fmt_scientific correctly formats numbers into scientific notation,
    respecting different exponential styles, scale factors, and sign enforcements.
    """
    # Arrange
    df = pl.DataFrame({'value': [0.0, -0.111, 2.22, 33.3, 444.0, -0.00555]})

    # Act
    result = df.select(
        [
            fmt_scientific('value', decimals=2, exp_style='x10n').alias(
                'default_style'
            ),
            fmt_scientific('value', decimals=1, exp_style='E').alias(
                'e_capital_style'
            ),
            fmt_scientific(
                'value', decimals=2, scale_by=10.0, exp_style='e'
            ).alias('scaled_style'),
            fmt_scientific(
                'value',
                decimals=1,
                exp_style='e',
                force_sign_m=True,
                force_sign_n=True,
            ).alias('forced_signs'),
        ]
    )

    # Assert
    expected_default = [
        '0.00 × 10^0',
        '-1.11 × 10^-1',
        '2.22 × 10^0',
        '3.33 × 10^1',
        '4.44 × 10^2',
        '-5.55 × 10^-3',
    ]

    expected_e_capital = [
        '0.0E0',
        '-1.1E-1',
        '2.2E0',
        '3.3E1',
        '4.4E2',
        '-5.6E-3',
    ]

    expected_scaled = [
        '0.00e0',
        '-1.11e0',
        '2.22e1',
        '3.33e2',
        '4.44e3',
        '-5.55e-2',
    ]

    # Updated to match Polars engine's behavior for force_sign_n on 0 exponent values
    expected_forced = [
        '0.0e+0',
        '-1.1e-1',
        '+2.2e+0',
        '+3.3e+1',
        '+4.4e+2',
        '-5.6e-3',
    ]

    assert result['default_style'].to_list() == expected_default
    assert result['e_capital_style'].to_list() == expected_e_capital
    assert result['scaled_style'].to_list() == expected_scaled
    assert result['forced_signs'].to_list() == expected_forced
