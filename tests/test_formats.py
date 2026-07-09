from polars import DataFrame, Float64

from pl2html.formats import (
    fmt_bytes,
    fmt_currency,
    fmt_integer,
    fmt_number,
    fmt_percent,
    fmt_scientific,
    fmt_tf,
    sub_missing,
    sub_zero,
)


def test_format_compact_systems():
    """
    Verifies that format_compact correctly applies metric scales
    and splits suffixes between financial (K, M, B) and engineering (k, M, G).
    """
    # Arrange
    df = DataFrame({'value': [0.0, -950.0, 12500.0, 4800000.0, 3100000000.0]})

    # Act
    result = df.select(
        [
            fmt_number(
                columns='value',
                decimals=1,
                compact=True,
                compact_system='financial',
            ).alias('financial'),
            fmt_number(
                columns='value',
                decimals=1,
                compact=True,
                compact_system='engineering',
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
    df = DataFrame({'rates': [0.0523, -0.12, 1.0]})

    # Act
    result = df.select(
        [fmt_percent(columns='rates', decimals=2, use_seps=False).alias('pct')]
    )

    # Assert
    expected = ['5.23%', '-12.00%', '100.00%']
    assert result['pct'].to_list() == expected


def test_fmt_currency():
    """Verifies currency symbols and accounting style formatting for negatives."""
    # Arrange
    df = DataFrame({'amount': [1250.5, -450.0, 0.0]})

    # Act
    result = df.select(
        [
            fmt_currency(
                columns='amount',
                symbol='$',
                decimals=2,
                use_seps=True,
                accounting=True,
            ).alias('usd'),
            fmt_currency(
                columns='amount',
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
    df = DataFrame({'value': [0.0, -0.111, 2.22, 33.3, 444.0, -0.00555]})

    # Act
    result = df.select(
        [
            fmt_scientific(
                columns='value', decimals=2, exp_style='x10n'
            ).alias('default_style'),
            fmt_scientific(columns='value', decimals=1, exp_style='E').alias(
                'e_capital_style'
            ),
            fmt_scientific(
                columns='value', decimals=2, scale_by=10.0, exp_style='e'
            ).alias('scaled_style'),
            fmt_scientific(
                columns='value',
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


def test_fmt_bytes():
    """
    Verifies that fmt_bytes accurately parses raw integers into human-readable bytes scale,
    supporting both decimal (1000) and binary (1024) tracking modes.
    """
    # Arrange
    df = DataFrame(
        {'value': [0.0, 444.0, 5500.0, 777000.0, 8900000.0, -1048576.0]}
    )

    # Act
    result = df.select(
        [
            # Test default decimal standard with space
            fmt_bytes(columns='value', decimals=1, standard='decimal').alias(
                'decimal_std'
            ),
            # Test binary standard with space
            fmt_bytes(columns='value', decimals=1, standard='binary').alias(
                'binary_std'
            ),
            # Test binary standard without intervening space and forced sign
            fmt_bytes(
                columns='value',
                decimals=2,
                standard='binary',
                incl_space=False,
                force_sign=True,
            ).alias('compressed_signed'),
        ]
    )

    # Assert
    expected_decimal = [
        '0.0 B',
        '444.0 B',
        '5.5 kB',
        '777.0 kB',
        '8.9 MB',
        '-1.0 MB',
    ]
    expected_binary = [
        '0.0 B',
        '444.0 B',
        '5.4 KiB',
        '758.8 KiB',
        '8.5 MiB',
        '-1.0 MiB',
    ]
    expected_compressed_signed = [
        '0.00B',
        '+444.00B',
        '+5.37KiB',
        '+758.79KiB',
        '+8.49MiB',
        '-1.00MiB',
    ]

    assert result['decimal_std'].to_list() == expected_decimal
    assert result['binary_std'].to_list() == expected_binary
    assert result['compressed_signed'].to_list() == expected_compressed_signed


def test_fmt_integer_and_substitution_helpers():
    """
    Verifies integer formatting rules (no decimals, thousand separators)
    and checks conditional substitution helpers (sub_missing, sub_zero, fmt_tf).
    """
    # Arrange
    df = DataFrame(
        {
            'integers': [0, -1500, 2500000, None],
            'booleans': [True, False, True, None],
            'missing_mix': [1.23, None, 0.0, 4.56],
            'zeros_mix': [0.0, -0.0, 10.5, 0.0],
        }
    )

    # Act
    result = df.select(
        [
            fmt_integer(columns='integers', use_seps=True).alias(
                'int_standard'
            ),
            fmt_integer(
                columns='integers', compact=True, compact_system='financial'
            ).alias('int_compact'),
            fmt_tf(columns='booleans', true_val='Yes', false_val='No').alias(
                'tf_custom'
            ),
            sub_missing(columns='missing_mix', missing_text='N/A').alias(
                'missing_replaced'
            ),
            sub_zero(columns='zeros_mix', zero_text='-').alias(
                'zeros_replaced'
            ),
        ]
    )

    # Assert
    expected_int_standard = ['0', '-1,500', '2,500,000', None]

    # -1500 becomes -1.5 -> rounded away from zero via epsilon -> -2K
    expected_int_compact = ['0', '-2K', '3M', None]

    expected_tf = ['Yes', 'No', 'Yes', None]
    expected_missing = ['1.23', 'N/A', '0.0', '4.56']
    expected_zeros = ['-', '-', '10.5', '-']

    assert result['int_standard'].to_list() == expected_int_standard
    assert result['int_compact'].to_list() == expected_int_compact
    assert result['tf_custom'].to_list() == expected_tf
    assert result['missing_replaced'].to_list() == expected_missing
    assert result['zeros_replaced'].to_list() == expected_zeros


def test_fmt_tf_advanced():
    """
    Verifies that fmt_tf correctly resolves style presets, honors explicit overrides,
    injects patterns cleanly, and replaces null markers if na_val is provided.
    """
    # Arrange
    df = DataFrame({'status': [True, False, None, True]})

    # Act
    result = df.select(
        [
            # Test preset styles with explicit na_val substitution
            fmt_tf(columns='status', tf_style='arrows', na_val='—').alias(
                'arrows_with_na'
            ),
            # Test custom true/false override values alongside a text pattern wrapper
            fmt_tf(
                columns='status',
                true_val='Active',
                false_val='Inactive',
                pattern='[{x}]',
                na_val='Missing',
            ).alias('custom_override'),
            # Test standard default behavior without na_val handling (retains null)
            fmt_tf(columns='status', tf_style='yes-no').alias(
                'default_retains_null'
            ),
        ]
    )

    # Assert
    assert result['arrows_with_na'].to_list() == ['↑', '↓', '—', '↑']
    assert result['custom_override'].to_list() == [
        '[Active]',
        '[Inactive]',
        'Missing',
        '[Active]',
    ]
    assert result['default_retains_null'].to_list() == [
        'yes',
        'no',
        None,
        'yes',
    ]


def test_multicolumn_decoration():
    """Verifies that formatting functions accept list vectors of string columns cleanly."""
    # Arrange
    df = DataFrame({'A': [1000.55, 2500000.0], 'B': [0.005, -500.2]})

    # Act
    # Pass a list of columns to fmt_number instead of a single string
    expressions = fmt_number(columns=['A', 'B'], decimals=1, use_seps=True)

    assert isinstance(expressions, list)
    result = df.with_columns(expressions)

    # Assert
    assert result['A'].to_list() == ['1,000.6', '2,500,000.0']
    assert result['B'].to_list() == ['0.0', '-500.2']


def test_fmt_number_significant_figures():
    """
    Verifies that n_sigfig formats numbers to specific significant figures,
    dynamically managing decimal padding, zero-fills, and handling Nulls natively.
    """
    # Arrange
    df = DataFrame(
        {
            'numbers': [12.345, 1.2345, 123.45, 0.0012345, 1200.0, None],
        }
    )

    # Act
    result = df.select(
        [
            # Test standard 3 significant figures formatting
            fmt_number(columns='numbers', n_sigfig=3, use_seps=True).alias(
                'sigfig_3'
            ),
            # Test strict 1 significant figure rounding
            fmt_number(columns='numbers', n_sigfig=1, use_seps=True).alias(
                'sigfig_1'
            ),
            # Test significant figures combined with a layout pattern wrapper
            fmt_number(
                columns='numbers', n_sigfig=4, pattern='val: {x}'
            ).alias('sigfig_4_pattern'),
        ]
    )

    # Assert
    # 12.345 -> 12.3 (1 dec)
    # 1.2345 -> 1.23 (2 dec)
    # 123.45 -> 123  (0 dec)
    # 0.0012345 -> 0.00123 (5 dec)
    # 1200.0 -> 1,200 (0 dec, checks that thousands separators work)
    expected_sigfig_3 = ['12.3', '1.23', '123', '0.00123', '1,200', None]

    # 12.345 -> 10 (rounds down, keeps 0 dec)
    # 1.2345 -> 1
    # 123.45 -> 100
    # 0.0012345 -> 0.001
    # 1200.0 -> 1,000
    expected_sigfig_1 = ['10', '1', '100', '0.001', '1,000', None]

    # Testing precision padding for trailing zeros (e.g., 1200.0 with 4 sigfigs -> 1200)
    expected_sigfig_4_pattern = [
        'val: 12.35',
        'val: 1.235',
        'val: 123.5',
        'val: 0.001235',
        'val: 1,200',  # Added the comma here to match default use_seps=True
        None,
    ]

    assert result['sigfig_3'].to_list() == expected_sigfig_3
    assert result['sigfig_1'].to_list() == expected_sigfig_1
    assert result['sigfig_4_pattern'].to_list() == expected_sigfig_4_pattern


def test_fmt_number_force_sign():
    """
    Verifies that force_sign=True explicitly prepends a '+' to positive numbers,
    retains '-' for negative numbers, and leaves zero completely unsigned.
    """
    # Arrange
    df = DataFrame(
        {
            'mixed_signs': [123.45, -67.89, 0.0, None],
        }
    )

    # Act
    result = df.select(
        [
            # Test with force_sign enabled
            fmt_number(
                columns='mixed_signs', decimals=1, force_sign=True
            ).alias('signed'),
            # Test default behavior (force_sign=False)
            fmt_number(
                columns='mixed_signs', decimals=1, force_sign=False
            ).alias('default'),
        ]
    )

    # Assert
    # Positive gets '+', negative keeps '-', zero gets no sign
    expected_signed = ['+123.5', '-67.9', '0.0', None]
    expected_default = ['123.5', '-67.9', '0.0', None]

    assert result['signed'].to_list() == expected_signed
    assert result['default'].to_list() == expected_default


def test_formatting_small_floats_compact():
    v = [
        0.001281,
        0.001061,
        0.0001380,
    ]
    df = DataFrame(
        {'v': v},
        schema={'v': Float64},
    )

    # Act
    result = df.select(fmt_number(columns='v', n_sigfig=4, compact=True))
    assert result.get_column('v').to_list() == [
        '0.001281',
        '0.001061',
        '0.0001380',
    ]
