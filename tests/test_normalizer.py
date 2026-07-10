from tests import normalize_html


def test_normalizer_preserves_significant_whitespace():
    """
    Ensures that HTMLNormalizer does not discard functionally significant
    whitespace or drop cells that consist entirely of spacing.
    """
    # HTML with varying types of structural whitespace
    raw_html = (
        '<table>\n'
        '  <tbody>\n'
        '    <tr>\n'
        '      <td> Leading Space</td>\n'
        '      <td>Trailing Space </td>\n'
        '      <td> </td>\n'  # Entirely whitespace
        '    </tr>\n'
        '  </tbody>\n'
        '</table>'
    )

    normalized = normalize_html(raw_html)

    # If data.strip() is present:
    # 1. " Leading Space" becomes "Leading Space"
    # 2. "Trailing Space " becomes "Trailing Space"
    # 3. " " becomes completely empty, skipping the append entirely!
    #
    # The assertion below explicitly verifies those spaces are intact.
    assert '<td> Leading Space</td>' in normalized
    assert '<td>Trailing Space </td>' in normalized
    assert '<td> </td>' in normalized
