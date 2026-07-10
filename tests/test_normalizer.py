from polars import DataFrame, col

from pl2html.compiler import to_html
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


def test_normalizer_retains_escaped_attribute_entities():
    """
    Ensures HTMLNormalizer with convert_charrefs=False safely preserves
    &quot;, &amp;, etc., inside tag attributes instead of expanding them
    back into raw layout-breaking characters.
    """
    df = DataFrame(
        {
            'username': ['Admin'],
            'bio': ['Likes to write "malicious" code & <script>tags</script>'],
        }
    )

    attrs = {'username': {'title': col('bio')}}
    actual_html = to_html(df, attrs=attrs, exclude_columns=['bio'])

    # With convert_charrefs=False, we can confidently assert against the
    # normalized output because the entities remain explicitly encoded.
    normalized = normalize_html(actual_html)

    assert (
        'title="Likes to write &quot;malicious&quot; code &amp; &lt;script&gt;tags&lt;/script&gt;"'
        in normalized
    )


def test_normalizer_reconstitutes_cell_entities_correctly():
    """
    Verifies that handle_entityref and handle_charref hooks properly capture
    and reconstruct escaped strings inside the table cells without dropping
    tokens or corrupting sequential text chunks.
    """
    # A raw row containing a mix of named and numeric/hex entities
    df = DataFrame({'escaped_text': ['A & B < C > D " E \' F']})

    actual_html = to_html(df)
    normalized = normalize_html(actual_html)

    # Confirm that the entire inner cell text block was perfectly stitched back
    # together by the normalizer's stream hooks.
    assert '<td>A &amp; B &lt; C &gt; D &quot; E &#x27; F</td>' in normalized
