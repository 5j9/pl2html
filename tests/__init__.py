from html.parser import HTMLParser
from re import compile as rc

attr_name_values = rc(r'([^\s=/>]+)\s*=\s*"([^"]*)"').findall


class HTMLNormalizer(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=False)
        self._stack = []
        self.output = []

    def handle_starttag(self, tag, attrs):
        self._stack.append(tag)
        raw_starttag = self.get_starttag_text()
        if raw_starttag is None:
            return

        # HTMLParser decodes attribute entities before passing `attrs` to
        # handle_starttag(). We instead parse the raw start tag so tests can
        # verify the exact escaping emitted by to_html().
        attr_pairs = attr_name_values(raw_starttag)
        attr_str = ''.join(f' {k}="{v}"' for k, v in attr_pairs)
        self.output.append(f'<{tag}{attr_str}>')

    def handle_endtag(self, tag):
        self._stack.pop()
        self.output.append(f'</{tag}>')

    def handle_data(self, data):
        if self._stack[-1] not in {'td', 'th'}:
            self.output.append(data.strip())
            return
        self.output.append(data)

    def handle_entityref(self, name):
        self.output.append(f'&{name};')

    def handle_charref(self, name):
        self.output.append(f'&#{name};')


def normalize_html(html_str: str) -> str:
    parser = HTMLNormalizer()
    parser.feed(html_str)
    return ''.join(parser.output)
