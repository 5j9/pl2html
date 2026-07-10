from html.parser import HTMLParser


class HTMLNormalizer(HTMLParser):
    def __init__(self):
        super().__init__()
        self.output = []

    def handle_starttag(self, tag, attrs):
        # Sort attributes alphabetically and force uniform single spacing
        sorted_attrs = sorted(attrs, key=lambda x: x[0])
        attr_str = ''.join(f' {k}="{v}"' for k, v in sorted_attrs)
        self.output.append(f'<{tag}{attr_str}>')

    def handle_endtag(self, tag):
        self.output.append(f'</{tag}>')

    def handle_data(self, data):
        # Drop pure inter-tag newline/indentation noise,
        # but preserve spaces if there is meaningful string content or text nodes
        if data.strip() == '' and '\n' in data:
            return

        self.output.append(data)


def normalize_html(html_str: str) -> str:
    parser = HTMLNormalizer()
    parser.feed(html_str)
    return ''.join(parser.output)
