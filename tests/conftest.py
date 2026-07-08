from pathlib import Path

import pytest

from tests import normalize_html

# Define the base directory where your HTML files live
TESTDATA_DIR = Path(__file__).parent / 'testdata'


@pytest.fixture
def expected_html(request) -> str:
    """
    Automatically loads the HTML file matching the name of the test function.
    Example: test_render_integers() loads html_fixtures/test_render_integers.html
    """
    # Get the name of the executing test function (e.g., 'test_basic_table')
    test_name = request.node.name

    # If the test is parameterized, request.node.name might include '[param_value]'.
    # We strip that out to keep filenames clean.
    clean_test_name = test_name.split('[')[0]

    file_path = TESTDATA_DIR / f'{clean_test_name}.html'

    if not file_path.exists():
        raise FileNotFoundError(
            f'Expected HTML fixture file not found for this test.\n'
            f'Please create: {file_path}'
        )

    return normalize_html(file_path.read_text(encoding='utf-8'))
