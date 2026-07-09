# pl2html

A fast, secure, and zero-dependency HTML engine for Polars. 

`pl2html` compiles Polars `DataFrame` and `LazyFrame` workflows natively into HTML table structures entirely inside the Polars expression engine. By leveraging parallelized Rust memory environments, it bypasses slow Python row loops (`.map_elements`), making it an incredibly lightweight alternative to styling engines like `great_tables` or pandas `.style`.

## Features

* 🚀 **Native Polars Performance**: Runs entirely inside the Polars expression graph. Streamable and lazy-execution friendly.
* 🛡️ **Injection-Safe by Default**: Automatically neutralizes sequential HTML injection vectors using a lookahead-free, native escaping algorithm.
* 🎨 **Advanced Formatting Toolkit**: Human-readable display rules for numbers, integers, currencies, percentages, scientific notations, bytes, and booleans running entirely in Rust.
* ➡️ **Multicolumn Support**: All formatting and substitution functions accept a single column string or an iterable list of columns automatically.
* 🌗 **Conditional Styling & Color Scales**: Native linear interpolation pipelines to generate multi-segment background color heatmaps by value or percentile rank.
* 🌌 **Auto-Contrast Accessibility**: Dynamic text foreground switches (`#000000` vs `#FFFFFF`) calculated via WCAG relative luminance algorithms on the fly.
* 📦 **Afraid of Bloat? Zero Dependencies**: Cleaned of heavy packages like `numpy` or `matplotlib`. It runs entirely on pure Python and Polars.

---

## Installation

```bash
pip install pl2html
```

---

## Quick Start

### 1. Basic HTML Rendering with Auto-Escaping
By default, `pl2html` infers datatypes to securely render human-readable tables. Strings are escaped, and integers/floats automatically get localized thousands separators.

```python
import polars as pl
from pl2html import to_html

df = pl.DataFrame(
    {
        'company': ["<script>alert('malicious')</script> Corp", 'Acme Inc.'],
        'revenue': [1420500, -50200],
        'margin': [0.0451, -0.0512],
    }
)

# Compiles safely to a Polars LazyFrame containing the HTML output
html_lazy = to_html(df)

# Collect and extract the compiled string
html_table_string = html_lazy.collect().item()
print(html_table_string)
```

Output:

<table>
 <thead>
  <tr>
   <th>company</th>
   <th>revenue</th>
   <th>margin</th>
  </tr>
 </thead>
 <tbody>
  <tr><td>&lt;script&gt;alert(&#x27;malicious&#x27;)&lt;/script&gt; Corp</td><td>1,420,500</td><td>0.045</td></tr>
  <tr><td>Acme Inc.</td><td>-50,200</td><td>-0.051</td></tr>
 </tbody>
</table>


### 2. Rich Data Formatting (Natively Vectorized)
`pl2html` exposes high-performance formatting expressions designed to prepare columns simultaneously before passing them to `to_html`. Every function natively supports passing lists of multiple columns at once.

```python
import polars as pl
from pl2html import to_html
from pl2html import formats as fmt

df = pl.DataFrame(
    {
        "asset_a": [5400600, 2100],
        "asset_b": [45100000, -850000],
        "growth": [0.1256, -0.024],
        "is_active": [True, False],
        "status_code": [0, 404]
    }
)

# Format multiple financial asset columns compactly in one pass using lists
df_f = df.with_columns(
    fmt.fmt_integer(["asset_a", "asset_b"], compact=True, compact_system="financial"),
    fmt.fmt_percent("growth", decimals=1),
    fmt.fmt_tf("is_active", tf_style="check-mark"),
    fmt.sub_zero("status_code", zero_text="OK")
)

html = to_html(df_f).collect().item()
```

### 3. Advanced Styling via Native Attributes (`attrs`)
Instead of embedding hacky structural HTML tags in your data, use the `attrs` parameter to inject dynamic CSS attributes purely. Use `data_color` or `rank_color` to construct high-performance heatmaps.

```python
import polars as pl
from pl2html import to_html
from pl2html.styles import data_color, rank_color

df = pl.DataFrame(
    {
        'employee': ['Alice', 'Bob', 'Charlie', 'David'],
        'performance_score': [98.5, 42.0, 81.2, 12.0],
        'utilization': [0.02, 0.41, 0.78, 0.22],
    }
)

styles = {}

# 1. Color Heatmap by Absolute Value (Custom 3-color palette with auto-contrast text)
styles.update(
    data_color(
        column='performance_score',
        palette=['#ff7675', '#fdcb6e', '#00b894'],
        auto_contrast=True
    )
)

# 2. Color Heatmap by Percentile Rank Order (Continuous interpolation)
styles.update(
    rank_color(
        column='utilization',
        palette=['#000000', '#FFFFFF'],
        descending=False,
        auto_contrast=True
    )
)

html_table = to_html(df, attrs=styles).collect().item()
```

Output (github may override table colors, but they are there):


<table>
 <thead>
  <tr>
   <th>employee</th>
   <th>performance_score</th>
   <th>utilization</th>
  </tr>
 </thead>
 <tbody>
  <tr><td>Alice</td><td style="background-color: rgb(0,184,148); color: #000000;">98.5</td><td style="background-color: rgb(255,255,255); color: #000000;">0.92</td></tr>
  <tr><td>Bob</td><td style="background-color: rgb(254,177,112); color: #000000;">42.0</td><td style="background-color: rgb(85,85,85); color: #FFFFFF;">0.41</td></tr>
  <tr><td>Charlie</td><td style="background-color: rgb(101,192,133); color: #000000;">81.2</td><td style="background-color: rgb(170,170,170); color: #000000;">0.78</td></tr>
  <tr><td>David</td><td style="background-color: rgb(255,118,117); color: #000000;">12.0</td><td style="background-color: rgb(0,0,0); color: #FFFFFF;">0.22</td></tr>
 </tbody>
</table>

---

## Architecture & Module Layout

The library is explicitly divided into three decoupled layers:

### 1. Core Compiler (`__init__.py`)
Responsible for reading the dataframe schema, iterating horizontally over visible columns, and joining tokens into structural row arrays natively in Rust. 
* Exposed via `to_html(df, *, attrs=None, exclude_columns=None)`.
* Automatically assigns high-performance format fallbacks:
  * `is_integer()`: Applies thousands separator reversals.
  * `is_float()`: Automatically base-truncates decimal positions and breaks layout chunks smoothly.
  * *Catchall*: Direct pass through `_escape_polars_string()` to secure against cross-site scripting (XSS).

### 2. Format Expressions (`formats.py`)
Contains native string-shaping expression generators wrapped in a `@_multicolumn` macro to cleanly display large datasets without dropping into slow Python loops.
* `fmt_number()` / `fmt_integer()`: Supports precision rounding, thousands separators, accounting parentheses, and financial/engineering compact suffixing (`K`, `M`, `B` vs `k`, `M`, `G`).
* `fmt_percent()` / `fmt_currency()`: Optimized wrappers handling percentage scaling and currency masking.
* `fmt_scientific()`: Natively parses mantissas and exponents with flexible styling (`x10n`, `e`, `E`) and forced signs.
* `fmt_bytes()`: Dynamically maps byte magnitudes to decimal (`kB`, `MB`) or binary (`KiB`, `MiB`) notations.
* `fmt_tf()`: Transforms booleans to presets (`arrows`, `check-mark`, `circles`) with native `na_val` null overrides.
* `sub_missing()` / `sub_zero()`: Clean conditional data substitution expressions for nulls and exact zeros.

### 3. Style Utilities (`styles.py`)
Generates structural Polars expression payloads mapped to specific HTML style attributes.
* `data_color(column, palette, domain=None, auto_contrast=True)`: Evaluates column boundaries (`.min()` / `.max()`) at evaluation time and maps row values via piece-wise linear RGB interpolation.
* `rank_color(column, palette, descending=False, auto_contrast=True)`: Replaces absolute numerical scaling with ordinal percentile ranks to smooth out heavily skewed datasets or extreme outliers.

**Every style output returns a clean nested dictionary structure (`{column: {attribute: expression}}`) to feed directly into `to_html(attrs=...)`.**

---

## Security

`pl2html` passes untrusted data arrays through a strict sequential escaping chain natively in Rust:
```python
.str.replace_all('&', '&amp;')
.str.replace_all('<', '&lt;')
.str.replace_all('>', '&gt;')
.str.replace_all('"', '&quot;')
.str.replace_all("'", '&#x27;')
```
