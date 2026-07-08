# pl2html

A fast, secure, and zero-dependency HTML engine for Polars. 

`pl2html` compiles Polars `DataFrame` and `LazyFrame` workflows natively into HTML table structures entirely inside the Polars expression engine. By leveraging parallelized Rust memory environments, it bypasses slow Python row loops (`.map_elements`), making it an incredibly lightweight alternative to styling engines like `great_tables` or pandas `.style`.

## Features

* 🚀 **Native Polars Performance**: Runs entirely inside the Polars expression graph. Streamable and lazy-execution friendly.
* 🛡️ **Injection-Safe by Default**: Automatically neutralizes sequential HTML injection vectors using a lookahead-free, native escaping algorithm.
* 🎨 **Conditional Styling & Color Scales**: Native linear interpolation pipelines to generate multi-segment background color heatmaps by value or percentile rank.
* 🌗 **Auto-Contrast Accessibility**: Dynamic text foreground switches (`#000000` vs `#FFFFFF`) calculated via WCAG relative luminance algorithms on the fly.
* 📦 **Zero External Dependencies**: Cleaned of thick packages like `numpy` or `matplotlib`. It runs entirely on pure Python and Polars.

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
        'margin': [0.2451, -0.0512],
    }
)

# Compiles safely to a Polars LazyFrame containing the HTML output
html_lazy = to_html(df)

# Collect and extract the compiled string
html_table_string = html_lazy.collect().item()
print(html_table_string)
```


Result:

<table>
  <thead>
    <tr>
      <th>company</th>
      <th>revenue</th>
      <th>margin</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>&lt;script&gt;alert(&#x27;malicious&#x27;)&lt;/script&gt; Corp</td><td>1,420,500</td><td>0.245</td></tr>
    <tr><td>Acme Inc.</td><td>-50,200</td><td>-0.051</td></tr>
  </tbody>
</table>


### 2. Advanced Styling via Native Attributes (`attrs`)
Instead of embedding hacky structural HTML tags in your data, use the `attrs` parameter to inject dynamic CSS attributes purely. Use `data_color` or `rank_color` to construct high-performance heatmaps.

```python
import polars as pl

from pl2html import to_html
from pl2html.styles import data_color, rank_color

df = pl.DataFrame(
    {
        'employee': ['Alice', 'Bob', 'Charlie', 'David'],
        'performance_score': [98.5, 42.0, 81.2, 12.0],
        'utilization': [0.92, 0.41, 0.78, 0.22],
    }
)

# Create advanced styling dictionaries containing native Polars expressions
styles = {}

# 1. Color Heatmap by Absolute Value (Custom 3-color palette with auto-contrast text)
styles.update(
    data_color(
        column='performance_score',
        palette=['#ff7675', '#fdcb6e', '#00b894'],  # Red -> Yellow -> Green
        auto_contrast=True,
    )
)

# 2. Color Heatmap by Percentile Rank Order (Continuous  interpolation)
styles.update(
    rank_color(
        column='utilization',
        palette=['#000000', '#FFFFFF'],  # Black -> White rank shift
        descending=False,
        auto_contrast=True,
    )
)

# Compile everything lazily without creating temporary data columns
html_table = to_html(df, attrs=styles).collect().item()
```

Result:

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

The library is explicitly divided into two decoupled layers:

### 1. Core Compiler (`__init__.py`)
Responsible for reading the dataframe schema, iterating horizontally over visible columns, and joining tokens structural row arrays. 
* Exposes `to_html(df, *, attrs=None, exclude_columns=None)`
* Automatically assigns high-performance format fallbacks:
  * `is_integer()`: Applies thousands separator reversals.
  * `is_float()`: Automatically base-truncates decimal positions and breaks layout chunks smoothly.
  * *Catchall*: Direct pass through `_escape_polars_string()` to secure against cross-site scripting (XSS).

### 2. Style Utilities (`styles.py`)
Generates structural Polars expression payloads mapped to specific HTML components.
* `data_color(column, palette, domain=None, auto_contrast=True)`: Evaluates column boundaries (`.min()` / `.max()`) lazily and maps specific row values via piece-wise linear rgb calculations.
* `rank_color(column, palette, descending=False, auto_contrast=True)`: Replaces absolute numerical scaling with ordinal ranks. Excellent for smoothing out heavily skewed datasets or extreme outliers.

---

## Security

`pl2html` passes untrusted data arrays through a strict sequential expression loop:
```python
.str.replace_all('&', '&amp;')
.str.replace_all('<', '&lt;')
.str.replace_all('>', '&gt;')
.str.replace_all('"', '&quot;')
.str.replace_all("'", '&#x27;")
```
