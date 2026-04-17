# state-space-rs

A Python extension module written in Rust using [PyO3](https://pyo3.rs) and [Maturin](https://www.maturin.rs).

## Prerequisites

- [Rust](https://rustup.rs/) (stable)
- Python ≥ 3.8
- [uv](https://github.com/astral-sh/uv)

## Setup

Create a virtual environment and install dependencies:

```bash
uv venv
source .venv/bin/activate
uv pip install maturin pytest
```

## Build

Compile the Rust code and install the module into your virtualenv:

```bash
maturin develop
```

For an optimised release build:

```bash
maturin develop --release
```

## Usage

```python
import kalman_rs

state_space_rs.sum_as_string(2, 3)  # "5"
state_space_rs.add(2, 3)            # 5
```

### Available Functions

| Function | Signature | Description |
|---|---|---|
| `sum_as_string` | `(a: int, b: int) -> str` | Returns the sum of two numbers as a string |
| `add` | `(a: int, b: int) -> int` | Returns the sum of two integers |


### Python Visualization Example

A complete example that samples data, runs filtering/smoothing, and plots latent-state estimates with 90% intervals is available at:

```bash
python examples/python_visualization.py
```

This example requires `matplotlib`:

```bash
pip install matplotlib
```

## Development Workflow

After every code change, rebuild the library and re-run the tests:

```bash
maturin develop && pytest tests/ -v
```

For an optimised rebuild:

```bash
maturin develop --release && pytest tests/ -v
```
