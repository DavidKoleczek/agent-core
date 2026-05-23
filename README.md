<h1 align="center">
    agent-core
</h1>
<p align="center">
    <p align="center">Common building blocks for creating AI agents.</p>
</p>
<p align="center">
    <a href="https://github.com/astral-sh/uv"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json" alt="uv"></a>
    <a href="https://github.com/astral-sh/ty"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json" alt="ty"></a>
    <a href="https://pypi.org/project/agent-core-toolkit/"><img src="https://img.shields.io/pypi/v/agent-core-toolkit" alt="PyPI"></a>
    <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
</p>

Production-ready components for building AI agents, optimized across LLMs and providers. Uses [InteropRouter](https://github.com/DavidKoleczek/interop-router) as a unified AI model provider interface.

> [!NOTE]
> This library is in early development and subject to change.


## Getting Started

### Installation

```bash
# With uv.
uv add agent-core-toolkit

# With pip.
pip install agent-core-toolkit
```

### Optional Dependencies

For the web fetch tool:

```bash
uv add agent-core-toolkit[web]

# Run crawl4ai post-installation setup
crawl4ai-setup
```


## Usage

Run the agent CLI.

```bash
uv run agent-core --prompt "Your prompt here" --working-dir /path/to/directory
```

With all options:

```bash
uv run agent-core --prompt "Your prompt here" --working-dir "/path/to/directory" --mode "permissive" --model "gpt-5.1-codex-max" --model-friendly-name "gpt-5.1-codex-max" --model-knowledge-cutoff "Sep 30, 2024" --timezone "America/New_York"
```

Or run with uvx:

```bash
uvx --from /path/to/agent-core agent-core --prompt "Your prompt here" --working-dir "/path/to/directory"
```


## Development

### Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- [prek](https://github.com/j178/prek/blob/master/README.md#installation)

### Setup

Create uv virtual environment and install dependencies:

```bash
uv sync --frozen --all-extras --all-groups
```

Set up git hooks:

```bash
prek install
```

To update dependencies (updates the lock file):

```bash
uv sync -U --all-extras --all-groups
```

Run formatting, linting, type checking, and tests in one command:

```bash
uv run ruff format && uv run ruff check --fix && uv run ty check && uv run pytest
```

### Further Information

[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)