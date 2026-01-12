# General Instructions
- This is a production-grade Python package. You must *always* follow best open-source Python practices.
- Shortcuts are not appropriate. When in doubt, you must work with the user for guidance.
- Any documentation you write, including in the README.md, should be clear, concise, and accurate like the official documentation of other production-grade Python packages.
- Make sure any comments in code are necessary. A necessary comment captures intent that cannot be encoded in names, types, or structure. Comments should be reserved for the "why", only used to record rationale, trade-offs, links to specs/papers, or non-obvious domain insights. They should add signal that code cannot.
- The current code in the package should be treated as an example of high quality code. Make sure to follow its style and tackle issues in similar ways where appropriate.
- Do not run the e2e/integration tests automatically unless asked since they take a while.
- Don't generate characters that a user could not type on a standard keyboard like fancy arrows.
- Anything is possible. Do not blame external factors after something doesn't work on the first try. Instead, investigate and test assumptions through debugging through first principles.

# Python Development Instructions
- `ty` by Astral is used for type checking. Always add appropriate type hints such that the code would pass ty's type check.
- Follow the Google Python Style Guide.
- After each code change, checks are automatically run. Fix any issues that arise.
- **IMPORTANT**: The checks will remove any unused imports after you make an edit to a file. So if you need to use a new import, be sure to use it FIRST (or do your edits at the same time) or else it will be automatically removed. DO NOT use local imports to get around this.
- Always prefer pathlib for dealing with files. Use `Path.open` instead of `open`.
- When using pathlib, **always** Use `.parents[i]` syntax to go up directories instead of using `.parent` multiple times.
- When writing tests, use pytest and pytest-asyncio.
- Prefer using loguru for logging instead of the built-in logging module. Do not add logging unless requested.
- NEVER use `# type: ignore`. It is better to leave the issue and have the user work with you to fix it.
- Don't put types in quotes unless it is absolutely necessary to avoid circular imports and forward references.
- When constructing long strings like prompts for LLMs, use `python-liquid`'s `render` function:
```python
from liquid import render

print(render("Hello, {{ you }}!", you="World"))
# Hello, World!
```
- To learn about how packages work, you should read from the relevant source code. This is especially important when determining which types to use.
  - `interop-router`: Start at `.venv/lib/python3.11/site-packages/interop_router/router.py` and `.venv/lib/python3.11/site-packages/interop_router/types.py`
  - `openai`: Key files stem from `.venv/lib/python3.11/site-packages/openai/resources` and `.venv/lib/python3.11/site-packages/openai/types`
  - `anthropic`: Key files stem from `.venv/lib/python3.11/site-packages/anthropic/resources/messages/messages.py`

# Documentation Instructions
- Keep it very concise
- No emojis or em dashes.

# Key Files

@README.md

@pyproject.toml

@src/agent_core/_types.py

@docs/DEVELOPMENT.md
