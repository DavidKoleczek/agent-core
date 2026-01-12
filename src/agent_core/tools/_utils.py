from collections.abc import Sequence
from enum import Enum
from fnmatch import fnmatch
from pathlib import Path

from pydantic import BaseModel


class ConstraintPolicy(Enum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


class ConstraintRule(BaseModel):
    pattern: str
    policy: ConstraintPolicy


def check_path_constraint(
    file_path: Path,
    rules: Sequence[ConstraintRule],
    working_dir: Path,
    default_policy: ConstraintPolicy,
) -> ConstraintPolicy:
    """Check if a path matches any constraint rules.

    Rules are evaluated in order; first match wins.

    Pattern syntax:
        - `//path` - Absolute path from filesystem root (e.g., `//etc/hosts`)
        - `path` or `./path` - Relative to working_dir (e.g., `*.env`)
        - `**/` requires at least one directory level. To match files at all
          levels, use two rules: `*.pdf` (root) and `**/*.pdf` (subdirs).
    """
    resolved_path = file_path.resolve()

    for rule in rules:
        if rule.pattern.startswith("//"):
            pattern = rule.pattern[1:]
            if fnmatch(str(resolved_path), pattern):
                return rule.policy
        else:
            pattern = rule.pattern.removeprefix("./")
            try:
                relative_path = resolved_path.relative_to(working_dir.resolve())
                if fnmatch(str(relative_path), pattern):
                    return rule.policy
            except ValueError:
                pass

    return default_policy
