from pathlib import Path

from agent_core.tools._utils import ConstraintPolicy, ConstraintRule, check_path_constraint


def test_allow_directory_and_everything_under_it(tmp_path: Path) -> None:
    rules = [ConstraintRule(pattern="src/**", policy=ConstraintPolicy.ALLOW)]

    assert (
        check_path_constraint(tmp_path / "src" / "main.py", rules, tmp_path, ConstraintPolicy.ASK)
        == ConstraintPolicy.ALLOW
    )
    assert (
        check_path_constraint(tmp_path / "src" / "deep" / "nested.py", rules, tmp_path, ConstraintPolicy.ASK)
        == ConstraintPolicy.ALLOW
    )
    assert (
        check_path_constraint(tmp_path / "other" / "file.txt", rules, tmp_path, ConstraintPolicy.ASK)
        == ConstraintPolicy.ASK
    )


def test_allow_dirs_but_deny_specific_subdirs(tmp_path: Path) -> None:
    rules = [
        ConstraintRule(pattern="data/public/secrets/**", policy=ConstraintPolicy.DENY),
        ConstraintRule(pattern="data/public/**", policy=ConstraintPolicy.ALLOW),
        ConstraintRule(pattern="data/logs/**", policy=ConstraintPolicy.ALLOW),
    ]

    assert (
        check_path_constraint(tmp_path / "data" / "public" / "readme.txt", rules, tmp_path, ConstraintPolicy.ASK)
        == ConstraintPolicy.ALLOW
    )
    assert (
        check_path_constraint(
            tmp_path / "data" / "public" / "secrets" / "keys.txt", rules, tmp_path, ConstraintPolicy.ASK
        )
        == ConstraintPolicy.DENY
    )
    assert (
        check_path_constraint(tmp_path / "data" / "logs" / "app.log", rules, tmp_path, ConstraintPolicy.ASK)
        == ConstraintPolicy.ALLOW
    )
    assert (
        check_path_constraint(tmp_path / "data" / "private" / "file.txt", rules, tmp_path, ConstraintPolicy.ASK)
        == ConstraintPolicy.ASK
    )
    assert (
        check_path_constraint(Path("/home/usr/temp.txt"), rules, tmp_path, ConstraintPolicy.ASK) == ConstraintPolicy.ASK
    )


def test_deny_dotfiles_but_allow_specific_extensions(tmp_path: Path) -> None:
    rules = [
        ConstraintRule(pattern="*.pdf", policy=ConstraintPolicy.ALLOW),
        ConstraintRule(pattern="**/*.pdf", policy=ConstraintPolicy.ALLOW),
        ConstraintRule(pattern="*.docx", policy=ConstraintPolicy.ALLOW),
        ConstraintRule(pattern="**/*.docx", policy=ConstraintPolicy.ALLOW),
        ConstraintRule(pattern=".gitignore", policy=ConstraintPolicy.ALLOW),
        ConstraintRule(pattern="**/.gitignore", policy=ConstraintPolicy.ALLOW),
        ConstraintRule(pattern=".*", policy=ConstraintPolicy.DENY),
        ConstraintRule(pattern="**/.*", policy=ConstraintPolicy.DENY),
    ]

    assert (
        check_path_constraint(tmp_path / "report.pdf", rules, tmp_path, ConstraintPolicy.ASK) == ConstraintPolicy.ALLOW
    )
    assert (
        check_path_constraint(tmp_path / "docs" / "report.pdf", rules, tmp_path, ConstraintPolicy.ASK)
        == ConstraintPolicy.ALLOW
    )
    assert check_path_constraint(tmp_path / "doc.docx", rules, tmp_path, ConstraintPolicy.ASK) == ConstraintPolicy.ALLOW
    assert (
        check_path_constraint(tmp_path / ".gitignore", rules, tmp_path, ConstraintPolicy.ASK) == ConstraintPolicy.ALLOW
    )
    assert (
        check_path_constraint(tmp_path / "config" / ".gitignore", rules, tmp_path, ConstraintPolicy.ASK)
        == ConstraintPolicy.ALLOW
    )
    assert check_path_constraint(tmp_path / ".env", rules, tmp_path, ConstraintPolicy.ASK) == ConstraintPolicy.DENY
    assert check_path_constraint(tmp_path / "readme.txt", rules, tmp_path, ConstraintPolicy.ASK) == ConstraintPolicy.ASK


def test_default_policy_when_no_rules_match(tmp_path: Path) -> None:
    rules: list[ConstraintRule] = []

    assert (
        check_path_constraint(tmp_path / "any" / "file.txt", rules, tmp_path, ConstraintPolicy.ASK)
        == ConstraintPolicy.ASK
    )


def test_first_matching_rule_wins(tmp_path: Path) -> None:
    rules = [
        ConstraintRule(pattern="//etc/hosts", policy=ConstraintPolicy.ALLOW),
        ConstraintRule(pattern="//etc/*", policy=ConstraintPolicy.DENY),
    ]

    assert check_path_constraint(Path("/etc/hosts"), rules, tmp_path, ConstraintPolicy.ASK) == ConstraintPolicy.ALLOW
    assert check_path_constraint(Path("/etc/passwd"), rules, tmp_path, ConstraintPolicy.ASK) == ConstraintPolicy.DENY


def test_absolute_pattern_with_double_slash(tmp_path: Path) -> None:
    rules = [ConstraintRule(pattern="//usr/local/**", policy=ConstraintPolicy.ALLOW)]

    assert (
        check_path_constraint(Path("/usr/local/bin/python"), rules, tmp_path, ConstraintPolicy.ASK)
        == ConstraintPolicy.ALLOW
    )
    assert check_path_constraint(Path("/usr/bin/python"), rules, tmp_path, ConstraintPolicy.ASK) == ConstraintPolicy.ASK
