"""生产源码必须依赖 canonical owner，而不是兼容导入路径。"""

from __future__ import annotations

import ast
from importlib.util import resolve_name
from pathlib import Path


_PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / "personagraph"
_EVALS_ROOT = Path(__file__).resolve().parents[1] / "evals"
_RETIRED_FACADES = {
    "personagraph.evidence_submission_contracts",
    "personagraph.execution_protocol",
    "personagraph.infrastructure",
    "personagraph.model_io.profiles",
    "personagraph.output_language",
    "personagraph.runtime.l1.contracts",
    "personagraph.security",
}


def _module_name(path: Path) -> str:
    relative = path.relative_to(_PACKAGE_ROOT).with_suffix("")
    parts = relative.parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(("personagraph", *parts))


def _assigns_exact_module_alias(tree: ast.AST) -> bool:
    """识别 ``sys.modules[__name__] = canonical_module`` 风格的 facade。"""

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Subscript):
                continue
            owner = target.value
            is_modules = (
                (isinstance(owner, ast.Name) and owner.id == "_modules")
                or (
                    isinstance(owner, ast.Attribute)
                    and isinstance(owner.value, ast.Name)
                    and owner.value.id == "sys"
                    and owner.attr == "modules"
                )
            )
            if (
                is_modules
                and isinstance(target.slice, ast.Name)
                and target.slice.id == "__name__"
            ):
                return True
    return False


def _resolve_import_targets(
    *, node: ast.Import | ast.ImportFrom, module_name: str, is_package: bool
) -> tuple[str, ...]:
    if isinstance(node, ast.Import):
        return tuple(alias.name for alias in node.names)

    if node.level == 0:
        base = node.module or ""
    else:
        package = module_name if is_package else module_name.rpartition(".")[0]
        base = resolve_name("." * node.level + (node.module or ""), package)
    if node.module is not None:
        return (base,)
    return tuple(f"{base}.{alias.name}" for alias in node.names)


def test_production_imports_use_canonical_module_owners() -> None:
    parsed: list[tuple[Path, str, bool, ast.AST]] = []
    legacy_facades: set[str] = set()

    for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        module_name = _module_name(path)
        is_package = path.name == "__init__.py"
        parsed.append((path, module_name, is_package, tree))
        if _assigns_exact_module_alias(tree):
            legacy_facades.add(module_name)

    assert not legacy_facades, (
        "module-alias compatibility facades are forbidden: "
        + ", ".join(sorted(legacy_facades))
    )
    for retired_facade in _RETIRED_FACADES:
        retired_base = _PACKAGE_ROOT.joinpath(
            *retired_facade.removeprefix("personagraph.").split(".")
        )
        retired_module = retired_base.with_suffix(".py")
        assert not retired_module.exists(), f"retired facade returned: {retired_module}"
        assert not retired_base.exists(), f"retired package returned: {retired_base}"
    forbidden_facades = legacy_facades | _RETIRED_FACADES

    violations: list[str] = []
    for path, module_name, is_package, tree in parsed:
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            for target in _resolve_import_targets(
                node=node,
                module_name=module_name,
                is_package=is_package,
            ):
                owner = next(
                    (
                        facade
                        for facade in forbidden_facades
                        if target == facade or target.startswith(f"{facade}.")
                    ),
                    None,
                )
                if owner is not None:
                    relative = path.relative_to(_PACKAGE_ROOT.parent)
                    violations.append(f"{relative}:{node.lineno} -> {target}")

    assert not violations, "legacy facade imports remain:\n" + "\n".join(violations)


def test_evals_do_not_import_retired_facades() -> None:
    violations: list[str] = []
    for path in sorted(_EVALS_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                targets = tuple(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                targets = (node.module or "",)
            else:
                continue
            for target in targets:
                owner = next(
                    (
                        facade
                        for facade in _RETIRED_FACADES
                        if target == facade or target.startswith(f"{facade}.")
                    ),
                    None,
                )
                if owner is not None:
                    relative = path.relative_to(_EVALS_ROOT.parent)
                    violations.append(f"{relative}:{node.lineno} -> {target}")

    assert not violations, "retired facade imports remain in evals:\n" + "\n".join(
        violations
    )


def test_turn_content_and_output_protocol_keep_one_way_dependencies() -> None:
    """持久内容是低层状态；模型输出协议只能单向引用它。"""

    forbidden_by_package = {
        "persistent_turn_content": (
            "personagraph.output_protocol",
            "personagraph.runtime",
            "personagraph.session",
            "personagraph.tools",
        ),
        "output_protocol": (
            "personagraph.runtime",
            "personagraph.session",
            "personagraph.tools",
        ),
    }
    violations: list[str] = []
    for package_name, forbidden_prefixes in forbidden_by_package.items():
        package_dir = _PACKAGE_ROOT / package_name
        for path in sorted(package_dir.rglob("*.py")):
            module_name = _module_name(path)
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, (ast.Import, ast.ImportFrom)):
                    continue
                for target in _resolve_import_targets(
                    node=node,
                    module_name=module_name,
                    is_package=path.name == "__init__.py",
                ):
                    if any(
                        target == prefix or target.startswith(f"{prefix}.")
                        for prefix in forbidden_prefixes
                    ):
                        violations.append(
                            f"{path.relative_to(_PACKAGE_ROOT.parent)}:"
                            f"{node.lineno} -> {target}"
                        )
    assert not violations, "protocol dependency direction changed:\n" + "\n".join(
        violations
    )
