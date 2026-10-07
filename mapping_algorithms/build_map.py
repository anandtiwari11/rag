"""
Codebase mapper: walk a repo → find every Python function/method → write map.json

Tracks:
  - parent_class / parent_function  (where the method lives)
  - calls / called_by               (who calls whom — "parent" callers)

Usage:
    python -m mapping_algorithms.build_map /path/to/repo
    python -m mapping_algorithms.build_map /path/to/repo --out utils/map.json
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".tox",
    "dist",
    "build",
    ".eggs",
}


@dataclass
class MethodEntry:
    id: int
    name: str
    file: str
    folder: str
    start_line: int
    end_line: int
    kind: str
    # Structural parents (enclosing scope)
    parent_class: str | None = None
    parent_function: str | None = None  # outer def, if nested
    qualified_name: str = ""
    # Call graph (filled after all files are parsed)
    calls: list[dict] = field(default_factory=list)       # [{"name": "...", "id": N|null}]
    called_by: list[int] = field(default_factory=list)     # method ids that call this one
    # Temporary: raw callee names collected during parse (not written to JSON)
    _raw_calls: list[str] = field(default_factory=list, repr=False)


def should_skip_dir(name: str) -> bool:
    return name in SKIP_DIRS or name.startswith(".")


def discover_py_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*.py"):
        rel_parts = path.relative_to(root).parts
        if any(should_skip_dir(part) for part in rel_parts[:-1]):
            continue
        if path.is_file():
            files.append(path)
    return sorted(files)


def _end_lineno(node: ast.AST) -> int:
    end = getattr(node, "end_lineno", None)
    if end is not None:
        return int(end)
    return int(node.lineno)  # type: ignore[arg-type]


def _call_name(func: ast.AST) -> str | None:
    """foo() → foo ; self.bar() / obj.bar() → bar"""
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _calls_in_function(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """Names called directly in this function body (not inside nested defs/classes)."""
    found: list[str] = []

    class CallCollector(ast.NodeVisitor):
        def visit_FunctionDef(self, n: ast.FunctionDef) -> None:
            return  # skip nested

        def visit_AsyncFunctionDef(self, n: ast.AsyncFunctionDef) -> None:
            return

        def visit_ClassDef(self, n: ast.ClassDef) -> None:
            return

        def visit_Call(self, n: ast.Call) -> None:
            name = _call_name(n.func)
            if name:
                found.append(name)
            self.generic_visit(n)

    for stmt in node.body:
        CallCollector().visit(stmt)
    return found


def parse_file(path: Path, root: Path, next_id: int) -> tuple[list[MethodEntry], int]:
    try:
        source = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        source = path.read_text(encoding="utf-8", errors="replace")

    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        print(f"skip (syntax error): {path.relative_to(root)} — {exc}", file=sys.stderr)
        return [], next_id

    rel = path.relative_to(root).as_posix()
    folder = path.parent.relative_to(root).as_posix()
    if folder == ".":
        folder = ""

    entries: list[MethodEntry] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.class_stack: list[str] = []
            self.func_stack: list[str] = []

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self.class_stack.append(node.name)
            self.generic_visit(node)
            self.class_stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._record(node, async_=False)
            self.func_stack.append(node.name)
            self.generic_visit(node)
            self.func_stack.pop()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._record(node, async_=True)
            self.func_stack.append(node.name)
            self.generic_visit(node)
            self.func_stack.pop()

        def _record(self, node: ast.FunctionDef | ast.AsyncFunctionDef, async_: bool) -> None:
            nonlocal next_id
            in_class = bool(self.class_stack)
            parent_class = self.class_stack[-1] if self.class_stack else None
            # outer function only (not self)
            parent_function = self.func_stack[-1] if self.func_stack else None

            if async_ and in_class:
                kind = "async_method"
            elif async_:
                kind = "async_function"
            elif in_class:
                kind = "method"
            else:
                kind = "function"

            parts = list(self.class_stack) + list(self.func_stack) + [node.name]
            qualified = ".".join(parts)

            entries.append(
                MethodEntry(
                    id=next_id,
                    name=node.name,
                    file=rel,
                    folder=folder,
                    start_line=int(node.lineno),
                    end_line=_end_lineno(node),
                    kind=kind,
                    parent_class=parent_class,
                    parent_function=parent_function,
                    qualified_name=qualified,
                    _raw_calls=_calls_in_function(node),
                )
            )
            next_id += 1

    Visitor().visit(tree)
    return entries, next_id


def link_call_graph(entries: list[MethodEntry]) -> None:
    """
    Resolve raw call names → method ids, fill calls + called_by.

    If several methods share a name:
      1) prefer one defined in the same file
      2) otherwise link to ALL of them (marked "ambiguous": true) — for
         "where is X used?" a possible caller is more useful than none.
    Names that match nothing in the repo (Django ORM .filter(), builtins, ...)
    keep id=null.
    """
    by_name: dict[str, list[MethodEntry]] = defaultdict(list)
    for e in entries:
        by_name[e.name].append(e)

    called_by_sets: dict[int, set[int]] = defaultdict(set)

    for caller in entries:
        resolved: list[dict] = []
        seen_names: set[str] = set()
        for callee_name in caller._raw_calls:
            if callee_name in seen_names:
                continue
            seen_names.add(callee_name)

            candidates = by_name.get(callee_name, [])
            targets: list[MethodEntry] = []
            ambiguous = False
            if len(candidates) == 1:
                targets = candidates
            elif len(candidates) > 1:
                same_file = [c for c in candidates if c.file == caller.file]
                if len(same_file) == 1:
                    targets = same_file
                else:
                    targets = candidates
                    ambiguous = True

            if not targets:
                resolved.append({"name": callee_name, "id": None})
                continue

            for target in targets:
                if target.id == caller.id and target.name == "__init__":
                    continue  # super().__init__() is not a self-call
                entry = {"name": callee_name, "id": target.id}
                if ambiguous:
                    entry["ambiguous"] = True
                resolved.append(entry)
                called_by_sets[target.id].add(caller.id)

        caller.calls = resolved

    for e in entries:
        e.called_by = sorted(called_by_sets.get(e.id, set()))


StableKey = tuple[str, str, int]  # (file, qualified_name, nth definition with that name in the file)


def _stable_keys(items: list) -> list[StableKey]:
    """
    Build a key per method. The same name can be defined more than once in one
    file (e.g. a function redefined lower down), so the key also carries which
    occurrence it is: 0 for the first definition, 1 for the second, ...
    Items must be in file order.
    """
    seen: dict[tuple[str, str], int] = {}
    keys: list[StableKey] = []
    for file, qualified_name in items:
        n = seen.get((file, qualified_name), 0)
        seen[(file, qualified_name)] = n + 1
        keys.append((file, qualified_name, n))
    return keys


def load_previous_ids(map_path: Path) -> dict[StableKey, int]:
    """Stable key -> id from an existing map.json, so ids survive re-indexing."""
    if not map_path.is_file():
        return {}
    try:
        data = json.loads(map_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    methods = data.get("methods", [])
    keys = _stable_keys([(m.get("file", ""), m.get("qualified_name") or m.get("name", "")) for m in methods])
    return {key: int(m["id"]) for key, m in zip(keys, methods)}


def assign_stable_ids(entries: list[MethodEntry], previous: dict[StableKey, int]) -> int:
    """
    Keep the id a method had last time. New methods get the next id after the
    highest one ever used, so an id is never reused for a different method.
    Returns how many methods are new.
    """
    if not previous:
        return len(entries)

    next_id = max(previous.values()) + 1
    new_count = 0
    keys = _stable_keys([(e.file, e.qualified_name) for e in entries])
    for e, key in zip(entries, keys):
        old = previous.get(key)
        if old is not None:
            e.id = old
        else:
            e.id = next_id
            next_id += 1
            new_count += 1
    return new_count


def build_map(root: Path, previous: dict[StableKey, int] | None = None) -> tuple[list[MethodEntry], int]:
    """Returns (entries, number_of_new_methods)."""
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Repo root not found: {root}")

    all_entries: list[MethodEntry] = []
    next_id = 1

    for path in discover_py_files(root):
        entries, next_id = parse_file(path, root, next_id)
        all_entries.extend(entries)

    new_count = assign_stable_ids(all_entries, previous or {})
    link_call_graph(all_entries)
    return all_entries, new_count


def entry_to_dict(e: MethodEntry) -> dict:
    d = asdict(e)
    d.pop("_raw_calls", None)
    return d


def write_map(entries: list[MethodEntry], out_path: Path, root: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "repo_root": str(root.resolve()),
        "method_count": len(entries),
        "methods": [entry_to_dict(e) for e in entries],
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build method map for a Python repo")
    parser.add_argument("repo_root", nargs="?", default=".")
    parser.add_argument("--out", default=None)
    parser.add_argument("--fresh", action="store_true", help="ignore existing ids and renumber from 1")
    args = parser.parse_args(argv)

    project_root = Path(__file__).resolve().parent.parent
    out_path = Path(args.out) if args.out else project_root / "utils" / "map.json"
    root = Path(args.repo_root).resolve()

    previous = {} if args.fresh else load_previous_ids(out_path)
    entries, new_count = build_map(root, previous)
    write_map(entries, out_path, root)

    with_callers = sum(1 for e in entries if e.called_by)
    print(f"Indexed {len(entries)} methods from {root}")
    if previous:
        removed = len(previous) - (len(entries) - new_count)
        print(f"Kept ids for {len(entries) - new_count}, new methods: {new_count}, removed: {removed}")
    print(f"Methods that have at least one caller: {with_callers}")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
