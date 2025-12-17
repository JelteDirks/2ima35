from typing import Any
from collections.abc import Mapping, Iterable
import random

def sample_edges(E: dict[int, dict[int, float]], m: int, seed:int=1):
    """
    Sample at most m edges from an undirected edge dict E.
    Preserves the E[i][j] with i < j structure.
    """
    rng = random.Random(seed)

    edges = [(i, j, w) for i, nbrs in E.items() for j, w in nbrs.items()]

    if m >= len(edges):
        return E, len(edges)

    sampled = rng.sample(edges, m)

    E_new = {}
    for i, j, w in sampled:
        E_new.setdefault(i, {})[j] = w

    return E_new, m

def describe_type(
    obj: Any,
    *,
    max_depth: int = 4,
    _depth: int = 0,
) -> str:
    # Depth safety
    if _depth >= max_depth:
        return type(obj).__name__

    t = type(obj)

    # Atomic / scalar types
    if obj is None or isinstance(obj, (int, float, bool, str, bytes)):
        return t.__name__

    # Mapping types (dict, defaultdict, etc.)
    if isinstance(obj, Mapping):
        if not obj:
            return f"{t.__name__}[empty]"
        key_types = set()
        val_types = set()
        for k, v in obj.items():
            key_types.add(describe_type(k, max_depth=max_depth, _depth=_depth + 1))
            val_types.add(describe_type(v, max_depth=max_depth, _depth=_depth + 1))
        return f"{t.__name__}[{_merge_types(key_types)} → {_merge_types(val_types)}]"

    # Sequence / container types
    if isinstance(obj, (list, tuple, set)):
        if not obj:
            return f"{t.__name__}[empty]"
        elem_types = {
            describe_type(e, max_depth=max_depth, _depth=_depth + 1)
            for e in obj
        }
        return f"{t.__name__}[{_merge_types(elem_types)}]"

    # Other iterables (e.g. generators)
    if isinstance(obj, Iterable):
        return f"{t.__name__}[Iterable]"

    # Custom objects: inspect attributes
    if hasattr(obj, "__dict__"):
        fields = {}
        for name, value in vars(obj).items():
            fields[name] = describe_type(value, max_depth=max_depth, _depth=_depth + 1)
        fields_str = ", ".join(f"{k}: {v}" for k, v in fields.items())
        return f"{t.__name__}{{{fields_str}}}"

    return t.__name__


def _merge_types(types: set[str]) -> str:
    if not types:
        return "?"
    if len(types) == 1:
        return next(iter(types))
    return " | ".join(sorted(types))


def assert_structure(
    obj,
    expected: str,
    *,
    max_depth: int = 4,
    name: str | None = None,
) -> None:
    actual = describe_type(obj, max_depth=max_depth)
    expected_norm = _normalize_structure(expected)
    actual_norm = _normalize_structure(actual)

    if actual_norm != expected_norm:
        var = f" '{name}'" if name else ""
        raise AssertionError(
            f"Structural type mismatch{var}:\n"
            f"  Expected: {expected_norm}\n"
            f"  Actual:   {actual_norm}"
        )


def _normalize_structure(s: str) -> str:
    # Remove redundant whitespace
    s = " ".join(s.strip().split())

    # Normalize unions: a | b | c (sorted)
    def normalize_union(part: str) -> str:
        parts = [p.strip() for p in part.split("|")]
        return " | ".join(sorted(parts))

    out = []
    buf = ""
    depth = 0

    for ch in s:
        if ch in "[{(":
            depth += 1
            buf += ch
        elif ch in "]})":
            depth -= 1
            buf += ch
        elif ch == "|" and depth == 0:
            out.append(buf)
            buf = ""
        else:
            buf += ch

    if buf:
        out.append(buf)

    if len(out) > 1:
        return normalize_union(" | ".join(out))
    return s
