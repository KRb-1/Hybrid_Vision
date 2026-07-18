from __future__ import annotations

import inspect
import re
import textwrap
from types import FunctionType

from .rgb_event_stem import RGBEventStem


def register_rgb_event_stem() -> None:
    """Register RGBEventStem in Ultralytics without editing site-packages.

    Recent Ultralytics versions keep the module registry inside parse_model().
    This function patches parse_model in memory so the custom stem receives the
    standard (c1, c2, ...) channel arguments and downstream layers see c2.
    """

    import ultralytics.nn.modules as modules
    import ultralytics.nn.tasks as tasks

    setattr(modules, "RGBEventStem", RGBEventStem)
    setattr(tasks, "RGBEventStem", RGBEventStem)

    current = getattr(tasks, "parse_model")
    if getattr(current, "_rgb_event_stem_registered", False):
        return

    source = _patch_parse_model_source(textwrap.dedent(inspect.getsource(current)))
    namespace = current.__globals__.copy()
    namespace["RGBEventStem"] = RGBEventStem
    exec(compile(source, inspect.getsourcefile(current) or "<parse_model_patch>", "exec"), namespace)
    patched = namespace["parse_model"]
    patched._rgb_event_stem_registered = True  # type: ignore[attr-defined]

    tasks.parse_model = FunctionType(
        patched.__code__,
        {**current.__globals__, "RGBEventStem": RGBEventStem},
        name=current.__name__,
        argdefs=current.__defaults__,
        closure=patched.__closure__,
    )
    tasks.parse_model._rgb_event_stem_registered = True  # type: ignore[attr-defined]


def _patch_parse_model_source(source: str) -> str:
    """Add RGBEventStem to the Ultralytics parse_model channel-aware module set."""

    source = _patch_yaml_input_channels(source)

    if "RGBEventStem" in source:
        return source

    inline_marker = "base_modules = frozenset({"
    if inline_marker in source:
        return source.replace(inline_marker, inline_marker + "\n        RGBEventStem,", 1)

    multiline_match = re.search(r"(base_modules\s*=\s*frozenset\(\s*\{\s*)", source)
    if multiline_match:
        insert_at = multiline_match.end()
        return source[:insert_at] + "\n            RGBEventStem," + source[insert_at:]

    patched = _insert_into_old_channel_aware_branch(source)
    if patched is not None:
        return patched

    raise RuntimeError(
        "Could not find the Ultralytics parse_model channel-aware module registry. "
        "Please send the output of `import inspect, ultralytics.nn.tasks as t; "
        "print(inspect.getsource(t.parse_model)[:4000])`."
    )


def _patch_yaml_input_channels(source: str) -> str:
    """Make parse_model honor top-level YAML `ch` for custom 6-channel models."""

    replacements = [
        ("ch = [ch]", "ch = [d.get('ch', ch)]"),
        ("ch=[ch]", "ch=[d.get('ch', ch)]"),
    ]
    for old, new in replacements:
        if old in source and new not in source:
            return source.replace(old, new, 1)
    return source


def _insert_into_old_channel_aware_branch(source: str) -> str | None:
    """Patch older Ultralytics parse_model versions."""

    patterns = [
        re.compile(r"if m in\s*\{", re.MULTILINE),
        re.compile(r"if m in\s*\(", re.MULTILINE),
    ]
    for pattern in patterns:
        for match in pattern.finditer(source):
            body_preview = source[match.end() : match.end() + 1200]
            compact = re.sub(r"\s+", "", body_preview)
            if "c1,c2=ch[f],args[0]" in compact or "c1,c2=ch[f],args[0]" in compact.replace(" ", ""):
                insert_at = match.end()
                indent = _module_set_indent(source, match.start())
                return source[:insert_at] + f"\n{indent}RGBEventStem," + source[insert_at:]
    return None


def _module_set_indent(source: str, position: int) -> str:
    line_start = source.rfind("\n", 0, position) + 1
    leading = source[line_start:position]
    base_indent = leading[: len(leading) - len(leading.lstrip())]
    return base_indent + "    "
