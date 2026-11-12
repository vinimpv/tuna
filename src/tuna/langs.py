"""Per-language specs for the tree-sitter symbol extractor.

A spec says which grammar parses the files, which syntax nodes are symbols,
which wrapper nodes (decorators and the like) belong to a symbol's text, and how
a file path becomes the dotted module prefix of every FQN in that file.

Adding a language means adding its grammar wheel (e.g. `tree-sitter-go`) to
pyproject.toml and one `LangSpec` to SPECS.
"""
import dataclasses
import importlib
from typing import Callable


@dataclasses.dataclass(frozen=True)
class LangSpec:
    name: str
    extensions: tuple[str, ...]
    grammar_module: str                 # importable module exposing language()
    symbols: dict[str, str]             # node type -> kind ("Class" / "Function")
    # Node types that wrap a symbol and are part of its text (e.g. decorators).
    wrappers: tuple[str, ...] = ()
    # Node types whose body holds members: a "Function" directly inside one is a "Method".
    member_scopes: tuple[str, ...] = ()
    name_field: str = "name"
    module_name: Callable[[str], str] = None  # repo-relative path -> dotted module

    def language(self):
        # One Language object per process, shared by every parser.
        if self.name not in _LANGUAGES:
            import tree_sitter
            _LANGUAGES[self.name] = tree_sitter.Language(importlib.import_module(self.grammar_module).language())
        return _LANGUAGES[self.name]


_LANGUAGES = {}


def python_module_name(path):
    """saleor/checkout/tasks.py -> saleor.checkout.tasks. `__init__` stays in the
    name (saleor/core/__init__.py -> saleor.core.__init__) so every FQN maps to
    exactly one file."""
    stem = path[:-3] if path.endswith(".py") else path
    return stem.replace("/", ".")


PYTHON = LangSpec(
    name="python",
    extensions=(".py",),
    grammar_module="tree_sitter_python",
    symbols={"function_definition": "Function", "class_definition": "Class"},
    wrappers=("decorated_definition",),
    member_scopes=("class_definition",),
    module_name=python_module_name,
)

SPECS = {s.name: s for s in (PYTHON,)}


def spec_for_path(path, enabled):
    for name in enabled:
        spec = SPECS[name]
        if path.endswith(spec.extensions):
            return spec
    return None
