# Copyright (c) 2025 Damien Boureille
# Licensed under the MIT License.

"""
Special forms registry for Sheaf compiler.

Organized into modules:
- base: Base SpecialForm class and utilities
- control: if, case, guard, repeat
- binding: defn, lambda, let, defmacro
- flow: ->, as->
- ml: vmap, scan, with-params, static
- utils: get, dict, last, use, quote
"""

from .binding import DefmacroForm, DefnForm, LambdaForm, LetForm
from .control import CaseForm, DoForm, GuardForm, IfForm, RepeatForm
from .flow import ThreadAsForm, ThreadFirstForm
from .ml import ScanForm, StaticForm, VmapForm, WithParamsForm
from .utils import AssocForm, DictForm, GetForm, GetInForm, LastForm, QuoteForm, UseForm

# Registry of all special forms
special_forms = {
    "->": ThreadFirstForm(),
    "as->": ThreadAsForm(),
    "assoc": AssocForm(),
    "case": CaseForm(),
    "defmacro": DefmacroForm(),
    "defn": DefnForm(),
    "dict": DictForm(),
    "do": DoForm(),
    "fn": LambdaForm(),
    "get": GetForm(),
    "get-in": GetInForm(),
    "guard": GuardForm(),
    "if": IfForm(),
    "last": LastForm(),
    "let": LetForm(),
    "repeat": RepeatForm(),
    "scan": ScanForm(),
    "static": StaticForm(),
    "use": UseForm(),
    "vmap": VmapForm(),
    "with-params": WithParamsForm(),
}

__all__ = [
    "special_forms",
    "ThreadFirstForm",
    "ThreadAsForm",
    "AssocForm",
    "CaseForm",
    "DefmacroForm",
    "DefnForm",
    "DictForm",
    "DoForm",
    "LambdaForm",
    "GetForm",
    "GetInForm",
    "GuardForm",
    "IfForm",
    "LastForm",
    "LetForm",
    "RepeatForm",
    "ScanForm",
    "StaticForm",
    "UseForm",
    "VmapForm",
    "WithParamsForm",
    "QuoteForm",
]
