"""
tex.py — what a formula may contain, and whether the speaker really said it.

check_tex():      whitelist of commands and characters. TeX is never executed by a TeX engine here, but the
                  whitelist still blocks \\input/\\write/\\def/\\href etc. so that no renderer (MathJax, mathtext,
                  a future LaTeX) can ever be asked to touch files, network or macros.
speech_support(): every symbol and operator of the formula must be named in the quote. A formula that only
                  looks right ("E = mc^2" over a vague remark about energy) gets no support and is not shown.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from .evidence import extract_numbers, number_supported, tokens

MAX_TEX_LEN = 120
MAX_DEPTH = 4
GREEK = {"alpha", "beta", "gamma", "delta", "epsilon", "theta", "lambda", "mu", "nu", "pi", "rho", "sigma", "tau",
         "phi", "omega", "Delta", "Omega", "Sigma", "Phi", "Pi", "Lambda", "Gamma", "Theta"}
ALLOWED_COMMANDS = GREEK | {"frac", "sqrt", "cdot", "times", "pm", "approx", "ne", "neq", "le", "leq", "ge", "geq",
                            "sum", "int", "lim", "to", "infty", "sin", "cos", "tan", "log", "ln", "mathrm", "left",
                            "right", "quad", ",", ";"}
_ALLOWED_CHARS = re.compile(r"^[A-Za-z0-9 +\-=/^_{}()\[\].,|<>!:']*$")
_COMMAND = re.compile(r"\\([A-Za-z]+|.)", re.S)


class TexRejected(ValueError):
    pass


def commands(tex: str) -> List[str]:
    return _COMMAND.findall(tex or "")


def check_tex(tex: str) -> str:
    """Returns the stripped TeX, or raises TexRejected with a short reason."""
    t = (tex or "").strip()
    if not t:
        raise TexRejected("empty")
    if len(t) > MAX_TEX_LEN:
        raise TexRejected("too_long")
    for c in commands(t):
        if c not in ALLOWED_COMMANDS:
            raise TexRejected(f"command_not_allowed:{c[:20]}")
    if not _ALLOWED_CHARS.match(_COMMAND.sub(" ", t)):
        raise TexRejected("char_not_allowed")
    depth = 0
    for ch in t:
        depth += ch == "{"
        depth -= ch == "}"
        if depth < 0 or depth > MAX_DEPTH:
            raise TexRejected("unbalanced_braces")
    if depth != 0:
        raise TexRejected("unbalanced_braces")
    plain = _COMMAND.sub(" x ", t)
    if re.search(r"[=+\-/^_]\s*$", plain) or re.search(r"^\s*[=+/^_]", plain) or re.search(r"[=+/^_]\s*[=+/^_]", plain):
        raise TexRejected("dangling_operator")
    if t.count("\\left") != t.count("\\right"):
        raise TexRejected("unbalanced_delimiters")
    return t


# How a symbol can be named aloud. Latin letters are matched by exact token; physical quantities by word stem.
_LETTER_NAMES: Dict[str, Tuple[str, ...]] = {
    "a": ("а", "эй", "a"), "b": ("б", "бэ", "бе", "b"), "c": ("цэ", "це", "си", "ц", "c"), "d": ("дэ", "де", "d"),
    "e": ("е", "э", "и", "e"), "f": ("эф", "ф", "f"), "g": ("же", "жэ", "гэ", "г", "g"), "h": ("аш", "эйч", "h"),
    "i": ("и", "ай", "i"), "k": ("ка", "к", "k"), "l": ("эль", "л", "l"), "m": ("эм", "м", "m"), "n": ("эн", "н", "n"),
    "p": ("пэ", "пе", "п", "пи", "p"), "q": ("ку", "q"), "r": ("эр", "р", "r"), "s": ("эс", "с", "s"),
    "t": ("тэ", "те", "т", "t"), "u": ("у", "ю", "u"), "v": ("вэ", "ве", "в", "v"), "w": ("дубль", "w"),
    "x": ("икс", "x"), "y": ("игрек", "y"), "z": ("зет", "z"),
}
_QUANTITY_STEMS: Dict[str, Tuple[str, ...]] = {
    "A": ("работ",), "E": ("энерг",), "F": ("сил", "force"), "I": ("сил", "ток"), "N": ("мощн", "power"),
    "P": ("мощн", "давлен", "power"), "Q": ("теплот", "заряд"), "R": ("сопротивл", "resistan", "радиус"),
    "S": ("площад", "путь", "пути", "расстоян"), "T": ("период", "температур"), "U": ("напряж", "voltage"),
    "V": ("объем", "скорост"), "a": ("ускорен", "катет"), "b": ("катет",), "c": ("гипотенуз", "скорост", "свет"),
    "h": ("высот",), "m": ("масс", "mass"), "s": ("путь", "пути", "расстоян"), "t": ("врем", "time"),
    "v": ("скорост", "velocity"), "g": ("ускорен",), "x": ("координат",),
}
_GREEK_NAMES = {"alpha": "альф", "beta": "бет", "gamma": "гамм", "delta": "дельт", "Delta": "дельт", "lambda": "лямбд",
                "pi": "пи", "rho": "ро", "sigma": "сигм", "tau": "тау", "phi": "фи", "omega": "омег", "Omega": "ом",
                "theta": "тет", "mu": "мю"}
_OPERATORS: Dict[str, Tuple[str, ...]] = {
    "=": ("равн", "равен", "есть", "это", "equal", "is"),
    "+": ("плюс", "сумм", "plus", "sum", "складыва", "прибав"),
    "-": ("минус", "разност", "вычит", "minus"),
    "/": ("дел", "раздел", "отношен", "divided", "over", "поделить"),
    "*": ("умнож", "произведен", "times", "multiplied", "помнож"),
    "^2": ("квадрат", "squared"),
    "^3": ("куб", "cubed"),
    "sqrt": ("корен", "корн", "root"),
}


def atoms(tex: str) -> Tuple[Set[str], Set[str], Set[float]]:
    """(variables, operators, numeric constants) of a TeX formula. An unusual power (^5) counts as a constant to be
    spoken; only squares and cubes have their own words."""
    t = tex
    ops: Set[str] = set()
    if "\\frac" in t or "/" in t:
        ops.add("/")
    if "\\sqrt" in t:
        ops.add("sqrt")
    if "\\cdot" in t or "\\times" in t:
        ops.add("*")
    t2 = re.sub(r"\^\{?2\}?", " ", t)
    if t2 != t:
        ops.add("^2")
    t3 = re.sub(r"\^\{?3\}?", " ", t2)
    if t3 != t2:
        ops.add("^3")
    greek = {c for c in commands(t3) if c in _GREEK_NAMES}
    body = _COMMAND.sub(" ", t3)
    for op in "=+-":
        if op in body:
            ops.add(op)
    variables = set(re.findall(r"[A-Za-z]", body)) | {"\\" + g for g in greek}
    no_subs = re.sub(r"_\{[^}]*\}|_.", " ", body)         # x_1 names a variable, it is not a constant
    nums = {float(n) for n in re.findall(r"[0-9]+(?:\.[0-9]+)?", no_subs)}
    return variables, ops, nums


@dataclass
class SpeechSupport:
    ok: bool
    missing: List[str]


def _name_index(var: str, toks: List[str]) -> Optional[int]:
    """Index of the first word in the quote that names this symbol; None if it is never named."""
    if var.startswith("\\"):
        stem = _GREEK_NAMES[var[1:]]
        return next((i for i, t in enumerate(toks) if t.startswith(stem)), None)
    names = _LETTER_NAMES.get(var.lower(), (var.lower(),))
    stems = _QUANTITY_STEMS.get(var, ())
    for i, t in enumerate(toks):
        if any(t.startswith(s) for s in stems):
            return i
        if t not in names:
            continue
        # one-letter Cyrillic words are mostly prepositions ("в", "с", "у", "и", "а"): they only count as a symbol
        # when the speech is reading a formula out, i.e. an operator word is right next to them
        if len(t) > 1 or t.isascii():
            return i
        near = toks[max(0, i - 2):i + 3]
        if any(n.startswith(p) for n in near for p in ("равн", "равен", "умнож", "дел", "плюс", "минус", "квадрат")):
            return i
    return None


def _said(op: str, toks: List[str]) -> bool:
    return any(t.startswith(s) for t in toks for s in _OPERATORS[op])


def speech_support(tex: str, quote: str) -> SpeechSupport:
    """Every symbol, operator and constant of the formula must be named in the quote, every operator named in the
    quote must be in the formula, and the structure must read the same way: E = m/c^2 over «эм умножить на цэ в
    квадрате» or t/A over «работа делить на время» does not pass."""
    variables, ops, nums = atoms(tex)
    toks = tokens(quote)
    missing: List[str] = []
    if "=" not in ops:
        missing.append("relation")
    index: Dict[str, int] = {}
    for v in sorted(variables):
        i = _name_index(v, toks)
        if i is None:
            missing.append(v)
        else:
            index[v] = i
    for op in sorted(ops):
        if _said(op, toks) or (op == "*" and "на" in toks and not _said("/", toks)):    # «у на и»
            continue
        missing.append("op:" + op)
    for op in ("/", "+", "-", "^2", "^3", "sqrt"):
        if _said(op, toks) and op not in ops:
            missing.append("said_not_written:" + op)
    spoken = extract_numbers(quote)
    for n in sorted(nums):
        if not number_supported(n, spoken):
            missing.append(f"num:{n:g}")
    frac = re.search(r"\\frac\{([A-Za-z])\}\{([A-Za-z])\}|([A-Za-z])\s*/\s*([A-Za-z])", tex)
    if frac:
        top, bottom = (frac.group(1), frac.group(2)) if frac.group(1) else (frac.group(3), frac.group(4))
        if top in index and bottom in index and index[top] > index[bottom]:
            missing.append("order:fraction")
    lhs = re.match(r"\s*([A-Za-z])\s*=", tex)
    if lhs and lhs.group(1) in index and len(index) > 1:
        positions = sorted(index.values())
        if index[lhs.group(1)] not in (positions[0], positions[-1]):
            missing.append("order:left_side")
    return SpeechSupport(not missing, missing)


def canonical(tex: Optional[str]) -> str:
    """Spacing-insensitive form used to compare a formula with the author's materials."""
    return re.sub(r"\s+", "", (tex or "")).replace("\\cdot", "").replace("{", "").replace("}", "")
