
import math
import re
from nltk.corpus import stopwords

UBIQUITY_RATIO = 0.8  

# Standard operators and universal constants to filter out
NON_VARIABLE_LATEX = {
    r"\infty", r"\pi", r"\int", r"\iint", r"\iiint", r"\oint",
    r"\sum", r"\prod", r"\partial", r"\nabla", r"\cdot", r"\times",
    r"\pm", r"\mp", r"\approx", r"\sim", r"\propto", r"\to",
    r"\rightarrow", r"\leftarrow", r"\mapsto", r"\forall", r"\exists",
    r"\in", r"\otimes", r"\oplus", r"\langle", r"\rangle",
    r"\hbar", r"\mathrm{i}", r"\imath", r"\ldots", r"\dots", r"\cdots",
}

# Regex matchers for numerical values, assignments, and structural tokens
_NUMERIC_RE = re.compile(r"^[\s\d.,+\-*/^_{}()\\]*\d[\s\d.,+\-*/^_{}()\\]*$")
_DEF_RE = re.compile(r"([A-Za-z\\][A-Za-z0-9_^{}\\]*)\s*=")
_STOP = set(stopwords.words("english"))
_PLACEHOLDER_WORDS = {"sym", "eqn", "meqn", "tempeqn"}


def _content_words(text):
    """Extract lowercased alphabetic words, excluding stopwords and placeholders."""
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return {w for w in words if w not in _STOP and w not in _PLACEHOLDER_WORDS}


def _is_single_letter(latex):
    """Check if the symbol is a single alphabetic character."""
    s = latex.strip()
    return len(s) == 1 and s.isalpha()


def _eq_number(eq):
    """Strip prefix to expose the raw equation number."""
    return eq.replace("EQN", "", 1)


def _lhs(latex):
    """Extract the set of symbols defined on the left-hand side of '='."""
    return {m.group(1) for m in _DEF_RE.finditer(latex)}


def _norm(latex):
    """Normalize whitespace out of a LaTeX string for structural comparisons."""
    return re.sub(r"\s+", "", latex)


def _is_meaningful_symbol(latex):
    """Filter out empty text, universal math constants, and standalone digits."""
    s = latex.strip()
    if not s or s in NON_VARIABLE_LATEX or _NUMERIC_RE.match(s):
        return False
    return True


def _ubiquitous_symbols(eqn_order, eq_to_syms):
    """Identify symbols that appear too frequently across equations to be discriminating."""
    n = len(eqn_order)
    if n < 2:
        return set()
    threshold = max(3, math.ceil(UBIQUITY_RATIO * n))
    counts = {}
    for eq in eqn_order:
        for sym in set(eq_to_syms.get(eq, [])):
            counts[sym] = counts.get(sym, 0) + 1
    return {sym for sym, c in counts.items() if c >= threshold}


def get_relations(target_eq, eqn_order, eq_to_syms, sym_mapping, eqn_mapping=None,
                  eq_context=None, audit=None):
    """Classify the relationship between a target equation and all other equations."""
    relations = {}
    
    # Track ubiquitous and standard equation symbols in LaTeX format
    ubiquitous = {sym_mapping.get(s, s) for s in _ubiquitous_symbols(eqn_order, eq_to_syms)}
    target_syms = {sym_mapping.get(s, s) for s in eq_to_syms.get(target_eq, [])}
    target_words = _content_words(eq_context.get(target_eq, "")) if eq_context is not None else set()

    for other_idx, other in enumerate(eqn_order):
        if other == target_eq:
            continue

        other_syms = {sym_mapping.get(s, s) for s in eq_to_syms.get(other, [])}

        # Filter out noise from shared symbols
        shared = (target_syms & other_syms) - ubiquitous
        shared = {s for s in shared if _is_meaningful_symbol(s)}

        # Check if the other equation defines a shared token
        defining = []
        if eqn_mapping is not None and shared:
            defined = {_norm(d) for d in _lhs(eqn_mapping.get(other, {}).get("latex", ""))}
            if defined:
                defining = [s for s in shared if _norm(s) in defined]

        multi = [s for s in shared if not _is_single_letter(s)]
        single = [s for s in shared if _is_single_letter(s)]

        # Determine relation grade
        if defining:
            grade = "strong"
            description = f"special case; equation {_eq_number(other)} defines {', '.join(sorted(defining))} used here"
        elif multi:
            grade = "strong"
            description = f"related; shares symbol(s): {', '.join(sorted(multi))}"
        elif single:
            grade = "potential"
            description = f"shares only single-letter symbol(s): {', '.join(sorted(single))}"
        else:
            grade = "none"
            description = ""

        # Secondary Pass: Upgrade if equations share distinct context terms
        if grade == "none" and eq_context is not None:
            other_words = _content_words(eq_context.get(other, ""))
            common = sorted(target_words & other_words)
            if common:
                grade = "potential"
                description = f"shares context terms: {', '.join(common)}"

        relations[_eq_number(other)] = {"grade": grade, "description": description}

        if audit is not None and grade != "none":
            audit.setdefault("get_relations", []).append(
                f"{_eq_number(target_eq)} -> {_eq_number(other)}: {grade} ({description})"
            )

    return relations