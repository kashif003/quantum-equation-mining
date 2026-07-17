import re

SYMBOL_RE = re.compile(r"^(?:SYM|EQN|MEQN)\d+$")
NOUN_POS = {"NOUN", "PROPN"}

DEF_VERB_LEMMAS = {
    "give", "define", "describe", "express", "write", "compute", "calculate",
    "obtain", "denote", "represent", "quantify", "yield", "capture", "read",
}
PREP_DEF = {"by", "as", "in", "from", "via", "through"}
EXPAND_PREPS = {"for", "of", "in", "with", "about", "at"}

# --- noise filters --------------------------------------------------------
PRONOUNS = {"it", "we", "they", "i", "he", "she", "you", "one", "ones"}
VAGUE_HEADS = {
    "expression", "expressions", "equation", "relation", "relations",
    "form", "forms", "formula", "result", "results", "output", "outputs",
    "quantity", "quantities", "term", "terms", "value", "values",
    "function", "functions", "case", "cases", "thing", "things",
    "approach", "approaches", "factor", "factors",
}
REF_WORDS = {
    "fig", "figs", "figure", "figures", "table", "tables", "tab",
    "eq", "eqn", "equation", "section", "sec", "panel", "panels",
    "appendix", "ref", "refs",
}

# --- Text Cleaning & Noise Reduction Patterns ---
_LABEL_RE = re.compile(r"^\(?[a-zA-Z0-9]{1,3}\)\s+")
_DEMO_RE = re.compile(r"^(this|that|these|those)\s+", re.I)
_ART_RE = re.compile(r"^(the|a|an)\s+", re.I)
_ORD_RE = re.compile(r"\s-(?:th|st|nd|rd)\b", re.I)
_TAILPREP_RE = re.compile(r"\s+(in|of|to|on|for|with|by|from|as|at|into|over|between)$", re.I)
_REF_RE = re.compile(
    r"^(figures?|figs?|tables?|tabs?|eqs?|eqns?|equations?|sections?|secs?|"
    r"appendix|app|panels?)\b\.?\s*\d", re.I)
_CITE_RE = re.compile(r"^(?:auto)?cite\w+$|^[a-z]+\d{4}[a-z]?$", re.I)
_VALUE_RE = re.compile(r"^[\d.,\s×x*+\-/()]*\d[\d.,\s×x*+\-/()]*[a-zA-Z%]{0,4}$")
_ID_RE = re.compile(r"^[A-Z]{2,5}\d+[a-zA-Z]?$")

_NLP = None


def _get_nlp(model="en_core_web_trf"):
    """
    Loads and caches the requested spaCy language model.
    Falls back to a smaller model if the transformer model isn't installed.
    """
    global _NLP
    if _NLP is None:
        import spacy
        spacy.prefer_gpu()          # use GPU if available; silently CPU if not
        try:
            _NLP = spacy.load(model)
        except OSError:
            print(f"⚠️ Model '{model}' not found. Falling back to 'en_core_web_sm'.")
            _NLP = spacy.load("en_core_web_sm")
    return _NLP


def extract_lhs(equation_text):
    """
    Extracts the Left-Hand Side (LHS) of a mathematical expression.
    """
    if not equation_text:
        return None
    # Split the equation at standard mathematical relation operators
    match = re.split(r'(?:=|\\approx|\\equiv|\\sim|\\propto)', equation_text)
    if match and len(match) > 1:
        return match[0].strip() # Return everything before the first operator
    return None


def check_duplicate_lhs(name_map):
    """
    Finds placeholder symbols that share the exact same Left-Hand Side (LHS).
    Returns a dictionary mapping a placeholder to its duplicates.
    """
    if not name_map:
        return {}
    
    #  Groups all placeholders by their extracted LHS string
    lhs_to_ph = {}
    for ph, eq_text in name_map.items():
        lhs = extract_lhs(eq_text)
        if lhs:
            lhs_to_ph.setdefault(lhs, []).append(ph)
            
    #  Identify groups with more than one item and map out the duplicates
    ph_to_duplicates = {}
    for lhs, ph_list in lhs_to_ph.items():
        if len(ph_list) > 1:
            for ph in ph_list:
                # Store every other placeholder in the list except the current one
                ph_to_duplicates[ph] = [item for item in ph_list if item != ph]
    return ph_to_duplicates



def _is_symbol(tok):
    """
    Checks if a token text matches predefined placeholder codes (e.g., SYM0, EQN1).
    """
    return bool(SYMBOL_RE.match(tok.text))


def _is_person(tok):
    """
    Checks if spaCy tagged the token as a person entity name.
    """
    return tok.ent_type_ == "PERSON"


def _is_pron(tok):
    """
    Checks if a token's part-of-speech tag indicates it's a pronoun.
    """
    return tok.pos_ == "PRON"


def _is_verb(tok):
    """
    Checks if a token's part-of-speech tag indicates it's a verb.
    """
    return tok.pos_ == "VERB"


def _bad_anchor(tok):
    """
    Checks if a token is an invalid anchor (e.g. is a placeholder symbol, name, or pronoun).
    """
    return _is_symbol(tok) or _is_person(tok) or _is_pron(tok)


def _clean_desc(desc):
    """
    Cleans up description strings by stripping brackets, articles, and hanging prepositions.
    """
    if not desc:
        return desc
    desc = desc.lstrip("\\([{ \t").strip()
    desc = _LABEL_RE.sub("", desc).strip()      # Remove leading list labels like "(a) "
    desc = _DEMO_RE.sub("", desc).strip()       # Remove demonstratives like "this ", "that "
    desc = _ART_RE.sub("", desc).strip()        # Remove articles like "the ", "a "
    desc = _ORD_RE.sub("", desc).strip()        # Remove ordinal flags like "-th"
    desc = _TAILPREP_RE.sub("", desc).strip()   # Remove trailing isolated prepositions
    return desc

def _is_meaningless(desc, chunk=None):
    """
    Returns True if the description string matches noisy or useless terms (like page refs or single generic words).
    """
    if not desc:
        return True
    low = desc.lower().strip()
    if SYMBOL_RE.match(desc):
        return True
    if _REF_RE.match(low):
        return True
    if _CITE_RE.match(low):
        return True
    if _VALUE_RE.match(desc):
        return True
    if _ID_RE.match(desc):
        return True

    core = re.sub(r"^(the|a|an)\s+", "", low).strip()

    # VAGUE HEAD CHECK (Definition 1): discard ONLY if the description is
    # exactly a vague word on its own (after stripping a leading article) e.g: "fucntion"
    if core in VAGUE_HEADS:
        return True

    if core in PRONOUNS or core in REF_WORDS or low in PRONOUNS or low in REF_WORDS:
        return True
    return False

def _find_symbol_token(doc, symbol):
    """
    Locates the specific token object matching the target symbol string in a parsed document.
    """
    for tok in doc:
        if tok.text == symbol:
            return tok
    return None


def _find_or_alias(anchor):
    """
    Looks within an anchor's tree for an 'or' conjunction alias (e.g., 'the matrix, or array, SYM').
    """
    doc = anchor.doc
    for t in anchor.subtree:
        # Check if we encounter an expression like ", or [noun]"
        if (t.dep_ == "cc" and t.lower_ == "or"
                and t.i - 1 >= 0 and doc[t.i - 1].text == ","):
            cand = t.head
            if cand.pos_ in NOUN_POS and not _is_symbol(cand):
                return cand
    return None


def _refine_anchor(anchor):
    """
    Refines the targeted descriptive token by resolving structural aliases.
    """
    alias = _find_or_alias(anchor)
    if alias is not None:
        return alias
    return anchor


def _anchor(tok):
    """
    Evaluates linguistic grammar dependencies around the symbol token to locate its descriptive noun anchor.
    """
    dep = tok.dep_
    head = tok.head

    # passive defining clause: 'X is given/defined by SYM'
    if dep == "pobj" and head.lemma_.lower() in PREP_DEF:
        part = head.head
        if part.tag_ in {"VBN", "VBD"} or part.lemma_ in DEF_VERB_LEMMAS:
            for c in part.children:
                if c.dep_ in {"nsubjpass", "nsubj"} and c.pos_ in NOUN_POS:
                    return c, "computes", "passive_def"
            if part.dep_ in {"acl", "relcl"} and part.head.pos_ in NOUN_POS:
                return part.head, "computes", "passive_def_relcl"

    # subject of copula / defining verb: 'SYM is the X' / 'SYM gives X'
    if dep in {"nsubj", "nsubjpass"}:
        verb = head
        for c in verb.children:
            if c.dep_ in {"attr", "oprd"} and c.pos_ in NOUN_POS:
                return c, "denotes", "copula"
        if verb.lemma_ in DEF_VERB_LEMMAS:
            for c in verb.children:
                if c.dep_ in {"dobj", "attr", "oprd"} and c.pos_ in NOUN_POS:
                    return c, "computes", "active_def"

    # INVERTED COPULA: 'The quantity is SYM'
    if dep in {"attr", "oprd"} and head.pos_ == "VERB":
        for c in head.children:
            if c.dep_ in {"nsubj", "nsubjpass"} and c.pos_ in NOUN_POS:
                return c, "denotes", "inverted_copula"

    # EXPANDED PREPOSITIONAL ANCHORS: 'The amplitude for SYM'
    if dep == "pobj" and head.lower_ in EXPAND_PREPS:
        prep_gov = head.head
        if prep_gov.pos_ in NOUN_POS and not _is_symbol(prep_gov):
            return prep_gov, "denotes", "prepositional_governor"

    # SYM has an appositive child: 'SYM, the X' / 'SYM (the X)'
    for c in tok.children:
        if c.dep_ == "appos" and c.pos_ in NOUN_POS:
            return c, "denotes", "appos_child"

    #  SYM attaches to a noun head: 'the X SYM' (trailing symbol)
    if dep in {"appos", "compound", "flat", "nmod", "nummod",
               "dep", "npadvmod", "conj", "amod"} and head.pos_ in NOUN_POS:
        return head, "denotes", "trailing_np"

    # parser made the PROPN symbol the chunk head; grab its noun modifier
    noun_mods = [c for c in tok.children
                 if c.dep_ in {"compound", "amod", "nmod", "appos"}
                 and c.pos_ in NOUN_POS and not _is_symbol(c)]
    if noun_mods:
        return noun_mods[-1], "denotes", "head_flip"

    return None, None, None


def _short_chunk(chunk, symbol_i):
    """
    Determines if a noun phrase chunk contains at most one non-symbol noun.
    """
    noun_ct = sum(1 for t in chunk
                  if t.pos_ in NOUN_POS and t.i != symbol_i and not _is_symbol(t))
    return noun_ct <= 1


def _extend_of_pp(doc, chunk, desc, chunks, symbol_i):
    """
    Extends a description string to include a following 'of' prepositional phrase.
    """
    if not _short_chunk(chunk, symbol_i):
        return desc
    j = chunk.end
    # Check if the text directly following this chunk starts with "of"
    if j < len(doc) and doc[j].lower_ == "of":
        for c2 in chunks:
            if c2.start == j + 1:
                extra = "".join(t.text_with_ws for t in c2
                                if t.i != symbol_i and not _is_symbol(t)).strip()
                if extra:
                    return (desc + " of " + extra).strip()
            if c2.start > j + 1:
                break
    return desc


def _prepend_of_governor(doc, chunk, desc, chunks, symbol_i):
    """
    Resolves prepositions by prepending the governing noun phrase structure.
    If 'of', keeps both components; for other prepositions, extracts just the governor.
    """
    start = chunk.start            # index of chunk's first token
    prev_i = start - 1             # token right before the chunk
    if prev_i < 0:
        return desc

    prev = doc[prev_i].lower_      # that token, lowercased
    if prev not in EXPAND_PREPS:   # not a preposition we handle
        return desc

    # governor = the noun chunk ending right before the preposition
    gov = ""
    for c2 in chunks:
        if c2.end == prev_i:
            gov = "".join(t.text_with_ws for t in c2
                          if t.i != symbol_i and not _is_symbol(t)).strip()
            break
    if not gov:
        return desc

    if prev == "of":
        if not _short_chunk(chunk, symbol_i):   # keep old 'of' guard
            return desc
        return (gov + " of " + desc).strip()

    # for/in/with/about/at -> governor is the real head
    return gov

def _np_and_head(doc, anchor, symbol_i, token_to_chunk, chunks):
    """
    Extracts the written phrase text and identifying head token matching the resolved noun anchor.
    """
    chunk = token_to_chunk.get(anchor.i) or token_to_chunk.get(symbol_i)
    if chunk is None:
        # Fallback to robust token tree extraction instead of rigid backtracking loop
        toks = [t for t in anchor.subtree
                if t.i <= anchor.i and t.i != symbol_i and not _is_symbol(t)]
        if not toks:
            toks = [anchor]
        return "".join(t.text_with_ws for t in toks).strip(), anchor

    head_tok = chunk.root
    # Ensure the designated head isn't numeric or an internal symbol code
    if head_tok.i == symbol_i or head_tok.like_num or _is_symbol(head_tok):
        nouns = [t for t in chunk
                 if t.pos_ in NOUN_POS and t.i != symbol_i and not _is_symbol(t)]
        if nouns:
            head_tok = nouns[-1]
    elif anchor.pos_ in NOUN_POS and not _is_symbol(anchor):
        head_tok = anchor

    desc = "".join(t.text_with_ws for t in chunk
                   if t.i != symbol_i and not _is_symbol(t)).strip()
    desc = _extend_of_pp(doc, chunk, desc, chunks, symbol_i)
    desc = _prepend_of_governor(doc, chunk, desc, chunks, symbol_i)
    return desc, head_tok


def _fallback_nearest_np(doc, tok, symbol_i, chunks):
    """
    Heuristic-based fallback strategy that scores nearby noun chunks to find the best 
    description when grammatical extraction rules do not find an exact match.
    """
    best = None
    best_score = float("-inf")

    for chunk in chunks:
        root = chunk.root
        # never describe the symbol with a placeholder/person/pronoun/verb chunk
        if (_is_symbol(root) or _is_person(root)
                or _is_pron(root) or _is_verb(root)):
            continue

        is_left = chunk.end <= symbol_i
        dist = symbol_i - chunk.end if is_left else chunk.start - symbol_i

        # Calculate custom weights based on orientation and proximity features
        score = -dist
        if is_left:
            score += 2
        if is_left and chunk.end == symbol_i:          # immediately left
            score += 4
        if (not is_left) and chunk.start == symbol_i + 1:  # immediately right
            score += 3

        # defining verb between symbol and a right-side chunk
        if not is_left:
            between = [doc[i].lemma_.lower()
                       for i in range(symbol_i + 1, chunk.start)
                       if 0 <= i < len(doc)]
            if any(w in DEF_VERB_LEMMAS for w in between):
                score += 5

        # preposition just before a left-side chunk (e.g. "of <NP>")
        if is_left and chunk.start - 1 >= 0:
            prev = doc[chunk.start - 1].lower_
            if prev in {"of", "with", "by", "as"}:
                score += 3

        # Award points for typical length variations
        length = len([t for t in chunk if t.i != symbol_i and not _is_symbol(t)])
        if 2 <= length <= 4:
            score += 2
        elif length > 6:
            score -= 3

        if score > best_score:
            best_score = score
            best = chunk

    if best is None:
        return None, None

    desc = "".join(t.text_with_ws for t in best
                   if t.i != symbol_i and not _is_symbol(t)).strip()
    desc = _extend_of_pp(doc, best, desc, chunks, symbol_i)
    desc = _prepend_of_governor(doc, best, desc, chunks, symbol_i)
    return desc, best.root


def _latex_context(doc, name_map, target_ph=None):
    """
    Wraps the specific target placeholder string in standard LaTeX ($...$) markdown inside text.
    """
    text = doc.text
    if not name_map or target_ph is None:
        return text
    # only the TARGET placeholder becomes latex, wrapped in $...$.
    # every other placeholder (SYM/EQN/MEQN) is left in its encoded form.
    target_latex = name_map.get(target_ph, target_ph)
    return text.replace(target_ph, f"${target_latex}$")


def _sentence_key(tok, head, name_map, target_ph):
    """
    Constructs a localized sentence string environment containing the symbol for tracking/debugging.
    """
    sents = [tok.sent]
    if head is not None and head.sent.start != tok.sent.start:
        sents.append(head.sent)
    sents.sort(key=lambda s: s.start)
    parts = [_latex_context(s.as_doc(), name_map, target_ph=target_ph)
             for s in sents]
    return " ".join(parts)


def extract_from_doc(doc, symbol, audit=None, name_map=None):
    """
    Core engine function processing an NLP document to find metadata mappings and descriptors for a symbol.
    """
    result = {"symbol": symbol, "relation": None, "description": None,
              "head": None, "rule": None, "confidence": None}
    tok = _find_symbol_token(doc, symbol)
    if tok is None:
        if audit is not None:
            audit.setdefault("extract_symbol_description", {})[
                _latex_context(doc, name_map)] = None
        return result

    chunks = list(doc.noun_chunks)
    token_to_chunk = {t.i: c for c in chunks for t in c}

    # Strategy 1: Attempt to extract using precise syntactic grammar rules
    anchor, relation, rule = _anchor(tok)

    if anchor is not None and _bad_anchor(anchor):
        anchor = None

    if anchor is not None:
        anchor = _refine_anchor(anchor)
        desc, head = _np_and_head(doc, anchor, tok.i, token_to_chunk, chunks)
        desc = _clean_desc(desc)
        
        chunk_obj = token_to_chunk.get(anchor.i)
        head_ok = head is None or head.pos_ in NOUN_POS
        if desc and head_ok and not _is_meaningless(desc, chunk=chunk_obj):
            result.update(relation=relation, description=desc,
                          head=head.text if head is not None else None,
                          rule=rule, confidence="high")
            if audit is not None:
                key = _sentence_key(tok, head, name_map, symbol)
                audit.setdefault("extract_symbol_description", {})[key] = (
                    f"{desc} (rule={rule}, conf=high)"
                )
            return result

    # Strategy 2: Fall back to proximity heuristic if grammar rules yielded nothing
    desc, head = _fallback_nearest_np(doc, tok, tok.i, chunks)
    desc = _clean_desc(desc)
    
    fallback_chunk = token_to_chunk.get(head.i) if head else None
    if desc and not _is_meaningless(desc, chunk=fallback_chunk):
        result.update(relation="denotes", description=desc,
                      head=head.text if head is not None else None,
                      rule="fallback_nearest", confidence="low")
        if audit is not None:
            key = _sentence_key(tok, head, name_map, symbol)
            audit.setdefault("extract_symbol_description", {})[key] = (
                f"{desc} (rule=fallback_nearest, conf=low)"
            )
    elif audit is not None:
        key = _sentence_key(tok, None, name_map, symbol)
        audit.setdefault("extract_symbol_description", {})[key] = None
    return result


def extract_symbol_description(text, symbol, model="en_core_web_trf", audit=None,
                              name_map=None):
    """
    Parses a plain text string with spaCy and extracts structural symbol descriptions.
    """
    nlp = _get_nlp(model)
    return extract_from_doc(nlp(text), symbol, audit=audit, name_map=name_map)


def get_description(text, symbol, model="en_core_web_trf", audit=None,
                   name_map=None, return_conf=False):
    """
    High-level API entry point to fetch the final description string (and optionally confidence metrics) for a target symbol.
    """
    res = extract_symbol_description(text, symbol, model, audit=audit,
                                     name_map=name_map)
    if return_conf:
        return res["description"], res["confidence"]
    return res["description"]