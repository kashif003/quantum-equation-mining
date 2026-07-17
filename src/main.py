from pathlib import Path
import json
from tqdm import tqdm
from nltk.tokenize import sent_tokenize

from utils import paper_ID_extractor, download_html, strip_backslash, get_meanings
from html_parser import HTML_Reader, map_symbols_to_equations
from relations import get_relations
from extract_description import get_description, extract_lhs

# Load assigned papers in order
paper_list = paper_ID_extractor("./paper_list_12.txt")

# Set up local HTML cache directory
current_dir = Path('./data/html_source')
current_dir.mkdir(parents=True, exist_ok=True)

html_files = [item.name[:-5] for item in current_dir.iterdir() if item.is_file()]

# Ensure output directory exists
Path("./results").mkdir(parents=True, exist_ok=True)

# Pipeline configuration
EQUATION_TARGET = 350   
total_equations = 0     
dataset = {}            

for paper_id in paper_list:
    print("[INFO] Downloading the paper:", paper_id)
    downloaded = download_html(paper_id)

    # Handle missing downloads safely
    if not downloaded:
        print("[IMPORTANT] Unable to download the paper:", paper_id)
        dataset[f"arXiv:{paper_id}"] = "Unable to download the paper"
        with open("./results/dataset.json", "w") as file:
            json.dump(dataset, file, indent=4)
        continue

    paper_dataset = {f"arXiv:{paper_id}": {}}

    # Extract clean text, equations, and symbols
    extractor = HTML_Reader(paper_id)
    clean_text, eqn_mapping, sym_mapping = extractor.extract()

    # Map symbols to equations and cache sentences
    eq_to_syms = map_symbols_to_equations(eqn_mapping, sym_mapping)
    equations = list(eqn_mapping.keys())
    paper_sentences = sent_tokenize(clean_text)

    # Create readable LaTeX mapping for audit trails
    name_map = dict(sym_mapping)
    name_map.update({e: d["latex"] for e, d in eqn_mapping.items()})

    print("[INFO] GETTING Meaning of equations......")

    eq_context = {}
    audits = {}
    eq_lhs = {}            
    eq_matched = {}        
    eq_meaning_by_eq = {}  

    # ---------- PASS 1: Extract Meanings per Equation and Symbol ----------
    for i, eq in enumerate(equations):
        index = i + 1  
        if index not in paper_dataset[f"arXiv:{paper_id}"]:
            paper_dataset[f"arXiv:{paper_id}"][index] = {}

        eq_audit = {}
        audits[eq] = eq_audit

        latex = eqn_mapping[eq]["latex"]
        eq_audit["extract_equations_method"] = latex
        
        symbols = eq_to_syms[eq]
        eq_audit["map_symbols_to_equations"] = {latex: [sym_mapping[s] for s in symbols]}
        paper_dataset[f"arXiv:{paper_id}"][index]["equation"] = latex

        # Process individual symbols
        sym_meanings_by_ph = {}
        for sym in symbols:
            sym_meaning = get_meanings(clean_text, sym, audit=eq_audit, name_map=name_map, sentences=paper_sentences)
            sym_meanings_by_ph[sym] = sym_meaning
            
            if "symbols" not in paper_dataset[f"arXiv:{paper_id}"][index]:
                paper_dataset[f"arXiv:{paper_id}"][index]["symbols"] = {}
            paper_dataset[f"arXiv:{paper_id}"][index]["symbols"][strip_backslash(sym_mapping[sym])] = sym_meaning

        # Process equation description
        eq_meaning, conf = get_description(clean_text, eq, audit=eq_audit, name_map=name_map, return_conf=True)

        # Fallback: Check if Left-Hand-Side (LHS) matches a symbol
        lhs = extract_lhs(latex)
        lhs_symbol = None
        if lhs:
            for s in symbols:
                if sym_mapping[s] == lhs:
                    lhs_symbol = s
                    break

        # Replace low-confidence meanings with LHS symbol meaning if available
        if (conf != "high") and lhs_symbol is not None and sym_meanings_by_ph.get(lhs_symbol):
            eq_meaning = sym_meanings_by_ph[lhs_symbol]
            eq_audit.setdefault("equation_meaning_method", {})[latex] = (
                f"Phase-2 result replaced: LHS '{lhs}' matches symbol -> {eq_meaning}"
            )

        paper_dataset[f"arXiv:{paper_id}"][index]["meaning"] = eq_meaning

        # Store properties for the grouping pass
        eq_lhs[eq] = lhs
        eq_matched[eq] = lhs_symbol is not None
        eq_meaning_by_eq[eq] = eq_meaning
        paper_dataset[f"arXiv:{paper_id}"][index]["audit-trail"] = eq_audit
        eq_context[eq] = eq_meaning or ""

    # ---------- PASS 2: Group Equations Sharing the Same LHS ----------
    lhs_groups = {}
    for eq in equations:
        l = eq_lhs.get(eq)
        if l:
            lhs_groups.setdefault(l, []).append(eq)

    for l, group in lhs_groups.items():
        if len(group) < 2:
            continue  

        # Determine the best meaning for the group
        winner = None
        for eq in group:
            if eq_matched.get(eq) and eq_meaning_by_eq.get(eq):
                winner = eq_meaning_by_eq[eq]
                break
        if winner is None:
            for eq in group:
                if eq_meaning_by_eq.get(eq):
                    winner = eq_meaning_by_eq[eq]
                    break
        if winner is None:
            continue  

        # Unify meanings across the group
        for eq in group:
            idx = equations.index(eq) + 1
            paper_dataset[f"arXiv:{paper_id}"][idx]["meaning"] = winner
            eq_context[eq] = winner or ""
            audits[eq].setdefault("equation_meaning_method", {})[eqn_mapping[eq]["latex"]] = (
                f"shares LHS '{l}' -> unified meaning: {winner}"
            )

    # ---------- PASS 2: Map Relations Between Equations ----------
    for i, eq in enumerate(equations):
        index = i + 1
        eq_audit = audits[eq]

        relations = get_relations(eq, equations, eq_to_syms, sym_mapping, eqn_mapping, eq_context=eq_context, audit=eq_audit)
        paper_dataset[f"arXiv:{paper_id}"][index]["relations"] = relations

    # ---------- Save and Update Progress ----------
    dataset[f"arXiv:{paper_id}"] = paper_dataset[f"arXiv:{paper_id}"]

    # Save after each paper to guarantee partial progress is written
    with open("./results/dataset.json", "w") as file:
        json.dump(dataset, file, indent=4)

    total_equations += len(equations)
    print(f"[INFO] {paper_id}: {len(equations)} equations | total so far: {total_equations}")

    # Enforce stop target
    if total_equations >= EQUATION_TARGET:
        print(f"[INFO] Reached {total_equations} equations (>= {EQUATION_TARGET}). Stopping.")
        break

# Final validation save
with open("./results/dataset.json", "w") as file:
    json.dump(dataset, file, indent=4)