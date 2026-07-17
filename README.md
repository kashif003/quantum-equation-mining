# Quantum Equation Retrieval & Relation Mapping

A rule-based NLP pipeline that automatically extracts **enumerated equations**, their **symbols**, plain-language **meanings**, and **inter-equation relations** from quantum physics papers on arXiv — producing a structured, graph-ready JSON dataset with a full audit trail for every extracted value. Unlike a purely LLM-based approach, no language model is ever prompted to *generate* content here: descriptions and relations come from structural HTML parsing and grammatical (dependency-parse) analysis, with spaCy used strictly as a parser.

> **Course project:** Ostbayerische Technische Hochschule (OTH) Amberg-Weiden, Faculty of Electrical Engineering, Media and Computer Science — Masters in Artificial Intelligence for Industrial Applications, 5th semester. Exam ID: `Exam-NLP-S26-ID_12`. Supervisor: Prof. Dr. Patrick Levi. Author: **Kashif Riyaz**.

## Motivation

Quantum physics papers are dense with equations, and every symbol within them carries a specific meaning. Working this out by hand is manageable for one paper but doesn't scale across a literature review. Given a list of arXiv paper IDs, this pipeline downloads each paper's HTML rendering, extracts every enumerated equation, identifies its symbols, infers a short natural-language meaning for both symbols and equations, and classifies how each equation relates to the others in the same paper.

## Pipeline overview

```
paper_list_12.txt (arXiv IDs)
        │
        ▼
main.py — for each paper:
        │
        ├─► utils.download_html()        → fetches https://arxiv.org/html/{id}, caches to data/html_source/
        │
        ├─► html_parser.HTML_Reader       → Phase 1: parses the HTML, extracts symbols/equations, replaces
        │      .extract()                    them with placeholders (SYM1, EQN1, MEQN1, TEMPEQN), returns
        │                                     clean placeholder text + equation/symbol dictionaries
        │
        ├─► html_parser.map_symbols_to_equations()
        │                                  → links each symbol to the equation(s) it appears in
        │
        ├─► utils.get_meanings() / extract_description.get_description()
        │      (Phase 2)                  → infers a plain-language meaning for each symbol and equation
        │                                     from its surrounding sentence context, via grammar rules first,
        │                                     falling back to nearest-noun-phrase proximity scoring
        │
        ├─► extract_description.extract_lhs() + LHS-matching pass
        │                                  → equations sharing the same left-hand side inherit a single,
        │                                     unified meaning from whichever has the highest-confidence one
        │
        ├─► relations.get_relations()      → Phase 3: grades every pair of equations in the paper as
        │                                     "strong" / "potential" / "none" based on shared, non-trivial symbols
        │                                     or shared description vocabulary
        │
        └─► writes/updates results/dataset.json after every paper (so partial progress is never lost)
```

The pipeline stops once it has collected 350 equations total (`EQUATION_TARGET` in `main.py`), processing papers from `paper_list_12.txt` in order.

### Phase 1 — HTML parsing (`html_parser.py`)

Rather than scraping arXiv's LaTeX source (disallowed by `robots.txt`) or reconstructing math from a rendered PDF (unreliable), the pipeline parses **arXiv's rendered HTML**, which wraps equations and symbols in distinct, unambiguous semantic tags:
- **Symbols** — read from the `alttext` attribute of `<math>` tags, each distinct symbol assigned a placeholder (`SYM1`, `SYM2`, ...) in order of appearance; standalone numbers are skipped.
- **Equations** — enumerated equations identified by their container ID (e.g. `id="S1.E1"`), labeled by printed number (equation (1) → `EQN1`). Non-enumerated/over-limit math becomes `TEMPEQN`; textual references to an equation become `MEQN1`.
- **Symbol-to-equation mapping** — boundary-aware matching (so a bare `a` doesn't spuriously match inside `\alpha`).

### Phase 2 — Description extraction (`extract_description.py`, `utils.get_meanings`)

For each symbol/equation, the surrounding sentence context (with a fallback to the sentence around its textual *mention*, if the main context yields nothing) is analyzed grammatically:
1. **Grammar rules (high confidence)** — dependency-parse patterns like a copula ("ψ is the wave function"), passive clause ("the amplitude is given by A"), appositive ("A, the amplitude,"), or prepositional governor ("the wave function for ψ").
2. **Proximity fallback (low confidence)** — nearby noun phrases scored by direction/distance, weighting phrases to the left more heavily.
3. **LHS unification** (`main.py`, Pass 2) — equations that share the same left-hand side (split at `=`) are grouped, and the whole group adopts the single best (highest-confidence) meaning found among them, so a low-confidence equation can inherit a high-confidence one's description.

All candidate descriptions are cleaned and filtered to reject vague heads ("function", "equation"), references, citations, bare numbers, and pronouns.

### Phase 3 — Relation classification (`relations.py`)

Every pair of equations in a paper is graded **strong**, **potential**, or **none**: a pair is related when it shares something *specific to that paper*, not something trivially shared by every equation.
- **Filtering** — ubiquitous symbols (in ≥80% of a paper's equations), non-variable tokens (numbers, operators, constants like `\pi`), and bare single-letter variables (weak evidence only) are excluded from consideration.
- **Grading** — **strong** if a pair shares a distinctive multi-character variable, or one equation defines a symbol the other uses; **potential** if they share only a single-letter symbol, or share a meaningful description word; **none** otherwise.

### Audit trail

Every step logs which method/rule produced each stored value (the matched context sentence, firing rule, and confidence level) directly into the dataset, so every entry can be traced back to the source text it came from.

## Results (from `report.pdf`)

Running against `paper_list_12` (keeping the first 7 numbered equations per paper):

| Quantity | Value |
|---|---|
| Papers processed | 78 |
| ...with no arXiv HTML source | 16 |
| ...with no enumerated equations | 7 |
| ...contributing equations | 55 |
| **Equations in dataset** | **350** |
| Equations with no matched symbol | 4 (1.1%) |
| Descriptions from grammar rules (Phase 1/high-confidence) | 57% |
| Descriptions from fallback (low-confidence) | 43% |

Equation extraction was fully reliable (zero false positives, since arXiv's HTML tags equations unambiguously). Symbol matching succeeded in 98.9% of cases — rare failures stem from small LaTeX formatting differences (e.g. an extra backslash) between a symbol's standalone and in-equation form. Description quality is the weakest link, since no LLM is used to fill gaps in author-stated meaning — it works well when authors state a symbol's meaning plainly, and can misfire (e.g. latching onto a reference like "Appendix B") when they don't.

## Repository structure

```
quantum-equation-mining/
├── README.md               # (currently just the repo title — this document is meant to replace it)
├── read_me.txt             # Quick-start run instructions
├── requirements.txt        # Pinned dependencies
├── .gitignore              # Ignores data/, logs/, venv/, src/__pycache__, results/, report
├── report.pdf              # Full project write-up (introduction, methodology, results, discussion, references)
├── dataset.json            # Committed snapshot of the output dataset
├── paper_list_12.txt       # ~26,900 arXiv IDs (format "arXiv:XXXX.XXXXX"); pipeline processes them in order
└── src/
    ├── main.py               # Orchestrates the full pipeline end-to-end (see flow above)
    ├── html_parser.py         # HTML_Reader class + map_symbols_to_equations() — Phase 1
    ├── extract_description.py # Grammar-rule + fallback description extraction — Phase 2
    ├── relations.py           # get_relations() equation-pair grading — Phase 3
    └── utils.py               # Paper ID loading, HTML downloading, sentence-context windowing, get_meanings()
```

**Note:** `report.pdf` is listed in `.gitignore` (under `report`), meaning it's tracked in the repo currently but would be excluded from future commits if regenerated — likely intentional so a stale local copy doesn't silently overwrite the submitted version, but worth double-checking that's the intended behavior.

## Requirements

Pinned in `requirements.txt`:
- `spacy>=3.7,<3.8` + `spacy-transformers` (with `en_core_web_sm` and `en_core_web_trf` model wheels, pinned to matching 3.7.x versions)
- `torch`, `transformers`, `sentencepiece`
- `nltk` (sentence tokenization — `punkt`/`punkt_tab`, downloaded automatically by `utils.py`)
- `scikit-learn`
- `beautifulsoup4` (HTML parsing)
- `boto3`
- `bertviz`
- `numpy<2`
- `tqdm` (used in `main.py` but not currently listed — see cleanup notes)

## Getting started

As documented in `read_me.txt`:

```bash
# 1. Create a virtual environment
python -m venv venv

# 2. Activate it and install dependencies
source venv/bin/activate        # macOS/Linux
pip install -r requirements.txt

# 3. Run the pipeline
python src/main.py
```

This reads `paper_list_12.txt`, downloads each paper's HTML to `data/html_source/`, and writes/updates `results/dataset.json` after every paper processed — so if it's interrupted, partial progress is preserved.

## AI tool use declaration

As stated in the report: Claude Opus was used to help refine wording and add docstrings to the code, but was **not** used to generate any part of the dataset — all equation, symbol, description, and relation extraction was performed by the rule-based pipeline without prompting any language model.
