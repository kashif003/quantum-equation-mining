import os
import re
import time
import requests
import nltk
from nltk.tokenize import sent_tokenize
from extract_description import get_description

# Ensure required NLTK tokenizer data is available locally
nltk.download("punkt")
nltk.download("punkt_tab")


def paper_ID_extractor(path, n=None):
    """Extract arXiv paper IDs from a text file by splitting after the first colon."""
    paper_list = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()         
            if line:                     
                paper_list.append(line.split(":", 1)[1])
    return paper_list[:n] if n else paper_list


def download_html(arxiv_id: str, save_dir: str = "./data/html_source") -> bool:
    """
    Download HTML version of an arXiv paper with a single-retry mechanism for network errors.
    
    - Network drops/timeouts: Wait 15s and retry once.
    - Bad request (e.g., status 404): Fail immediately without retrying.
    """
    time.sleep(3)  # Polite rate limiting
    url = f"https://arxiv.org/html/{arxiv_id}"

    response = None
    try:
        response = requests.get(url, allow_redirects=True)
    except requests.RequestException:
        time.sleep(15)  # Network failure retry delay
        try:
            response = requests.get(url, allow_redirects=True)
        except requests.RequestException:
            return False

    if response.status_code != 200:
        return False

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{arxiv_id}.html")
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(response.text)

    return True


def get_sentences_around_label(text, label, window=1, sentences=None):
    """
    Extract a window of sentences surrounding a specific equation or symbol label.
    Accepts pre-tokenized sentences to eliminate redundant whole-paper processing.
    """
    if sentences is None:
        sentences = sent_tokenize(text)

    main_marker = label
    mention_marker = f"M{label}"

    main_re = re.compile(r"\b" + re.escape(main_marker) + r"\b")
    mention_re = re.compile(r"\b" + re.escape(mention_marker) + r"\b")

    result = {"main_context": [], "mention_context": [], "window": window}
    for i, sent in enumerate(sentences):
        start = max(0, i - window)
        end = min(len(sentences), i + window + 1)
        context = " ".join(sentences[start:end])

        if main_re.search(sent):
            result["main_context"].append(context)
        if mention_re.search(sent):
            result["mention_context"].append(context)
    return result


def strip_backslash(s):
    """Remove backslashes from a LaTeX string to make it safe for use as a JSON key."""
    return s.replace("\\", "")


def get_meanings(clean_text, label, audit=None, name_map=None, sentences=None):
    """
    Extract description for a label, checking main context before falling back to mentions.
    Uses cached sentences to bypass repeated tokenization steps.
    """
    full_context = get_sentences_around_label(clean_text, label, sentences=sentences)
    
    # Try main structural context
    main_context = " ".join(full_context["main_context"])
    eq_disc = get_description(main_context, label, audit=audit, name_map=name_map)
    
    # Fallback to textual mention context if description is missing
    if eq_disc is None:
        mention_context = " ".join(full_context["mention_context"])
        eq_disc = get_description(mention_context, label, audit=audit, name_map=name_map)
        
    return eq_disc