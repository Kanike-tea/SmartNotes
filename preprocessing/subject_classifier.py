"""
subject_classifier.py

VTU CSE - 5th Semester subject keywords (Scheme 2022 / effective 2023-24)
Returns:
    subject (str): best matched subject (or "Unknown Subject")
    keywords_matched (list[str]): list of matched keywords found in text
    confidence (float): simple confidence score in [0.0, 1.0]
"""

import re
from collections import Counter

VTU_SUBJECT_KEYWORDS = {
    "BCS501 - Software Engineering & Project Management": [
        "software engineering", "software process", "sdlc", "requirement",
        "software design", "uml", "testing", "verification", "validation",
        "project management", "cost estimation", "cocomo", "risk management",
        "version control", "agile", "waterfall", "maintenance", "scrum",
        "development", "deployment", "team", "planning"
    ],
    "BCS502 - Computer Networks": [
        "computer networks", "tcp", "udp", "routing", "ip", "switching",
        "osi model", "network layer", "transport layer", "congestion",
        "arp", "dhcp", "dns", "mac", "ethernet", "socket", "tcp/ip",
        "three-way handshake", "congestion control", "protocol", "bandwidth",
        "gateway", "firewall", "vpn", "network security", "port"
    ],
    "BCS503 - Theory of Computation": [
        "automata", "finite automata", "dfa", "nfa", "regular expression",
        "regular languages", "context free grammar", "pushdown automata",
        "turing machine", "decidability", "reduction", "p vs np", "grammar",
        "pumping lemma", "complexity", "algorithm", "language", "computation",
        "state machine", "transition"
    ],
    "BCSL504 - Web Technology Lab": [
        "html", "css", "javascript", "dom", "frontend", "http", "web",
        "rest", "ajax", "node", "flask", "php", "web api", "forms",
        "responsive design", "bootstrap", "server", "client", "browser",
        "markup", "styling", "scripting", "database"
    ],
    "BCS515A - Computer Graphics (Professional Elective)": [
        "computer graphics", "rasterization", "opengl", "transformations",
        "2d graphics", "3d graphics", "clipping", "modeling", "lighting",
        "shading", "bezier", "bresenham", "projection", "rendering",
        "animation", "texture", "polygon", "image"
    ],
    "BCS515B - Artificial Intelligence (Professional Elective)": [
        "artificial intelligence", "search", "a*", "heuristic", "knowledge",
        "prolog", "machine learning", "classification", "expert system",
        "agents", "bayes", "neural network", "deep learning", "ai",
        "learning", "reasoning", "inference", "optimization"
    ],
    "BCS515C - Unix System Programming (Professional Elective)": [
        "unix", "system calls", "fork", "exec", "signals", "process",
        "interprocess communication", "pipes", "threads", "makefile",
        "shell scripting", "file descriptors", "linux", "operating system",
        "kernel", "ipc", "semaphore", "mutex"
    ],
    "BCS515D - Distributed Systems (Professional Elective)": [
        "distributed system", "distributed algorithms", "rpc", "consensus",
        "lamport", "mutual exclusion", "replication", "distributed file system",
        "clock synchronization", "cap theorem", "cluster", "parallel",
        "message passing", "coordination", "fault tolerance"
    ],
    "BCS586 - Mini Project": [
        "mini project", "project", "prototype", "implementation",
        "report", "presentation", "demo", "mini-project", "development"
    ],
    "BRMK557 - Research Methodology and IPR": [
        "research methodology", "ipr", "patent", "literature review",
        "research design", "ethics", "plagiarism", "citation", "bibliography",
        "research", "intellectual property", "publication"
    ],
    "BCS508 - Environmental Studies and E-waste Management": [
        "environmental studies", "e-waste", "sustainability", "pollution",
        "waste management", "environment", "recycling", "hazardous waste",
        "ecology", "conservation", "climate", "green"
    ],
    # Adding common college subject patterns for better fallback
    "Biology": [
        "cell", "dna", "protein", "organism", "biology", "genetic", "mutation",
        "evolution", "photosynthesis", "respiration", "reproduction", "enzyme",
        "tissue", "organ", "system", "nervous", "digestive", "circulatory",
        "metabolism", "taxonomy", "ecology", "biosphere", "ecosystem"
    ],
    "Chemistry": [
        "atom", "molecule", "chemical", "reaction", "element", "compound",
        "bonding", "periodic table", "acid", "base", "salt", "oxidation",
        "reduction", "equilibrium", "kinetics", "thermodynamics", "organic",
        "inorganic", "polymer", "solution"
    ],
    "Physics": [
        "force", "energy", "motion", "wave", "quantum", "particle", "physics",
        "velocity", "acceleration", "momentum", "work", "power", "heat",
        "temperature", "light", "optics", "electricity", "magnetism",
        "relativity", "gravity", "potential"
    ],
    "Mathematics": [
        "equation", "calculus", "algebra", "geometry", "derivative", "integral",
        "function", "limit", "series", "matrix", "vector", "trigonometry",
        "probability", "statistics", "theorem", "proof", "number theory"
    ]
}

# Precompile regex patterns for each keyword for speed + safer matching
def _build_keyword_patterns(vtu_keywords):
    patterns = {}
    for subj, keywords in vtu_keywords.items():
        compiled = []
        for kw in keywords:
            # escape special chars and match word boundaries for multi-word phrases too
            esc = re.escape(kw.lower())
            # Use lookaround to match phrase boundaries (allow punctuation/space around)
            pat = re.compile(rf'(?<!\w){esc}(?!\w)', flags=re.IGNORECASE)
            compiled.append((kw, pat))
        patterns[subj] = compiled
    return patterns

_KEYWORD_PATTERNS = _build_keyword_patterns(VTU_SUBJECT_KEYWORDS)


def _clean_text(text):
    if not text:
        return ""
    # normalize whitespace and lowercasing (we keep punctuation to help phrase matching)
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text


def classify_subject(text):
    """
    Given OCR text (string), return a tuple:
      (best_subject:str, keywords_matched:list[str], confidence:float)
    Confidence is computed as: matched_count / number_of_keywords_for_subject (clamped to 1.0)
    If no subject matches, returns ("Unknown Subject", [], 0.0)
    """
    text = _clean_text(text)
    if not text:
        return "Unknown Subject", [], 0.0

    subject_scores = []
    text_for_search = text.lower()

    # Count matches per subject
    for subj, patterns in _KEYWORD_PATTERNS.items():
        matched = []
        for kw, pat in patterns:
            if pat.search(text_for_search):
                matched.append(kw)
        if matched:
            # confidence: fraction of keywords matched (simple heuristic)
            conf = len(matched) / max(1, len(VTU_SUBJECT_KEYWORDS[subj]))
            subject_scores.append((subj, matched, conf))

    if not subject_scores:
        return "Unknown Subject", [], 0.0

    # Choose subject with highest confidence; tie-breaker by most matches
    subject_scores.sort(key=lambda x: (x[2], len(x[1])), reverse=True)
    best_subj, best_matched, best_conf = subject_scores[0]

    # Round confidence to 2 decimals for readability
    return best_subj, best_matched, round(best_conf, 2)


# Simple helper wrapper to preserve previous single-value API if needed
def classify_subject_simple(text):
    subj, matched, conf = classify_subject(text)
    return subj


# Example usage (for quick local test):
# if __name__ == "__main__":
#     s, k, c = classify_subject("Explain TCP three-way handshake and congestion control")
#     print(s, k, c)
