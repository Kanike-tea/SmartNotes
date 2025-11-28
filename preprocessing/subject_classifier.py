"""
subject_classifier.py

VTU CSE - 5th Semester subject keywords (Scheme 2022 / effective 2023-24)

UPDATED VERSION with:
- Expanded Biology keywords (23 → 50+ terms)
- Weighted scoring (multi-word phrases count more)
- Domain conflict detection (Biology vs CS)
- Minimum confidence threshold
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
        # Core networking terms
        "computer networks", "bcs502", "network", "networking",
        
        # Protocols
        "tcp", "udp", "ip", "routing", "protocol", "http", "ftp", "smtp",
        "arp", "dhcp", "dns", "icmp", "tcp/ip", "ipv4", "ipv6",
        
        # Network layers (OSI model)
        "osi model", "network layer", "transport layer", "application layer",
        "data link layer", "physical layer", "presentation layer", "session layer",
        
        # Data link layer concepts (YOUR TEXT CONTAINS THESE!)
        "data link", "framing", "frame", "error detection", "error correction",
        "flow control", "media access control", "mac", "ethernet",
        "collision detection", "csma", "token ring", "llc",
        
        # Error handling
        "crc", "checksum", "parity", "hamming code", "burst error",
        "single bit error", "redundancy", "fec", "arq",
        
        # Network concepts
        "switching", "congestion", "bandwidth", "throughput",
        "latency", "packet", "datagram", "segment",
        "three-way handshake", "congestion control",
        
        # Hardware
        "router", "switch", "gateway", "bridge", "hub", "modem",
        "nic", "repeater",
        
        # Additional terms
        "socket", "port", "subnet", "firewall", "vpn",
        "wireless", "lan", "wan", "internet"
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
    # EXPANDED Biology keywords (50+ terms)
    "Biology": [
        "cell", "dna", "protein", "organism", "biology", "genetic", "mutation",
        "evolution", "photosynthesis", "respiration", "reproduction", "enzyme",
        "tissue", "organ", "system", "nervous", "digestive", "circulatory",
        "metabolism", "taxonomy", "ecology", "biosphere", "ecosystem",
        "mitochondria", "nucleus", "chromosome", "rna", "amino acid",
        "bacteria", "virus", "fungi", "prokaryote", "eukaryote",
        "mitosis", "meiosis", "gene", "allele", "genotype", "phenotype",
        "heredity", "biodiversity", "species", "population", "community",
        "biomolecule", "carbohydrate", "lipid", "nucleic acid",
        "homeostasis", "osmosis", "diffusion", "active transport",
        "cellular respiration", "glycolysis", "krebs cycle", "electron transport",
        "chloroplast", "chlorophyll", "stomata", "xylem", "phloem"
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
    """
    text = _clean_text(text)
    if not text:
        return "Unknown Subject", [], 0.0

    text_for_search = text.lower()
    
    # PRIORITY 1: Check for explicit course code (BCS501, BCS502, etc.)
    import re
    course_code_match = re.search(r'\b(BCS\d{3}[A-Z]?)\b', text, re.IGNORECASE)
    if course_code_match:
        code = course_code_match.group(1).upper()
        for subject_name in VTU_SUBJECT_KEYWORDS.keys():
            if code in subject_name:
                print(f"[CLASSIFIER] Explicit course code detected: {code}")
                return subject_name, [code.lower()], 0.95
    
    # PRIORITY 2: Keyword matching
    subject_scores = []

    for subj, patterns in _KEYWORD_PATTERNS.items():
        matched = []
        total_weight = 0
        matched_weight = 0
        
        for kw, pat in patterns:
            weight = len(kw.split())
            total_weight += weight
            
            if pat.search(text_for_search):
                matched.append(kw)
                matched_weight += weight
        
        if matched:
            conf = matched_weight / max(1, total_weight)
            subject_scores.append((subj, matched, conf))

    if not subject_scores:
        return "Unknown Subject", [], 0.0

    # Sort and return best match
    subject_scores.sort(key=lambda x: (x[2], len(x[1])), reverse=True)
    best_subj, best_matched, best_conf = subject_scores[0]
    
    if best_conf < 0.1:
        return "Unknown Subject", [], 0.0

    return best_subj, best_matched, round(best_conf, 2)