# subject_classifier.py — (Anushka Backend)

VTU_SUBJECT_KEYWORDS = {
    "Operating Systems (18CS44)": [
        "deadlock", "semaphore", "paging", "thrashing", "synchronization",
        "process", "cpu scheduling", "fifo", "lru"
    ],
    "DBMS (18CS53)": [
        "normalization", "schema", "er diagram", "acid", "transaction",
        "sql", "join", "relational", "primary key"
    ],
    "DAA (18CS42)": [
        "divide and conquer", "greedy", "dynamic programming",
        "knapsack", "spanning tree", "avl", "sorting"
    ],
    "Computer Networks (18CS52)": [
        "tcp", "udp", "routing", "osi model", "congestion", "network layer"
    ],
    "OOPJ (Java)": [
        "class", "object", "inheritance", "polymorphism", "java"
    ]
}


def classify_subject(text):
    text_lower = text.lower()

    for subject, keywords in VTU_SUBJECT_KEYWORDS.items():
        for word in keywords:
            if word.lower() in text_lower:
                return subject

    return "Unknown Subject"
