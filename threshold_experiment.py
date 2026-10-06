from sentence_transformers import SentenceTransformer
from playbook import search_playbook_top_k


print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Model loaded.")


# =========================================================
# TEST DATA
# =========================================================

TEST_CASES = [

    # Relevant queries
    {
        "query": "What is the difference between ML research and ML production?",
        "expected": "HIT"
    },

    {
        "query": "How does machine learning research differ from deploying ML models in production?",
        "expected": "HIT"
    },

    {
        "query": "How is machine learning deployment different in research and production?",
        "expected": "HIT"
    },

    # Irrelevant queries
    {
        "query": "What is the capital of France?",
        "expected": "MISS"
    },

    {
        "query": "How does photosynthesis work in plants?",
        "expected": "MISS"
    }
]


# =========================================================
# THRESHOLDS
# =========================================================

THRESHOLDS = [
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80
]


# =========================================================
# EXPERIMENT
# =========================================================

print("\n========================================")
print("PLAYBOOK THRESHOLD EXPERIMENT")
print("========================================")


for threshold in THRESHOLDS:

    correct = 0
    false_positives = 0
    false_negatives = 0

    print("\n----------------------------------------")
    print(f"THRESHOLD = {threshold}")
    print("----------------------------------------")

    for test in TEST_CASES:

        results = search_playbook_top_k(
            query=test["query"],
            model=model,
            k=1,
            threshold=threshold
        )

        actual = "HIT" if results else "MISS"

        if actual == test["expected"]:
            correct += 1

        if test["expected"] == "MISS" and actual == "HIT":
            false_positives += 1

        if test["expected"] == "HIT" and actual == "MISS":
            false_negatives += 1

        print(
            f"{actual:4} | "
            f"Expected: {test['expected']:4} | "
            f"{test['query']}"
        )

    accuracy = correct / len(TEST_CASES) * 100

    print("\nAccuracy:", f"{accuracy:.2f}%")
    print("False positives:", false_positives)
    print("False negatives:", false_negatives)