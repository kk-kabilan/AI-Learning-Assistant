from sentence_transformers import SentenceTransformer
from playbook import search_playbook_top_k


# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Model loaded.")


# =========================================================
# TEST QUESTIONS
# =========================================================

TEST_CASES = [

    {
        "name": "Exact question",
        "query": "What is the difference between ML research and ML production?",
        "expected": "HIT"
    },

    {
        "name": "Paraphrased question",
        "query": "How does machine learning research differ from deploying ML models in production?",
        "expected": "HIT"
    },

    {
        "name": "Related question",
        "query": "How is machine learning deployment different in research and production?",
        "expected": "HIT"
    },

    {
        "name": "Unrelated question",
        "query": "What is the capital of France?",
        "expected": "MISS"
    },

    {
        "name": "Unrelated question 2",
        "query": "How does photosynthesis work in plants?",
        "expected": "MISS"
    }
]


# =========================================================
# RUN EVALUATION
# =========================================================

hits = 0
correct = 0

print("\n========================================")
print("PLAYBOOK RETRIEVAL EVALUATION")
print("========================================")

for test in TEST_CASES:

    query = test["query"]
    expected = test["expected"]

    print("\n----------------------------------------")
    print("Test:", test["name"])
    print("Query:", query)
    print("Expected:", expected)

    results = search_playbook_top_k(
        query=query,
        model=model,
        k=1,
        threshold=0.65
    )

    if results:

        score = results[0]["score"]

        print("Result: HIT")
        print(f"Similarity: {score:.4f}")

        print(
            "Matched memory:",
            results[0]["entry"].get("question", "")
        )

        actual = "HIT"

    else:

        print("Result: MISS")

        actual = "MISS"

    if actual == "HIT":
        hits += 1

    if actual == expected:
        correct += 1


# =========================================================
# SUMMARY
# =========================================================

total = len(TEST_CASES)

accuracy = correct / total * 100
hit_rate = hits / total * 100

print("\n========================================")
print("RESULTS")
print("========================================")

print(f"Total tests: {total}")
print(f"Correct predictions: {correct}")
print(f"Accuracy: {accuracy:.2f}%")
print(f"Overall hit rate: {hit_rate:.2f}%")