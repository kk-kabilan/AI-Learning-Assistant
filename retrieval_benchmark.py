from sentence_transformers import SentenceTransformer


# =========================================================
# LOAD MODEL
# =========================================================

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Model loaded.")


# =========================================================
# CONTROLLED PLAYBOOK
# =========================================================

PLAYBOOK = [

    {
        "topic": "research_vs_production",
        "question": "What is the difference between ML research and ML production?",
        "knowledge": (
            "ML research focuses on advancing algorithms and improving "
            "model performance, while ML production focuses on reliability, "
            "scalability, deployment, and operational efficiency."
        )
    },

    {
        "topic": "data_handling",
        "question": "How is data handling different in ML research and production?",
        "knowledge": (
            "Research often uses curated datasets, while production "
            "systems handle large-scale, noisy, changing, and real-world data."
        )
    },

    {
        "topic": "model_deployment",
        "question": "What is model deployment in ML production?",
        "knowledge": (
            "Model deployment makes a trained machine learning model "
            "available for real-world applications and users."
        )
    },

    {
        "topic": "scalability",
        "question": "Why is scalability important in ML production?",
        "knowledge": (
            "Production ML systems must handle increasing workloads, "
            "users, requests, and data efficiently."
        )
    },

    {
        "topic": "security",
        "question": "Why are security and compliance important in ML production?",
        "knowledge": (
            "Production systems must protect data, follow security "
            "requirements, and satisfy privacy and compliance requirements."
        )
    }
]


# =========================================================
# TEST QUERIES
# =========================================================

TEST_CASES = [

    # -------------------------
    # RELEVANT QUERIES
    # -------------------------

    {
        "query": "How are ML research and production different?",
        "expected_topic": "research_vs_production"
    },

    {
        "query": "What changes when an ML model moves from research to production?",
        "expected_topic": "research_vs_production"
    },

    {
        "query": "How does production machine learning handle data differently?",
        "expected_topic": "data_handling"
    },

    {
        "query": "What happens when a machine learning model is deployed?",
        "expected_topic": "model_deployment"
    },

    {
        "query": "Why does an ML production system need to scale?",
        "expected_topic": "scalability"
    },

    {
        "query": "Why does production ML need security and privacy?",
        "expected_topic": "security"
    },

    # -------------------------
    # IRRELEVANT QUERIES
    # -------------------------

    {
        "query": "What is the capital of France?",
        "expected_topic": None
    },

    {
        "query": "How does photosynthesis work in plants?",
        "expected_topic": None
    },

    {
        "query": "What is the boiling point of water?",
        "expected_topic": None
    },

    {
        "query": "How do planets orbit the Sun?",
        "expected_topic": None
    },

    {
        "query": "What is the history of the Roman Empire?",
        "expected_topic": None
    },

    {
        "query": "How does a bicycle work?",
        "expected_topic": None
    }
]


# =========================================================
# CREATE PLAYBOOK EMBEDDINGS
# =========================================================

print("\nCreating controlled Playbook embeddings...")

playbook_texts = []

for entry in PLAYBOOK:

    text = (
        entry["question"]
        + " "
        + entry["knowledge"]
        + " "
        + entry["topic"]
    )

    playbook_texts.append(text)


playbook_embeddings = model.encode(
    playbook_texts,
    normalize_embeddings=True
)

print("Embeddings created.")


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
# RUN EXPERIMENT
# =========================================================

print("\n========================================")
print("CONTROLLED PLAYBOOK RETRIEVAL BENCHMARK")
print("========================================")


for threshold in THRESHOLDS:

    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0

    print("\n========================================")
    print(f"THRESHOLD = {threshold}")
    print("========================================")

    for test in TEST_CASES:

        query = test["query"]
        expected_topic = test["expected_topic"]

        query_embedding = model.encode(
            [query],
            normalize_embeddings=True
        )[0]

        # Cosine similarity because embeddings are normalized
        scores = playbook_embeddings @ query_embedding

        best_index = scores.argmax()
        best_score = float(scores[best_index])

        best_entry = PLAYBOOK[best_index]

        # Decide whether retrieval happens
        retrieved = best_score >= threshold

        # Expected relevant / irrelevant
        expected_relevant = expected_topic is not None

        # Actual relevant / irrelevant
        actual_relevant = retrieved

        # Classification
        if expected_relevant and actual_relevant:

            if best_entry["topic"] == expected_topic:
                true_positive += 1
                result = "TP"
            else:
                false_positive += 1
                false_negative += 1
                result = "WRONG TOPIC"

        elif expected_relevant and not actual_relevant:

            false_negative += 1
            result = "FN"

        elif not expected_relevant and actual_relevant:

            false_positive += 1
            result = "FP"

        else:

            true_negative += 1
            result = "TN"

        print(
            f"{result:11} | "
            f"{best_score:.4f} | "
            f"{query}"
        )


    # =====================================================
    # METRICS
    # =====================================================

    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative

    if precision_denominator > 0:
        precision = (
            true_positive / precision_denominator
        )
    else:
        precision = 0

    if recall_denominator > 0:
        recall = (
            true_positive / recall_denominator
        )
    else:
        recall = 0

    if precision + recall > 0:
        f1 = (
            2 * precision * recall
            / (precision + recall)
        )
    else:
        f1 = 0


    accuracy = (
        true_positive + true_negative
    ) / len(TEST_CASES)


    print("\nMetrics:")

    print(
        f"Accuracy:  {accuracy * 100:.2f}%"
    )

    print(
        f"Precision: {precision * 100:.2f}%"
    )

    print(
        f"Recall:    {recall * 100:.2f}%"
    )

    print(
        f"F1 Score:  {f1 * 100:.2f}%"
    )

    print(
        f"TP={true_positive}, "
        f"FP={false_positive}, "
        f"TN={true_negative}, "
        f"FN={false_negative}"
    )