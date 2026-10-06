import json
import os


FAILURE_MEMORY_FILE = "failure_memory.json"


# =========================================================
# LOAD FAILURE MEMORY
# =========================================================

def load_failure_memory():

    if not os.path.exists(FAILURE_MEMORY_FILE):
        return []

    with open(
        FAILURE_MEMORY_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# =========================================================
# SAVE FAILURE MEMORY
# =========================================================

def save_failure_memory(memory):

    with open(
        FAILURE_MEMORY_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            memory,
            file,
            indent=4,
            ensure_ascii=False
        )


# =========================================================
# ADD FAILURE
# =========================================================

def add_failure(
    query,
    answer,
    failure_reason
):

    memory = load_failure_memory()

    entry = {
        "id": f"failure_{len(memory) + 1:04d}",
        "query": query,
        "failed_answer": answer,
        "failure_reason": failure_reason,
        "resolved": False
    }

    memory.append(entry)

    save_failure_memory(memory)

    print("Failure memory updated.")


# =========================================================
# GET RELEVANT FAILURES
# =========================================================

def get_relevant_failures(
    query,
    limit=3
):

    memory = load_failure_memory()

    if not memory:
        return []

    # Simple keyword-based filtering for now.
    # We will replace this with semantic retrieval later.
    query_words = set(
        query.lower().split()
    )

    scored = []

    for entry in memory:

        stored_words = set(
            entry["query"].lower().split()
        )

        overlap = len(
            query_words & stored_words
        )

        scored.append(
            {
                "entry": entry,
                "score": overlap
            }
        )

    scored.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return [
        item["entry"]
        for item in scored[:limit]
        if item["score"] > 0
    ]