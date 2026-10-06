from failure_memory import (
    add_failure,
    load_failure_memory,
    get_relevant_failures
)


print("========================================")
print("FAILURE MEMORY TEST")
print("========================================")


query = "What is the difference between ML research and ML production?"

answer = """
ML research and ML production are exactly the same.
"""

failure_reason = (
    "The answer incorrectly claims that ML research "
    "and ML production are the same."
)


# =========================================================
# STORE FAILURE
# =========================================================

add_failure(
    query,
    answer,
    failure_reason
)


# =========================================================
# DISPLAY MEMORY
# =========================================================

memory = load_failure_memory()

print("\nStored failures:")

for entry in memory:

    print(
        f"{entry['id']} - "
        f"{entry['query']}"
    )

    print(
        f"Reason: {entry['failure_reason']}"
    )


# =========================================================
# RETRIEVE FAILURE
# =========================================================

print("\nRelevant failures:")

results = get_relevant_failures(
    "How does ML research differ from production?"
)

for result in results:

    print(
        result["failure_reason"]
    )