import json
import os
import shutil
import time

from rag import (
    load_pdf,
    chunk_text,
    create_vector_index
)

from playbook import search_playbook_top_k

from agent import agent


# =========================================================
# QUESTIONS
# =========================================================

QUESTIONS = [
    "What is the difference between ML research and ML production?",
    "What is the goal and focus of ML research?",
    "How is data handling different in ML research and production?",
    "What is model deployment in ML production?",
    "Why is scalability important in ML production?"
]


# =========================================================
# BACKUP FILES
# =========================================================

PLAYBOOK_FILE = "playbook.json"
PLAYBOOK_BACKUP = "full_eval_playbook_backup.json"

FAILURE_FILE = "failure_memory.json"
FAILURE_BACKUP = "full_eval_failure_backup.json"


# =========================================================
# BACKUP CURRENT STATE
# =========================================================

print("=" * 60)
print("FULL SYSTEM EVALUATION")
print("=" * 60)

print("\nCreating temporary backups...")


if os.path.exists(PLAYBOOK_FILE):

    shutil.copy2(
        PLAYBOOK_FILE,
        PLAYBOOK_BACKUP
    )


if os.path.exists(FAILURE_FILE):

    shutil.copy2(
        FAILURE_FILE,
        FAILURE_BACKUP
    )


# =========================================================
# LOAD PDF
# =========================================================

print("\nLoading PDF...")

text = load_pdf(
    "knowledge/DBMS.pdf"
)

chunks = chunk_text(text)

print(
    "Number of chunks:",
    len(chunks)
)


# =========================================================
# CREATE INDEX
# =========================================================

embedding_model, index = create_vector_index(
    chunks
)


# =========================================================
# EVALUATION
# =========================================================

successful_runs = 0
playbook_hits = 0

latencies = []


for number, query in enumerate(
    QUESTIONS,
    start=1
):

    print("\n")
    print("=" * 60)

    print(
        f"QUESTION {number}/{len(QUESTIONS)}"
    )

    print(
        query
    )

    print("=" * 60)


    # -----------------------------------------------------
    # Check Playbook before agent execution
    # -----------------------------------------------------

    playbook_results = search_playbook_top_k(
        query=query,
        model=embedding_model,
        k=1,
        threshold=0.60
    )


    if playbook_results:

        playbook_hits += 1

        print(
            "\nPlaybook memory available."
        )

    else:

        print(
            "\nNo Playbook memory available."
        )


    # -----------------------------------------------------
    # Run complete agent
    # -----------------------------------------------------

    start_time = time.perf_counter()


    try:

        answer = agent(
            query,
            embedding_model,
            index,
            chunks
        )

        successful_runs += 1

    except Exception as error:

        print(
            "\nAgent error:",
            error
        )

        answer = None


    elapsed = (
        time.perf_counter()
        - start_time
    )

    latencies.append(elapsed)


    print(
        f"\nExecution time: "
        f"{elapsed:.2f} seconds"
    )


# =========================================================
# METRICS
# =========================================================

total = len(QUESTIONS)

execution_success = (
    successful_runs / total
    if total
    else 0
)

playbook_hit_rate = (
    playbook_hits / total
    if total
    else 0
)

average_latency = (
    sum(latencies) / len(latencies)
    if latencies
    else 0
)


# =========================================================
# RESULTS
# =========================================================

print("\n")
print("=" * 60)
print("FULL SYSTEM RESULTS")
print("=" * 60)

print(
    f"Questions tested: {total}"
)

print(
    f"Successful agent executions: "
    f"{successful_runs}/{total}"
)

print(
    f"Execution success rate: "
    f"{execution_success * 100:.2f}%"
)

print(
    f"Playbook hits: "
    f"{playbook_hits}/{total}"
)

print(
    f"Playbook hit rate: "
    f"{playbook_hit_rate * 100:.2f}%"
)

print(
    f"Average latency: "
    f"{average_latency:.2f} seconds"
)


# =========================================================
# RESTORE PLAYBOOK
# =========================================================

print("\nRestoring evaluation state...")


if os.path.exists(PLAYBOOK_BACKUP):

    shutil.copy2(
        PLAYBOOK_BACKUP,
        PLAYBOOK_FILE
    )

    os.remove(
        PLAYBOOK_BACKUP
    )


# =========================================================
# RESTORE FAILURE MEMORY
# =========================================================

if os.path.exists(FAILURE_BACKUP):

    shutil.copy2(
        FAILURE_BACKUP,
        FAILURE_FILE
    )

    os.remove(
        FAILURE_BACKUP
    )

elif os.path.exists(FAILURE_FILE):

    # Remove failures created during evaluation
    os.remove(
        FAILURE_FILE
    )


print(
    "Evaluation state restored."
)

print(
    "\nFull system evaluation complete."
)