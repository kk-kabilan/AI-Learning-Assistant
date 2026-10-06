import os
import shutil

from rag import (
    load_pdf,
    chunk_text,
    create_vector_index,
    retrieve_context,
    generate_answer
)

from playbook import search_playbook_top_k
from reflector import reflect_answer
from agent import agent


QUESTIONS = [
    {
        "question": "What is the difference between ML research and ML production?",
        "expected": "hit"
    },
    {
        "question": "What is the goal and focus of ML research?",
        "expected": "hit"
    },
    {
        "question": "How is data handling different in ML research and production?",
        "expected": "hit"
    },
    {
        "question": "What is model deployment in ML production?",
        "expected": "hit"
    },
    {
        "question": "Why is scalability important in ML production?",
        "expected": "hit"
    },
    {
        "question": "What is the capital of France?",
        "expected": "miss"
    },
    {
        "question": "How does photosynthesis work in plants?",
        "expected": "miss"
    },
    {
        "question": "What is the boiling point of water?",
        "expected": "miss"
    },
    {
        "question": "How do planets orbit the Sun?",
        "expected": "miss"
    },
    {
        "question": "What is the history of the Roman Empire?",
        "expected": "miss"
    },
    {
        "question": "How does a bicycle work?",
        "expected": "miss"
    }
]


PLAYBOOK_FILE = "playbook.json"
PLAYBOOK_BACKUP = "playbook_eval_backup.json"


print("Loading PDF...")

text = load_pdf("knowledge/DBMS.pdf")
chunks = chunk_text(text)

print("Number of chunks:", len(chunks))

embedding_model, index = create_vector_index(chunks)


# =========================================================
# PLAYBOOK EVALUATION
# =========================================================

def check_playbook(query):

    results = search_playbook_top_k(
        query=query,
        model=embedding_model,
        k=1,
        threshold=0.60
    )

    if results:
        return True, results[0]["score"]

    return False, None


def evaluate_playbook():

    print("\n")
    print("=" * 60)
    print("PLAYBOOK EVALUATION")
    print("=" * 60)

    tp = fp = tn = fn = 0

    for item in QUESTIONS:

        query = item["question"]
        expected = item["expected"]

        hit, score = check_playbook(query)

        predicted = "hit" if hit else "miss"

        if expected == "hit" and predicted == "hit":
            tp += 1

        elif expected == "miss" and predicted == "hit":
            fp += 1

        elif expected == "miss" and predicted == "miss":
            tn += 1

        elif expected == "hit" and predicted == "miss":
            fn += 1

        print(
            f"{predicted.upper():5} | "
            f"Expected: {expected.upper():5} | "
            f"{query}"
        )

        if score is not None:
            print(
                f"       similarity = {score:.4f}"
            )

    total = tp + fp + tn + fn

    accuracy = (
        (tp + tn) / total
        if total
        else 0
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp)
        else 0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn)
        else 0
    )

    f1 = (
        2 * precision * recall /
        (precision + recall)
        if (precision + recall)
        else 0
    )

    print("\nMetrics:")

    print(
        f"Accuracy: {accuracy * 100:.2f}%"
    )

    print(
        f"Precision: {precision * 100:.2f}%"
    )

    print(
        f"Recall: {recall * 100:.2f}%"
    )

    print(
        f"F1 Score: {f1 * 100:.2f}%"
    )

    print(
        f"TP={tp}, FP={fp}, TN={tn}, FN={fn}"
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# =========================================================
# RAG EVALUATION
# =========================================================

def evaluate_rag():

    print("\n")
    print("=" * 60)
    print("RAG RETRIEVAL EVALUATION")
    print("=" * 60)

    relevant_hits = 0
    relevant_total = 0

    rejected_correctly = 0
    irrelevant_total = 0

    for item in QUESTIONS:

        query = item["question"]
        expected = item["expected"]

        context = retrieve_context(
            query,
            embedding_model,
            index,
            chunks
        )

        found = context is not None

        if expected == "hit":

            relevant_total += 1

            if found:
                relevant_hits += 1

            print(
                f"{'HIT' if found else 'MISS':5} | "
                f"IN-DOMAIN | {query}"
            )

        else:

            irrelevant_total += 1

            if not found:
                rejected_correctly += 1

            print(
                f"{'MISS' if not found else 'HIT':5} | "
                f"OUT-OF-DOMAIN | {query}"
            )

    retrieval_rate = (
        relevant_hits / relevant_total
        if relevant_total
        else 0
    )

    rejection_rate = (
        rejected_correctly / irrelevant_total
        if irrelevant_total
        else 0
    )

    print("\nRAG Results:")

    print(
        f"In-domain retrieval: "
        f"{relevant_hits}/{relevant_total} "
        f"({retrieval_rate * 100:.2f}%)"
    )

    print(
        f"Out-of-domain rejection: "
        f"{rejected_correctly}/{irrelevant_total} "
        f"({rejection_rate * 100:.2f}%)"
    )

    return {
        "retrieval_rate": retrieval_rate,
        "rejection_rate": rejection_rate
    }


# =========================================================
# REFLECTION EVALUATION
# =========================================================

def evaluate_reflection():

    print("\n")
    print("=" * 60)
    print("REFLECTION EVALUATION")
    print("=" * 60)

    passed = 0
    tested = 0

    for item in QUESTIONS:

        query = item["question"]

        context = retrieve_context(
            query,
            embedding_model,
            index,
            chunks
        )

        if context is None:
            continue

        answer = generate_answer(
            query,
            context
        )

        reflection = reflect_answer(
            query,
            answer,
            context
        )

        tested += 1

        if reflection.startswith("PASS"):
            passed += 1

        print(
            f"{'PASS' if reflection.startswith('PASS') else 'FAIL':5} | "
            f"{query}"
        )

    rate = (
        passed / tested
        if tested
        else 0
    )

    print(
        f"\nReflection pass rate: "
        f"{rate * 100:.2f}%"
    )

    return rate


# =========================================================
# FULL AGENT EVALUATION
# =========================================================

def evaluate_full_agent():

    print("\n")
    print("=" * 60)
    print("FULL AGENT EVALUATION")
    print("=" * 60)

    successful = 0
    reflection_passes = 0
    tested = 0

    for item in QUESTIONS:

        query = item["question"]

        print("\n")
        print("-" * 60)
        print(
            f"Question: {query}"
        )
        print("-" * 60)

        try:

            result = agent(
                query,
                embedding_model,
                index,
                chunks,
                return_details=True
            )

            successful += 1
            tested += 1

            if (
                isinstance(result, dict)
                and result.get("reflection") == "PASS"
            ):
                reflection_passes += 1

            print(
                "\nFinal reflection:",
                result.get("reflection")
                if isinstance(result, dict)
                else "UNKNOWN"
            )

            if isinstance(result, dict):

                print(
                    "Playbook memories used:",
                    result.get("playbook_hits", 0)
                )

                print(
                    "Failure memories used:",
                    result.get("failure_hits", 0)
                )

        except Exception as e:

            tested += 1

            print(
                "Agent execution error:",
                e
            )

    execution_rate = (
        successful / tested
        if tested
        else 0
    )

    reflection_rate = (
        reflection_passes / tested
        if tested
        else 0
    )

    print("\n")
    print("=" * 60)
    print("FULL AGENT RESULTS")
    print("=" * 60)

    print(
        f"Questions tested: {tested}"
    )

    print(
        f"Successful executions: "
        f"{successful}/{tested}"
    )

    print(
        f"Execution success rate: "
        f"{execution_rate * 100:.2f}%"
    )

    print(
        f"Reflection passes: "
        f"{reflection_passes}/{tested}"
    )

    print(
        f"Reflection pass rate: "
        f"{reflection_rate * 100:.2f}%"
    )

    return {
        "execution_rate": execution_rate,
        "reflection_rate": reflection_rate
    }


# =========================================================
# CONTROLLED PLAYBOOK
# =========================================================

print("\n")
print("=" * 60)
print("CREATING CONTROLLED PLAYBOOK STATE")
print("=" * 60)

if os.path.exists(PLAYBOOK_FILE):

    shutil.copy2(
        PLAYBOOK_FILE,
        PLAYBOOK_BACKUP
    )

    print(
        "Original Playbook backed up."
    )

else:

    print(
        "No Playbook file found."
    )


# =========================================================
# FINAL ABLATION
# =========================================================

print("\n")
print("=" * 60)
print("FINAL CONTROLLED ABLATION EVALUATION")
print("=" * 60)


print("\nConfiguration 1: RAG")
rag_score = evaluate_rag()


print("\nConfiguration 2: RAG + Playbook")
playbook_score = evaluate_playbook()


print("\nConfiguration 3: RAG + Reflection")
reflection_score = evaluate_reflection()


print("\nConfiguration 4: Full Agent")
full_agent_score = evaluate_full_agent()


# =========================================================
# RESTORE ORIGINAL PLAYBOOK
# =========================================================

print("\n")
print("=" * 60)
print("RESTORING ORIGINAL PLAYBOOK")
print("=" * 60)

if os.path.exists(PLAYBOOK_BACKUP):

    shutil.copy2(
        PLAYBOOK_BACKUP,
        PLAYBOOK_FILE
    )

    os.remove(
        PLAYBOOK_BACKUP
    )

    print(
        "Original Playbook restored."
    )


# =========================================================
# FINAL RESULTS
# =========================================================

print("\n")
print("=" * 60)
print("FINAL CONTROLLED RESULTS")
print("=" * 60)

print(
    f"RAG in-domain retrieval: "
    f"{rag_score['retrieval_rate'] * 100:.2f}%"
)

print(
    f"RAG out-of-domain rejection: "
    f"{rag_score['rejection_rate'] * 100:.2f}%"
)

print(
    f"Playbook accuracy: "
    f"{playbook_score['accuracy'] * 100:.2f}%"
)

print(
    f"Playbook precision: "
    f"{playbook_score['precision'] * 100:.2f}%"
)

print(
    f"Playbook recall: "
    f"{playbook_score['recall'] * 100:.2f}%"
)

print(
    f"Playbook F1: "
    f"{playbook_score['f1'] * 100:.2f}%"
)

print(
    f"Reflection pass rate: "
    f"{reflection_score * 100:.2f}%"
)

print(
    f"Full agent execution success: "
    f"{full_agent_score['execution_rate'] * 100:.2f}%"
)

print(
    f"Full agent reflection pass rate: "
    f"{full_agent_score['reflection_rate'] * 100:.2f}%"
)

print("\nEvaluation complete.")