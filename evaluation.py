from playbook import search_playbook
from rag import (
    load_pdf,
    chunk_text,
    create_vector_index
)


QUESTIONS = [
    "What is the difference between ML research and ML production?",

    "What is the goal and focus of ML research?",

    "How is data handling different in ML research and production?",

    "What is model deployment in ML production?",

    "Why is scalability important in ML production?"
]


def evaluate_playbook(
    questions,
    model
):

    results = []

    for question in questions:

        print("\n--------------------------------")
        print("Question:", question)

        entry = search_playbook(
            question,
            model
        )

        if entry is not None:

            print(
                "Playbook: HIT"
            )

            results.append({
                "question": question,
                "hit": True
            })

        else:

            print(
                "Playbook: MISS"
            )

            results.append({
                "question": question,
                "hit": False
            })


    return results


def print_results(results):

    total = len(results)

    hits = sum(
        1
        for result in results
        if result["hit"]
    )

    hit_rate = (
        hits / total
    ) * 100

    print("\n================================")
    print("PLAYBOOK EVALUATION")
    print("================================")

    print(
        f"Questions tested: {total}"
    )

    print(
        f"Playbook hits: {hits}/{total}"
    )

    print(
        f"Playbook hit rate: {hit_rate:.1f}%"
    )


if __name__ == "__main__":

    print(
        "Loading study material..."
    )

    text = load_pdf(
        "knowledge/DBMS.pdf"
    )

    chunks = chunk_text(
        text
    )

    print(
        "Creating vector model..."
    )

    model, index = create_vector_index(
        chunks
    )

    print(
        "\nStarting Playbook evaluation..."
    )

    results = evaluate_playbook(
        QUESTIONS,
        model
    )

    print_results(
        results
    )