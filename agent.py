from rag import (
    load_pdf,
    chunk_text,
    create_vector_index,
    retrieve_context,
    generate_answer
)

from reflector import reflect_answer

from playbook import (
    search_playbook,
    add_to_playbook
)

from curator import curate_knowledge


# --------------------------------------------------
# AI Tutor Agent
# --------------------------------------------------

def agent(
    query,
    model,
    index,
    chunks
):

    print("\n--- Agent ---")

    print(
        "Analyzing question..."
    )


    # =================================================
    # STEP 1: Check Playbook
    # =================================================

    previous_entry = search_playbook(
        query,
        model
    )


    if previous_entry is not None:

        print(
            "Agent decision: Use Playbook memory."
        )

        return previous_entry["answer"]


    # =================================================
    # STEP 2: Search Study Material
    # =================================================

    print(
        "Agent decision: Search study material."
    )

    context = retrieve_context(
        query,
        model,
        index,
        chunks
    )


    if context is None:

        print(
            "No relevant study material found."
        )

        return (
            "I couldn't find this information "
            "in the provided notes."
        )


    # =================================================
    # STEP 3: Generate Answer
    # =================================================

    print(
        "\n--- Generator ---"
    )

    answer = generate_answer(
        query,
        context
    )


    # =================================================
    # STEP 4: Reflect
    # =================================================

    print(
        "\n--- Reflector ---"
    )

    reflection = reflect_answer(
        query,
        context,
        answer
    )

    print(
        "Reflection:",
        reflection
    )


    # =================================================
    # STEP 5: Curator
    # =================================================

    if reflection.startswith("PASS"):

        print(
            "\n--- Curator ---"
        )

        curated_knowledge = curate_knowledge(
            query,
            answer,
            reflection
        )

        print(
            "Curated knowledge:"
        )

        print(
            curated_knowledge
        )


        # ---------------------------------------------
        # Save to Playbook
        # ---------------------------------------------

        if curated_knowledge != "NO_UPDATE":

            add_to_playbook(
                query,
                curated_knowledge,
                context
            )

        else:

            print(
                "Curator decided not to update Playbook."
            )


    else:

        print(
            "Answer was not sent to Curator."
        )


    # =================================================
    # STEP 6: Return Answer
    # =================================================

    return answer


# --------------------------------------------------
# Terminal version
# --------------------------------------------------

if __name__ == "__main__":

    print(
        "Starting AI Tutor Agent..."
    )


    # ---------------------------------------------
    # Load PDF
    # ---------------------------------------------

    text = load_pdf(
        "knowledge/DBMS.pdf"
    )

    print(
        "PDF loaded."
    )


    # ---------------------------------------------
    # Create chunks
    # ---------------------------------------------

    chunks = chunk_text(
        text
    )

    print(
        "Chunks created:",
        len(chunks)
    )


    # ---------------------------------------------
    # Create vector database
    # ---------------------------------------------

    model, index = create_vector_index(
        chunks
    )


    # ---------------------------------------------
    # Get question
    # ---------------------------------------------

    query = input(
        "\nAsk your question: "
    )


    # ---------------------------------------------
    # Run Agent
    # ---------------------------------------------

    answer = agent(
        query,
        model,
        index,
        chunks
    )


    # ---------------------------------------------
    # Display answer
    # ---------------------------------------------

    print(
        "\n--- AI Tutor Answer ---\n"
    )

    print(
        answer
    )