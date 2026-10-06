from rag import (
    load_pdf,
    chunk_text,
    create_vector_index,
    retrieve_context,
    generate_answer
)

from playbook import (
    add_to_playbook,
    search_playbook_top_k
)

from failure_memory import (
    add_failure,
    get_relevant_failures
)

from reflector import reflect_answer
from curator import curate_knowledge


# =========================================================
# AGENT
# =========================================================

def agent(
    query,
    embedding_model,
    index,
    chunks,
    return_details=False
):

    print("\n--- Agent ---")
    print("Analyzing question...")


    # =====================================================
    # 1. SEARCH PLAYBOOK
    # =====================================================

    playbook_results = search_playbook_top_k(
        query=query,
        model=embedding_model,
        k=3,
        threshold=0.60
    )


    print("\n--- Playbook Retrieval ---")

    if playbook_results:

        print(
            f"Semantic Playbook memories found: "
            f"{len(playbook_results)}"
        )

        for i, result in enumerate(
            playbook_results,
            start=1
        ):

            entry = result["entry"]

            print(
                f"Memory {i}: "
                f"{result['score']:.4f} - "
                f"{entry.get('question', '')}"
            )

    else:

        print("No relevant Playbook memory found.")


    # =====================================================
    # 2. SEARCH FAILURE MEMORY
    # =====================================================

    failure_results = get_relevant_failures(
        query,
        limit=3
    )


    print("\n--- Failure Memory ---")

    if failure_results:

        print(
            f"Relevant previous failures: "
            f"{len(failure_results)}"
        )

        for i, failure in enumerate(
            failure_results,
            start=1
        ):

            print(
                f"Failure {i}: "
                f"{failure['failure_reason']}"
            )

    else:

        print("No relevant previous failures found.")


    # =====================================================
    # 3. RETRIEVE CURRENT STUDY MATERIAL
    # =====================================================

    print(
        "\n--- Current Study Material Retrieval ---"
    )

    rag_context = retrieve_context(
        query,
        embedding_model,
        index,
        chunks
    )

    if rag_context:

        print(
            "Relevant study material found."
        )

    else:

        print(
            "No relevant study material found."
        )


    # =====================================================
    # 4. BUILD COMBINED CONTEXT
    # =====================================================

    combined_parts = []


    # -----------------------------------------------------
    # Playbook
    # -----------------------------------------------------

    if playbook_results:

        playbook_context = "\n\n".join(
            [
                (
                    f"Playbook knowledge:\n"
                    f"{result['entry'].get('knowledge', result['entry'].get('answer', ''))}"
                )
                for result in playbook_results
            ]
        )

        combined_parts.append(
            "=== VERIFIED PLAYBOOK KNOWLEDGE ===\n"
            + playbook_context
        )


    # -----------------------------------------------------
    # Failure Memory
    # -----------------------------------------------------

    if failure_results:

        failure_context = "\n\n".join(
            [
                (
                    f"Previous failure:\n"
                    f"{failure['failure_reason']}"
                )
                for failure in failure_results
            ]
        )

        combined_parts.append(
            "=== PREVIOUS FAILURE WARNINGS ===\n"
            + failure_context
        )


    # -----------------------------------------------------
    # Current curriculum
    # -----------------------------------------------------

    if rag_context:

        combined_parts.append(
            "=== CURRENT STUDY MATERIAL ===\n"
            + rag_context
        )


    if not combined_parts:
        answer = (
            "I couldn't find this information "
            "in the provided notes."
        )

        if return_details:
            return {
                "answer": answer,
                "reflection": "NO_CONTEXT",
                "playbook_hits": len(playbook_results),
                "failure_hits": len(failure_results)
            }

        return answer


    combined_context = "\n\n".join(
        combined_parts
    )


    # =====================================================
    # 5. GENERATE ANSWER
    # =====================================================

    print("\n--- Generator ---")

    print(
        "Generating answer using "
        "Playbook + Failure Memory + study material..."
    )

    answer = generate_answer(
        query,
        combined_context
    )


    # =====================================================
    # 6. REFLECTION + RECOVERY LOOP
    # =====================================================

    max_reflections = 2

    reflection = "REFLECTION_FAILED"


    for attempt in range(
        max_reflections
    ):

        print(
            f"\n--- Reflector "
            f"(attempt {attempt + 1}) ---"
        )


        reflection = reflect_answer(
            query,
            answer,
            combined_context
        )


        print(
            "Reflection:",
            reflection
        )


        # =================================================
        # PASS
        # =================================================

        if reflection.startswith("PASS"):

            print(
                "Answer passed verification."
            )

            break


        # =================================================
        # FAIL
        # =================================================

        if reflection.startswith("FAIL"):

            print(
                "Answer failed verification."
            )


            # ---------------------------------------------
            # SAVE FAILURE
            # ---------------------------------------------

            failure_reason = reflection


            add_failure(
                query,
                answer,
                failure_reason
            )


            # ---------------------------------------------
            # REGENERATE
            # ---------------------------------------------

            if attempt < max_reflections - 1:

                print(
                    "Regenerating answer "
                    "using reflector feedback..."
                )


                answer = generate_answer(
                    query,
                    combined_context,
                    feedback=reflection
                )

            else:

                print(
                    "Maximum regeneration attempts reached."
                )


        else:

            print(
                "Reflection failed."
            )

            break


    # =====================================================
    # 7. CURATE ONLY VERIFIED ANSWERS
    # =====================================================

    if reflection.startswith("PASS"):

        print("\n--- Curator ---")


        curated_knowledge = curate_knowledge(
            query,
            answer,
            reflection,
            combined_context
        )


        print(
            "Curated knowledge:"
        )

        print(
            curated_knowledge
        )


        if (
            curated_knowledge
            and curated_knowledge != "NO_UPDATE"
        ):

            add_to_playbook(
                query,
                curated_knowledge,
                combined_context
            )

        else:

            print(
                "No Playbook update."
            )


    else:

        print(
            "\nAnswer was not verified."
        )

        print(
            "Playbook was NOT updated."
        )


    # =====================================================
    # 8. RETURN ANSWER
    # =====================================================

    if return_details:

        return {
            "answer": answer,
            "reflection": reflection,
            "playbook_hits": len(playbook_results),
            "failure_hits": len(failure_results)
        }

    return answer


# =========================================================
# MAIN PROGRAM
# =========================================================

if __name__ == "__main__":

    print("Loading PDF...")


    text = load_pdf(
        "knowledge/DBMS.pdf"
    )


    print(
        "PDF loaded successfully."
    )


    chunks = chunk_text(text)


    print(
        "Number of chunks:",
        len(chunks)
    )


    embedding_model, index = (
        create_vector_index(chunks)
    )


    query = input(
        "\nAsk your question: "
    )


    answer = agent(
        query,
        embedding_model,
        index,
        chunks
    )


    print(
        "\n--- AI Tutor Answer ---\n"
    )


    print(answer)