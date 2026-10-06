from reflector import reflect_answer
from rag import generate_answer


query = "What is the difference between ML research and ML production?"

context = """
ML research focuses on advancing theory, discovering algorithms,
and improving model performance.

ML production focuses on deployment, scalability, reliability,
real-world usage, and operational efficiency.
"""



bad_answer = """
ML research and ML production are exactly the same.
Both have the same goals, the same data requirements,
and the same deployment requirements.
"""


print("========================================")
print("FAILURE RECOVERY TEST")
print("========================================")


# =========================================================
# STEP 1: REFLECT BAD ANSWER
# =========================================================

print("\n--- Reflector ---")

reflection = reflect_answer(
    query,
    bad_answer,
    context
)

print("Reflection:", reflection)


# =========================================================
# STEP 2: REGENERATE IF FAILED
# =========================================================

if reflection.startswith("FAIL"):

    print("\nBad answer detected.")
    print("Regenerating using reflector feedback...")

    new_answer = generate_answer(
        query,
        context,
        feedback=reflection
    )

    print("\n--- Regenerated Answer ---")
    print(new_answer)


    # =====================================================
    # STEP 3: REFLECT AGAIN
    # =====================================================

    print("\n--- Second Reflection ---")

    second_reflection = reflect_answer(
        query,
        new_answer,
        context
    )

    print(
        "Second reflection:",
        second_reflection
    )


    if second_reflection.startswith("PASS"):

        print("\nSUCCESS:")
        print("The agent successfully recovered from a failed answer.")

    else:

        print("\nThe regenerated answer still failed verification.")

else:

    print(
        "\nUnexpected result: "
        "the deliberately incorrect answer passed."
    )