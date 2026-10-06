from google import genai
import os
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


def reflect_answer(query, answer, context):

    prompt = f"""
You are a strict evaluator for an AI learning assistant.

Evaluate whether the AI answer is fully supported by the provided study material.

Student question:
{query}

AI answer:
{answer}

Study material:
{context}

Rules:

1. Return PASS only if the answer is supported by the study material.
2. Return FAIL if the answer contains unsupported or invented information.
3. If the answer is partially supported, return FAIL.
4. Keep the response short.

Return exactly one of:

PASS

or

FAIL: <brief reason>
"""

    try:
        interaction = client.interactions.create(
            model="gemini-3.5-flash-lite",
            input=prompt
        )

        result = interaction.output_text.strip()

        if not result:
            return "REFLECTION_FAILED"

        return result

    except Exception as e:
        print("Reflector error:", e)
        return "REFLECTION_FAILED"