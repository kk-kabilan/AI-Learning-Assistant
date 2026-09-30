from google import genai
import os
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


def reflect_answer(query, context, answer):

    prompt = f"""
You are a response evaluator for an AI tutor.

Your job is to check whether the generated answer is supported
by the provided study material.

Study material:
{context}

Student question:
{query}

Generated answer:
{answer}

Check the answer for:

1. Factual errors
2. Information not supported by the study material
3. Missing important information
4. Whether the answer actually answers the student's question

Return exactly one of these:

PASS

or

FAIL: <brief explanation of the problem>

Do not use outside knowledge.
"""

    try:

        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=prompt
        )

        return response.text.strip()

    except Exception as e:

        print("Reflection error:", e)

        return "REFLECTION_FAILED"