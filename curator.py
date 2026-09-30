from google import genai
import os
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


def curate_knowledge(query, answer, reflection):

    prompt = f"""
You are the curator of an AI tutor's persistent Playbook.

Student question:
{query}

Generated answer:
{answer}

Reflector evaluation:
{reflection}

Your task is to extract only the most useful reusable knowledge
from this interaction.

If the reflection is not PASS, return:

NO_UPDATE

If the reflection is PASS, create a short knowledge entry that
could help answer similar future questions.

Do not add information that is not present in the generated answer.

Return only the reusable knowledge.
"""

    try:

        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=prompt
        )

        return response.text.strip()

    except Exception as e:

        print("Curator error:", e)

        return "NO_UPDATE"