from google import genai
import os
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


def curate_knowledge(query, answer, reflection, context=None):

    # Only curate verified answers
    if not reflection.startswith("PASS"):
        return "NO_UPDATE"

    prompt = f"""
You are a knowledge curator for an AI learning assistant.

The AI tutor has answered a student's question and a reflector has
verified that the answer is acceptable.

Student question:
{query}

Verified answer:
{answer}

"""

    # Include study material when it is available
    if context:
        prompt += f"""
Study material:
{context}

Use the study material as the source of truth.
"""

    prompt += """
Create a short, reusable knowledge entry that could help answer
similar future questions.

Requirements:

1. Keep only important reusable knowledge.
2. Do not invent information.
3. Do not include conversational phrases.
4. Make the knowledge concise and clear.
5. Preserve important technical details.

Return only the reusable knowledge.
"""

    try:
        interaction = client.interactions.create(
            model="gemini-3.5-flash-lite",
            input=prompt
        )

        result = interaction.output_text.strip()

        if not result:
            return "NO_UPDATE"

        return result

    except Exception as e:
        print("Curator error:", e)
        return "NO_UPDATE"