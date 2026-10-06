from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss

from google import genai
import os
from dotenv import load_dotenv
import time

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


# =========================================================
# LOAD PDF
# =========================================================

def load_pdf(path):

    reader = PdfReader(path)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# =========================================================
# CHUNK TEXT
# =========================================================

def chunk_text(text, chunk_size=500, overlap=100):

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(words[start:end])

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# =========================================================
# CREATE VECTOR INDEX
# =========================================================

def create_vector_index(chunks):

    print("\nCreating embeddings...")

    model = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings = model.encode(chunks)

    print("Embedding shape:", embeddings.shape)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    print("FAISS index created!")

    print("Number of vectors:", index.ntotal)

    return model, index


# =========================================================
# RETRIEVE CURRENT STUDY MATERIAL
# =========================================================

def retrieve_context(query, model, index, chunks, k=3):

    query_embedding = model.encode([query])

    distances, indices = index.search(
        query_embedding,
        k
    )

    threshold = 1.5

    if distances[0][0] > threshold:

        return None

    context = ""

    for i in range(k):

        chunk = chunks[indices[0][i]]

        context += chunk + "\n\n"

    return context


# =========================================================
# GENERATE ANSWER
# =========================================================

def generate_answer(
    query,
    context,
    feedback=None
):

    if context is None:

        return "I couldn't find this information in the provided notes."

    prompt = f"""
You are an AI tutor.

Answer the student's question using ONLY the provided study material
and verified Playbook knowledge.

If the answer is not present in the provided information, say:

"I couldn't find this information in the provided notes."

Do not invent information.

Study material and verified knowledge:

{context}

Student question:

{query}

Give a clear and simple explanation suitable for a college student.
"""

    # =====================================================
    # ADD REFLECTION FEEDBACK DURING REGENERATION
    # =====================================================

    if feedback:

        prompt += f"""

IMPORTANT:

A previous answer failed verification.

Reflector feedback:

{feedback}

Generate a new answer that fixes the problem identified by
the reflector.

Do not repeat unsupported claims.
"""


    max_retries = 3

    for attempt in range(max_retries):

        try:

            interaction = client.interactions.create(
                model="gemini-3.5-flash-lite",
                input=prompt
            )

            answer = interaction.output_text

            if not answer or not answer.strip():

                return "Gemini returned an empty answer."

            return answer.strip()

        except Exception as e:

            error_message = str(e)

            if (
                "429" in error_message
                or "RESOURCE_EXHAUSTED" in error_message
            ):

                return "Gemini API quota has been exhausted."

            if attempt < max_retries - 1:

                wait_time = 2 ** attempt

                print(
                    f"Gemini temporarily unavailable. "
                    f"Retrying in {wait_time} seconds..."
                )

                time.sleep(wait_time)

            else:

                print(
                    "Generator error:",
                    error_message
                )

                return "Gemini is currently unavailable."