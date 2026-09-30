# 🤖 AI Learning Assistant

An agentic AI tutoring system that answers questions from curriculum-based study material using **RAG, semantic Playbook memory, reflection, and curation**.

This project is a lightweight student-scale implementation inspired by the agentic tutoring architecture described in *DeepEdu-v1*. It is **not a reproduction of the full research system**.

---

## 🚀 Features

- **PDF-based knowledge base** — uses course/study material as the source of knowledge.
- **RAG** — retrieves relevant chunks using Sentence Transformers and FAISS.
- **Agentic routing** — checks persistent Playbook memory before searching the study material.
- **Gemini generation** — generates answers grounded in retrieved context.
- **Reflection** — evaluates whether generated answers are supported by the retrieved material.
- **Curator** — extracts reusable knowledge from successful interactions.
- **Persistent Playbook** — stores verified knowledge in JSON for future interactions.
- **Semantic memory search** — uses embedding similarity rather than exact question matching.
- **Streamlit UI** — provides a simple browser-based interface.
- **Evaluation pipeline** — measures retrieval success, reflection pass rate, memory reuse, and response time.

---

## 🧠 Architecture

```text
                    Student
                       │
                       ▼
               ┌───────────────┐
               │  Streamlit UI │
               └───────┬───────┘
                       │
                       ▼
               ┌───────────────┐
               │     Agent     │
               └───────┬───────┘
                       │
              ┌────────┴────────┐
              ▼                 ▼
       ┌─────────────┐   ┌─────────────┐
       │   Playbook  │   │     RAG     │
       │   Memory    │   │ FAISS + PDF │
       └──────┬──────┘   └──────┬──────┘
              │                 │
              │                 ▼
              │          ┌─────────────┐
              │          │   Gemini    │
              │          │  Generator  │
              │          └──────┬──────┘
              │                 │
              │                 ▼
              │          ┌─────────────┐
              │          │  Reflector  │
              │          └──────┬──────┘
              │                 │
              │                 ▼
              └────────────┐
                           ▼
                    ┌─────────────┐
                    │   Curator   │
                    └──────┬──────┘
                           │
                           ▼
                     Playbook JSON



🔄 Agent Workflow

For each student question:

1.The Agent receives the question.
2.The Agent checks the semantic Playbook memory.
3.If a sufficiently similar verified entry exists, the stored knowledge is reused.
4.Otherwise, the question is sent to the RAG pipeline.
5.FAISS retrieves relevant study-material chunks.
6.Gemini generates a grounded answer.
7.The Reflector checks the generated answer against the retrieved context.
8.If reflection passes, the Curator extracts reusable knowledge.
9.The curated knowledge is stored in playbook.json.
10.Future semantically similar questions can reuse the stored knowledge.


## 🛠️ Tech Stack
Python
Google Gemini API
Streamlit
Sentence Transformers
FAISS
PyPDF
NumPy
JSON
python-dotenv





📁 Project Structure
AI_LEARNING_ASSISTENT/
│
├── .streamlit/
│   └── config.toml
│
├── knowledge/
│   └── DBMS.pdf
│
├── app.py
├── agent.py
├── rag.py
├── reflector.py
├── curator.py
├── playbook.py
├── playbook.json
├── evaluation.py
├── evaluation_results.txt
├── requirements.txt
├── README.md
└── .gitignore





📊 Evaluation

A fixed set of 5 curriculum-grounded questions from the study material was used for the evaluation.

Basic RAG
Metric	Result
Questions tested	5
Relevant context retrieved	5/5
Retrieval success rate	100%
Average response time	1.65 s
RAG + Reflection
Metric	Result
Questions tested	5
Reflection passed	5/5
Reflection pass rate	100%
Average response time	2.75 s

Reflection increased the measured average response time because it adds an additional LLM evaluation step.

Playbook Memory
Metric	Result
Questions tested	5
Playbook hits	2/5
Playbook hit rate	40%

The Playbook result measures semantic memory reuse, not answer accuracy. The evaluation used the existing Playbook accumulated during development rather than an empty Playbook.

Evaluation Limitations
The evaluation dataset contains only 5 questions.
Results are specific to the study material and configuration used during testing.
Response time depends on the local environment and Gemini API latency.
The Playbook hit rate measures memory reuse rather than overall tutoring accuracy.




⚙️ Installation
1. Clone the repository
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd AI_LEARNING_ASSISTENT
2. Create a virtual environment

Windows:

python -m venv venv
venv\Scripts\activate
3. Install dependencies
pip install -r requirements.txt
4. Configure the Gemini API key

Create a .env file in the project root:

GEMINI_API_KEY=your_api_key_here

Never commit .env or your API key to GitHub.

5. Add the study material

The current project uses:

knowledge/DBMS.pdf
▶️ Run the Application

Start the Streamlit application:

streamlit run app.py

The application will open in your browser.

🧪 Run Evaluation

Run the evaluation script:

python evaluation.py

The evaluation measures:

RAG retrieval success
Response time
Reflection pass rate
Playbook memory hits

Results are also stored in:

evaluation_results.txt
💾 Persistent Playbook

The Playbook is stored in:

playbook.json

A successful interaction can be curated into reusable knowledge.

Example:

{
    "question": "What is the difference between ML research and ML production?",
    "answer": "Reusable curated knowledge...",
    "context": "Relevant study material..."
}

The system uses embedding similarity to determine whether previously stored knowledge is relevant to a new question.

This allows semantically similar questions to reuse previously curated knowledge instead of always performing a new retrieval and generation process.

🔐 Security

The Gemini API key is loaded through an environment variable:

GEMINI_API_KEY

The .env file should remain local and is excluded through .gitignore.

Never place API keys directly inside Python source files or commit them to GitHub.

📌 Project Scope

This is a student-scale implementation of an agentic AI learning assistant.

The project focuses on:

Retrieval-Augmented Generation
Agentic routing
Answer reflection
Knowledge curation
Persistent contextual memory

The implementation is inspired by the agentic tutoring concepts described in DeepEdu-v1.

It does not attempt to reproduce the complete DeepEdu-v1 research infrastructure, long-context inference optimizations, training setup, or benchmark environment.

🔮 Future Improvements
Precompute and persist Playbook embeddings instead of recomputing them for every query.
Add a larger automated evaluation dataset.
Add answer-quality and groundedness metrics.
Support multiple course PDFs.
Add source citations to retrieved answers.
Add user-specific learning progress.
Add richer agent planning and tool use.
Replace JSON storage with a vector database for larger Playbooks.
Add experiment tracking with tools such as Weights & Biases.