# 🤖 AI Learning Assistant

An agentic AI tutoring system that answers questions from curriculum-based study material using **Retrieval-Augmented Generation (RAG), persistent semantic memory, reflection, curation, and failure memory**.

This project is a **student-scale research implementation inspired by the self-improving agentic layer of DeepEdu-v1**. It is **not a reproduction of the complete DeepEdu-v1 research system**.

---

## 🎯 Research Question

> **Does persistent verified semantic external memory improve a curriculum-grounded RAG tutor?**

The project investigates whether an AI tutor can reuse knowledge from previous verified interactions without modifying the underlying LLM weights.

Instead of changing the model itself, the system maintains an external **Playbook** that evolves through verified interactions.

---

## 🧠 Core Idea

A conventional LLM follows:

```text
Question → LLM → Answer

A basic RAG system follows:
Question
    ↓
Retrieve documents
    ↓
LLM
    ↓
Answer

This project extends the architecture with persistent memory and feedback:
                         Student Question
                                │
                ┌───────────────┼───────────────┐
                ↓               ↓               ↓
        Curriculum RAG     Semantic         Failure
                           Playbook          Memory
                │               │               │
                └───────────────┼───────────────┘
                                ↓
                           Generator
                                ↓
                           Reflector
                          /         \
                       FAIL         PASS
                        ↓             ↓
                Failure Memory     Curator
                        │             │
                        │             ↓
                        │          Playbook
                        │             │
                        ↓             │
                   Regenerate ←───────┘

The underlying LLM weights remain fixed. The system improves through external contextual memory.
🚀 Features
- PDF-based knowledge base — uses curriculum/study material as the source of knowledge.
- Retrieval-Augmented Generation — retrieves relevant study-material chunks using Sentence Transformers and FAISS.
- Semantic Playbook memory — retrieves previous knowledge using embedding similarity.
- Persistent external memory — verified knowledge can persist across interactions.
- Generator — generates answers using retrieved curriculum evidence and relevant memory.
- Reflector — checks whether generated answers are supported by the available evidence.
- Curator — converts successful interactions into reusable knowledge.
- Failure Memory — stores failed interactions and their failure reasons.
- Failure recovery — failed answers can be regenerated using reflector feedback.
- Out-of-domain rejection — the system can abstain when relevant curriculum context is unavailable.
- Streamlit interface — provides a browser-based interface.
- Research evaluation pipeline — evaluates retrieval, semantic memory, reflection, and full-agent execution.
🔄 Agent Workflow
For each student question:
1. Question
The student submits a question through the Streamlit interface.
2. Semantic Playbook Retrieval
The system converts the query into an embedding and searches the persistent Playbook using semantic similarity.
Relevant memories can be retrieved even when the wording of the new question differs from the stored question.
3. Curriculum Retrieval
The RAG pipeline searches the curriculum material:
PDF
 ↓
Text Extraction
 ↓
Chunking
 ↓
Sentence Transformer Embeddings
 ↓
FAISS Vector Search
 ↓
Relevant Study Material

4. Failure Memory Retrieval
Relevant previous failure patterns can also be retrieved and provided as warnings to the Generator.
5. Generation
Gemini generates an answer using the available:
- curriculum evidence
- verified Playbook knowledge
- previous failure warnings
6. Reflection
The Reflector checks whether the generated answer is supported by the retrieved evidence.
7. Failure Path
If the answer fails reflection:
FAIL
 ↓
Store Failure
 ↓
Use Reflection Feedback
 ↓
Regenerate Answer
 ↓
Reflect Again

8. Successful Path
If the answer passes reflection:
PASS
 ↓
Curator
 ↓
Reusable Knowledge
 ↓
Playbook

The curated knowledge can then be retrieved during future semantically related interactions.
🧠 Persistent Verified Semantic Memory
The Playbook combines three important properties.
Persistent
The knowledge is stored outside a single interaction and can be reused later.
Verified
The system attempts to update the Playbook only after the generated answer passes reflection.
Semantic
The system uses embedding similarity to retrieve memories based on meaning rather than exact wording.
Semantic Similarity
The system represents both the query and stored memory as embedding vectors.
The similarity is calculated using cosine similarity:
\[
\mathrm{Similarity}(q,m)
=
\frac{q \cdot m}
{\lVert q\rVert \lVert m\rVert}
\]
where:
- $q$ = query embedding
- $m$ = memory embedding
- $q \cdot m$ = dot product
- $\lVert q\rVert$ = magnitude of the query vector
- $\lVert m\rVert$ = magnitude of the memory vector
The implementation normalizes the embeddings before calculating their dot product:
query_embedding = model.encode(    [query],    normalize_embeddings=True)[0]stored_embedding = model.encode(    [search_text],    normalize_embeddings=True)[0]score = float(query_embedding @ stored_embedding)


Because the vectors are normalized, the dot product is equivalent to cosine similarity.
A higher score indicates greater semantic similarity between the query and stored memory.
🧩 Generator–Reflector–Curator
The agent uses three main components.
Generator
Produces the answer using the retrieved curriculum evidence, Playbook knowledge, and failure warnings.
Reflector
Evaluates whether the generated answer is supported by the available study material.
Curator
Processes a successfully verified answer and converts it into reusable Playbook knowledge.
This creates the following loop:
Generate
   ↓
Reflect
   ↓
 ┌───────────────┐
 │               │
FAIL           PASS
 │               │
 ↓               ↓
Regenerate     Curate
 │               │
 └───────→ Playbook

🧠 Failure Memory
The system also maintains a separate Failure Memory.
Instead of storing failed answers as trusted knowledge, the system records:
Query
Failed Answer
Failure Reason
Resolved Status

This prevents unsuccessful interactions from automatically becoming Playbook knowledge.
The current Failure Memory retrieval uses keyword overlap.
Future versions can replace this with semantic embedding-based retrieval.
🛠️ Tech Stack
Component	Technology
Programming Language	Python
LLM	Google Gemini
User Interface	Streamlit
Embeddings	Sentence Transformers
Vector Search	FAISS
PDF Processing	PyPDF
Persistent Memory	JSON
Environment Variables	python-dotenv


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
├── failure_memory.py
│
├── ablation_evaluation.py
├── playbook_evaluation.py
├── full_system_evaluation.py
├── retrieval_benchmark.py
├── threshold_experiment.py
├── failure_memory_test.py
├── failure_recovery_test.py
│
├── research_results.txt
├── requirements.txt
├── README.md
└── .gitignore

Runtime-generated files such as:
playbook.json
failure_memory.json
evaluation_results.txt

are intentionally excluded from version control.
📊 Evaluation
The final controlled evaluation used a fixed set of 11 questions:
- 5 in-domain curriculum questions
- 6 out-of-domain questions
The evaluation tested curriculum retrieval, out-of-domain rejection, semantic Playbook retrieval, reflection, and complete agent execution.
Important: The evaluation dataset is small and project-specific. These results should not be interpreted as universal performance guarantees.

Final Controlled Results
Metric	Result
In-domain RAG retrieval	5/5 (100%)
Out-of-domain rejection	6/6 (100%)
Playbook precision	100%
Playbook recall	100%
Playbook F1	100%
In-domain reflection pass rate	5/5 (100%)
Full-agent execution success	11/11 (100%)


The six out-of-domain questions correctly returned:
NO_CONTEXT

rather than proceeding with unsupported curriculum answers.
The full-agent reflection count was 5/11 because the six out-of-domain questions correctly abstained and therefore did not enter the normal reflection path.
🔬 Research Evaluation
The project contains several controlled experiments.
RAG Retrieval
Tests whether relevant curriculum material can be retrieved for in-domain questions and rejected for out-of-domain questions.
Playbook Retrieval
Evaluates semantic memory retrieval using:
- Precision
- Recall
- F1 score
Reflection
Tests whether generated answers pass curriculum-grounded verification.
Failure Recovery
Tests whether the Reflector can detect an unsuccessful answer and trigger regeneration using reflection feedback.
Full-Agent Evaluation
Tests the complete:
Retrieval
   ↓
Generation
   ↓
Reflection
   ↓
Curation / Failure
   ↓
Memory Update

workflow.
🔬 Research Positioning
The central research question is:
Does persistent verified semantic external memory improve a curriculum-grounded RAG tutor?

The project investigates a curriculum-grounded RAG system extended with:
- persistent semantic Playbook memory
- reflection
- curation
- Failure Memory
The underlying LLM weights remain unchanged.
The learning mechanism is therefore external:
LLM Parameters
      │
      │ remain fixed
      ↓

External Context
      │
      ├── Playbook
      └── Failure Memory
              │
              ↓
       evolves over time

This makes the memory mechanism explicit and experimentally inspectable.
📚 Relation to DeepEdu-v1
This project is inspired by the self-improving agentic layer described in DeepEdu-v1.
The implementation focuses on:
- persistent Playbook memory
- semantic memory retrieval
- Generator–Reflector–Curator behavior
- Failure Memory
- curriculum-grounded RAG
It does not reproduce the complete DeepEdu-v1 system.
In particular, this project does not implement Similarity Chunk Rolling (SCR) or the paper's long-context sparse-attention inference infrastructure.
Similarity Chunk Rolling (SCR)
SCR is a long-context inference optimization introduced in DeepEdu-v1.
It groups consecutive sub-chunks with high similarity into clusters and performs expensive selection operations at the cluster level rather than independently for every sub-chunk.
This project does not implement SCR.
The focus here is the self-improving external-memory agentic layer.
⚠️ Limitations
1. Small Evaluation Dataset
The final controlled evaluation contains only 11 questions.
A larger evaluation dataset is required to make stronger claims.
2. Keyword-Based Failure Memory
Failure Memory currently uses keyword overlap rather than semantic retrieval.
3. Playbook Duplication
The current Playbook can accumulate semantically similar entries.
A future version should support semantic deduplication and memory merging.
4. Limited Answer-Quality Evaluation
The current evaluation focuses strongly on:
- retrieval
- memory retrieval
- reflection
- execution behavior
A larger study should also evaluate answer correctness and groundedness using a stronger dataset.
5. Single Curriculum
The current implementation uses a single study-material source.
Testing across multiple subjects and curricula is needed for broader evaluation.
🔮 Future Improvements
- Replace keyword-based Failure Memory retrieval with semantic retrieval.
- Add semantic Playbook deduplication and memory merging.
- Precompute and persist Playbook embeddings.
- Build a larger evaluation dataset.
- Add answer correctness and groundedness metrics.
- Support multiple curriculum documents.
- Add source/chunk citations to answers.
- Add user-specific learning progress.
- Add richer agent planning and tool use.
- Replace JSON storage with a scalable vector database.
- Add experiment tracking with Weights & Biases.
- Investigate lightweight domain-specific language models as alternative generators.
⚙️ Installation
1. Clone the repository
git clone https://github.com/kk-kabilan/AI-Learning-Assistant.git
cd AI-Learning-Assistant

2. Create a virtual environment
Windows
python -m venv venv
venv\Scripts\activate

3. Install dependencies
pip install -r requirements.txt

4. Configure the Gemini API key
Create a .env file in the project root:
GEMINI_API_KEY=your_api_key_here

Never commit your .env file or API key to GitHub.
5. Add study material
The current project uses:
knowledge/DBMS.pdf

▶️ Run the Application
Start the Streamlit application:
streamlit run app.py

The application will open in your browser.
🧪 Run the Experiments
Run the main controlled evaluation:
python ablation_evaluation.py

Run the full-agent evaluation:
python full_system_evaluation.py

Run Playbook evaluation:
python playbook_evaluation.py

Run retrieval benchmarking:
python retrieval_benchmark.py

Run the threshold experiment:
python threshold_experiment.py

Run Failure Memory tests:
python failure_memory_test.py

python failure_recovery_test.py

🔐 Security
The Gemini API key is loaded through:
GEMINI_API_KEY

The .env file is excluded through .gitignore.
Never place API keys directly inside Python source files or commit them to GitHub.
📌 Project Scope
This is a student-scale Agentic AI research project focused on:
- Retrieval-Augmented Generation
- Semantic external memory
- Agentic orchestration
- Reflection
- Knowledge curation
- Failure memory
- Curriculum grounding
- Controlled evaluation
The project is inspired by concepts from DeepEdu-v1, but it does not attempt to reproduce the complete research system.
👨‍💻 Author
Kabilan
B.Tech — Artificial Intelligence & Machine Learning