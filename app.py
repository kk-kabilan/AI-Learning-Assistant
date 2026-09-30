import streamlit as st

from agent import agent

from rag import (
    load_pdf,
    chunk_text,
    create_vector_index
)


# ==================================================
# Page configuration
# ==================================================

st.set_page_config(
    page_title="AI Learning Assistant",
    page_icon="🤖",
    layout="centered"
)


# ==================================================
# Title
# ==================================================

st.title(
    "🤖 AI Learning Assistant"
)

st.write(
    "Ask questions about your study material."
)


# ==================================================
# Load knowledge base
# ==================================================

@st.cache_resource
def setup():

    text = load_pdf(
        "knowledge/DBMS.pdf"
    )

    chunks = chunk_text(
        text
    )

    model, index = create_vector_index(
        chunks
    )

    return model, index, chunks


# ==================================================
# Initialize RAG system
# ==================================================

model, index, chunks = setup()


# ==================================================
# User input
# ==================================================

query = st.text_input(
    "Ask your question:"
)


# ==================================================
# Ask button
# ==================================================

if st.button(
    "Ask Tutor"
):

    if not query.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        with st.spinner(
            "AI Tutor is thinking..."
        ):

            answer = agent(
                query,
                model,
                index,
                chunks
            )


        # ------------------------------------------
        # Display answer
        # ------------------------------------------

        st.subheader(
            "🤖 AI Tutor Answer"
        )

        st.write(
            answer
        )