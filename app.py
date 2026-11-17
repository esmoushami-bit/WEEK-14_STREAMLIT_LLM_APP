
import streamlit as st
import time
import hashlib
import re

from pathlib import Path

from pdf_qa import (
    extract_pdf_text,
    split_text,
    create_vector_store,
    retrieve_chunks,
    generate_answer
)


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="PDF Question Answering System",
    page_icon="📚",
    layout="wide"
)


# ==========================================
# 2. PROJECT PATHS
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = PROJECT_DIR / "dataset" / "uploaded_files"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 3. SESSION STATE
# ==========================================

if "documents" not in st.session_state:
    st.session_state.documents = {}

if "chat_histories" not in st.session_state:
    st.session_state.chat_histories = {}

if "active_pdf" not in st.session_state:
    st.session_state.active_pdf = "All PDFs"


# ==========================================
# 4. HELPER FUNCTIONS
# ==========================================

def collection_name(filename):

    file_hash = hashlib.md5(
        filename.encode()
    ).hexdigest()[:12]

    safe_name = re.sub(
        r"[^a-zA-Z0-9_-]",
        "_",
        Path(filename).stem
    )[:30]

    return f"pdf_{safe_name}_{file_hash}"


def get_history(pdf_name):

    if pdf_name not in st.session_state.chat_histories:

        st.session_state.chat_histories[pdf_name] = []

    return st.session_state.chat_histories[pdf_name]


# ==========================================
# 5. APPLICATION TITLE
# ==========================================

st.title("📚 PDF Question Answering System")

st.write(
    "Upload multiple PDF documents and ask questions "
    "using AI-powered Retrieval Augmented Generation (RAG)."
)

st.divider()


# ==========================================
# 6. SIDEBAR
# ==========================================

with st.sidebar:

    st.header("⚙️ Settings")

    st.write("LLM Model: llama3.2")

    st.write("Embedding Model:")

    st.caption("all-MiniLM-L6-v2")

    st.divider()

    st.subheader("Chunk Settings")

    chunk_size = st.slider(
        "Chunk Size",
        300, 1000, 500, 100
    )

    chunk_overlap = st.slider(
        "Chunk Overlap",
        50, 200, 100, 50
    )

    st.divider()

    st.subheader("Retrieval Settings")

    top_k = st.slider(
        "Top-K Retrieved Chunks",
        min_value=3,
        max_value=15,
        value=10
    )

    st.caption(
        "The application ranks chunks across all selected PDFs."
    )

    st.divider()

    st.subheader("Uploaded Documents")

    st.write(
        f"Total PDFs: {len(st.session_state.documents)}"
    )

    if st.button("🗑️ Clear All Chat Histories"):

        st.session_state.chat_histories = {}

        st.rerun()

    if st.button("🗑️ Remove All PDFs"):

        st.session_state.documents = {}
        st.session_state.chat_histories = {}
        st.session_state.active_pdf = "All PDFs"

        st.rerun()

    st.divider()

    st.caption(
        "Week 14 - Streamlit LLM Application"
    )


# ==========================================
# 7. MULTIPLE PDF UPLOAD
# ==========================================

st.header("📄 Upload PDF Documents")

uploaded_files = st.file_uploader(
    "Choose one or more PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:

    if st.button("🚀 Process Uploaded PDFs"):

        for uploaded_file in uploaded_files:

            filename = uploaded_file.name

            if filename in st.session_state.documents:

                st.info(
                    f"{filename} is already processed."
                )

                continue

            try:

                with st.spinner(
                    f"Processing {filename}..."
                ):

                    pdf_path = (
                        UPLOAD_DIR / Path(filename).name
                    )

                    pdf_path.write_bytes(
                        uploaded_file.getvalue()
                    )

                    # Extract PDF text
                    extracted_text, total_pages = (
                        extract_pdf_text(str(pdf_path))
                    )

                    if not extracted_text.strip():

                        st.warning(
                            f"No readable text found in {filename}."
                        )

                        continue

                    # Split into chunks
                    chunks = split_text(
                        extracted_text,
                        chunk_size,
                        chunk_overlap
                    )

                    # Create separate collection
                    name = collection_name(filename)

                    collection = create_vector_store(
                        chunks,
                        collection_name=name
                    )

                    # Store document details
                    st.session_state.documents[filename] = {

                        "collection": collection,

                        "total_pages": total_pages,

                        "total_chunks": len(chunks),

                        "collection_name": name

                    }

                    st.success(
                        f"{filename} processed successfully!"
                    )

            except Exception as e:

                st.error(
                    f"Error processing {filename}: {e}"
                )

        st.rerun()


# ==========================================
# 8. SELECT PDF DOCUMENT
# ==========================================

if st.session_state.documents:

    st.divider()

    st.header("📂 Select PDF Document")

    pdf_names = [
        "All PDFs"
    ] + list(
        st.session_state.documents.keys()
    )

    if st.session_state.active_pdf not in pdf_names:

        st.session_state.active_pdf = "All PDFs"

    selected_pdf = st.selectbox(
        "Choose a PDF to ask questions",
        pdf_names,
        index=pdf_names.index(
            st.session_state.active_pdf
        )
    )

    st.session_state.active_pdf = selected_pdf


    # ======================================
    # 9. SELECT DOCUMENTS FOR SEARCH
    # ======================================

    if selected_pdf == "All PDFs":

        documents_to_search = list(
            st.session_state.documents.items()
        )

        total_pages = sum(
            doc["total_pages"]
            for doc in st.session_state.documents.values()
        )

        total_chunks = sum(
            doc["total_chunks"]
            for doc in st.session_state.documents.values()
        )

    else:

        documents_to_search = [

            (
                selected_pdf,
                st.session_state.documents[selected_pdf]
            )

        ]

        total_pages = (
            st.session_state.documents[selected_pdf]["total_pages"]
        )

        total_chunks = (
            st.session_state.documents[selected_pdf]["total_chunks"]
        )


    # ======================================
    # 10. PDF INFORMATION
    # ======================================

    st.subheader("📊 PDF Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "PDF Pages",
            total_pages
        )

    with col2:

        st.metric(
            "Total Chunks",
            total_chunks
        )

    with col3:

        st.metric(
            "Top-K Retrieval",
            top_k
        )

    st.success(
        f"Active Selection: {selected_pdf}"
    )

    st.divider()


    # ======================================
    # 11. CHAT HISTORY
    # ======================================

    st.header("💬 Ask Questions")

    messages = get_history(selected_pdf)

    for message in messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

            if "response_time" in message:

                st.caption(
                    f"⏱️ Response Time: "
                    f"{message['response_time']} seconds"
                )

            if "sources" in message:

                with st.expander(
                    "📚 View Retrieved Chunks"
                ):

                    for i, source in enumerate(
                        message["sources"],
                        start=1
                    ):

                        st.markdown(
                            f"**Retrieved Chunk {i}**"
                        )

                        st.caption(
                            f"Source PDF: {source['filename']}"
                        )

                        st.write(source["content"])

                        st.caption(
                            f"Distance Score: "
                            f"{source['distance']:.4f}"
                        )

                        st.divider()


    # ======================================
    # 12. QUESTION INPUT
    # ======================================

    question = st.chat_input(
        f"Ask something about {selected_pdf}..."
    )

    if question:

        messages.append({

            "role": "user",

            "content": question

        })

        with st.chat_message("user"):

            st.markdown(question)

        try:

            start_time = time.time()

            all_retrieved_chunks = []

            with st.spinner(
                "Searching all selected PDFs..."
            ):

                # Retrieve Top-K from each PDF
                for filename, pdf_document in documents_to_search:

                    results = retrieve_chunks(
                        pdf_document["collection"],
                        question,
                        top_k=top_k
                    )

                    retrieved_chunks = (
                        results["documents"][0]
                    )

                    distances = (
                        results["distances"][0]
                    )

                    for chunk, distance in zip(
                        retrieved_chunks,
                        distances
                    ):

                        all_retrieved_chunks.append({

                            "filename": filename,

                            "content": chunk,

                            "distance": float(distance)

                        })


                # Rank all chunks globally
                all_retrieved_chunks.sort(
                    key=lambda item: item["distance"]
                )


                # Select globally best Top-K
                retrieved_documents = (
                    all_retrieved_chunks[:top_k]
                )


                # Prepare source-labelled context
                context_parts = []

                for source in retrieved_documents:

                    context_parts.append(
                        f"DOCUMENT NAME: {source['filename']}\n"
                        f"DOCUMENT CONTENT:\n{source['content']}"
                    )

                context = "\n\n".join(
                    context_parts
                )


                # Improved question instructions
                if selected_pdf == "All PDFs":

                    prompt_question = f"""
You are an accurate PDF Question Answering Assistant.

The user has uploaded multiple PDF documents.

Answer the user's question using ONLY the retrieved PDF context.

IMPORTANT INSTRUCTIONS:

1. Identify each source PDF using its exact document name.
2. If asked for a summary of both PDFs, summarize each document separately.
3. If asked about characters, identify the characters and explain
   their roles using the available PDF information.
4. Compare information from different PDFs when requested.
5. Keep information from different PDFs clearly separated.
6. Do not mix up the titles, authors, characters or events.
7. Do not invent facts or unsupported details.
8. If a document does not contain enough information, say so.
9. Provide a clear, detailed, well-organized answer.

USER QUESTION:
{question}
"""

                else:

                    prompt_question = f"""
Answer the user's question using only the provided PDF context.

Instructions:
1. Give a clear and detailed answer.
2. Include important names and facts.
3. Do not invent information.
4. If the answer is unavailable, say so.

USER QUESTION:
{question}
"""


                # Generate answer
                answer = generate_answer(
                    prompt_question,
                    context
                )


            response_time = round(
                time.time() - start_time,
                2
            )


            # Save assistant response
            messages.append({

                "role": "assistant",

                "content": answer,

                "sources": retrieved_documents,

                "response_time": response_time

            })

            st.rerun()


        except Exception as e:

            st.error(
                f"Error generating answer: {e}"
            )


else:

    st.info(
        "Upload and process PDF documents to start chatting."
    )


# ==========================================
# 13. FOOTER
# ==========================================

st.divider()

st.caption(
    "Week 14 | Streamlit + LangChain + ChromaDB + Ollama"
)
