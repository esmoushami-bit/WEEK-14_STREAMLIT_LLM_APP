
import chromadb

from pathlib import Path
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM


# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent
DB_PATH = PROJECT_DIR / "chroma_db"

DB_PATH.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. LOAD MODELS
# ==========================================

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

llm = OllamaLLM(model="llama3.2")


# ==========================================
# 3. CHROMADB CLIENT
# ==========================================

client = chromadb.PersistentClient(
    path=str(DB_PATH)
)


# ==========================================
# 4. EXTRACT PDF TEXT
# ==========================================

def extract_pdf_text(pdf_path):

    reader = PdfReader(pdf_path)

    extracted_text = ""

    for page_number, page in enumerate(
        reader.pages, start=1
    ):

        text = page.extract_text()

        if text:

            extracted_text += (
                f"\n--- Page {page_number} ---\n{text}"
            )

    return extracted_text, len(reader.pages)


# ==========================================
# 5. SPLIT TEXT INTO CHUNKS
# ==========================================

def split_text(
    text,
    chunk_size=500,
    chunk_overlap=100
):

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "Chunk overlap must be smaller than chunk size."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    return splitter.split_text(text)


# ==========================================
# 6. CREATE SEPARATE VECTOR STORE
# ==========================================

def create_vector_store(
    chunks,
    collection_name="week14_pdf_collection"
):

    if not chunks:
        raise ValueError("No chunks available for indexing.")

    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    embeddings = embedding_model.encode(
        chunks,
        show_progress_bar=True
    )

    collection.upsert(
        ids=[
            f"chunk_{i}" for i in range(len(chunks))
        ],
        documents=chunks,
        embeddings=embeddings.tolist()
    )

    return collection


# ==========================================
# 7. RETRIEVE TOP-K CHUNKS
# ==========================================

def retrieve_chunks(
    collection,
    question,
    top_k=5
):

    query_embedding = embedding_model.encode(
        [question]
    ).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, collection.count())
    )

    return results


# ==========================================
# 8. GENERATE ANSWER
# ==========================================

def generate_answer(
    question,
    context=""
):

    if context:

        prompt = f"""
You are a PDF Question Answering Assistant.

Answer the question using only the information
provided in the PDF context.

Instructions:
1. Give a clear and accurate answer.
2. Do not add unsupported information.
3. Do not invent facts.
4. If the answer is not available in the context,
   clearly state that it is not available in the PDF.

PDF Context:
{context}

Question:
{question}

Answer:
"""

    else:

        prompt = f"""
Answer the following question using your
general knowledge.

Question:
{question}

Answer:
"""

    return llm.invoke(prompt)
