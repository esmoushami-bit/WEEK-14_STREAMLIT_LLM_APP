
import pandas as pd
import chromadb

from pathlib import Path
from sentence_transformers import SentenceTransformer
from pdf_qa import generate_answer

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "dataset"
DB_PATH = PROJECT_DIR / "chroma_db"

# 1. Load embedding model
print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# 2. Connect to ChromaDB
client = chromadb.PersistentClient(
    path=str(DB_PATH)
)

collection = client.get_collection(
    name="week14_pdf_collection"
)

print("ChromaDB connected successfully!")
print("Total chunks:", collection.count())


# 3. Prepare 15 questions
questions = [
    "What is Artificial Intelligence?",
    "What are the goals of Artificial Intelligence?",
    "Explain machine learning.",
    "What is deep learning?",
    "What is Natural Language Processing?",
    "Explain computer vision.",
    "What is Generative AI?",
    "What is Transformer architecture?",
    "What are Large Language Models?",
    "Explain Retrieval-Augmented Generation.",
    "What are text embeddings?",
    "What is ChromaDB?",
    "Explain semantic search.",
    "What is keyword-based search?",
    "What is Ollama?"
]

pd.DataFrame({
    "Question": questions
}).to_csv(
    DATASET_DIR / "queries.csv",
    index=False
)


# 4. Retrieve Top-5 relevant chunks
print("\nRetrieving Top-5 relevant chunks...")

retrieval_records = []
comparison_records = []

for question_number, question in enumerate(questions, start=1):

    print(
        f"\nProcessing Question {question_number}/15..."
    )

    query_embedding = embedding_model.encode(
        [question]
    ).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(5, collection.count())
    )

    retrieved_documents = results["documents"][0]
    retrieved_ids = results["ids"][0]
    distances = results["distances"][0]

    # Store retrieved chunks
    for rank, (chunk_id, document, distance) in enumerate(
        zip(retrieved_ids, retrieved_documents, distances),
        start=1
    ):

        retrieval_records.append({
            "Question": question,
            "Rank": rank,
            "Chunk_ID": chunk_id,
            "Retrieved_Chunk": document,
            "Similarity_Score": 1 - distance
        })

    # Combine retrieved chunks
    context = "\n\n".join(retrieved_documents)

    # Answer without retrieval
    answer_without_rag = generate_answer(
        question,
        context=""
    )

    # Answer with retrieval
    answer_with_rag = generate_answer(
        question,
        context=context
    )

    comparison_records.append({
        "Question": question,
        "Answer_Without_Retrieval": answer_without_rag,
        "Answer_With_Retrieval": answer_with_rag
    })


# 5. Save retrieved chunks
retrieval_df = pd.DataFrame(retrieval_records)

retrieval_df.to_csv(
    DATASET_DIR / "retrieval_results.csv",
    index=False
)


# 6. Save answer comparison
comparison_df = pd.DataFrame(comparison_records)

comparison_df.to_csv(
    DATASET_DIR / "answer_comparison.csv",
    index=False
)


# 7. Save RAG answers separately
rag_answers_df = comparison_df[
    ["Question", "Answer_With_Retrieval"]
].rename(columns={
    "Answer_With_Retrieval": "Answer"
})

rag_answers_df.to_csv(
    DATASET_DIR / "answers.csv",
    index=False
)


# 8. Display results
print("\n--------------------------------")
print("WEEK 14 RETRIEVAL COMPLETED")
print("--------------------------------")

print("Total Questions:", len(questions))
print("Top-K Chunks: 5")
print("Total Retrieved Chunks:", len(retrieval_records))
print("Total Answers:", len(comparison_records))

print("\nCSV files saved successfully!")
print("1. queries.csv")
print("2. retrieval_results.csv")
print("3. answer_comparison.csv")
print("4. answers.csv")

