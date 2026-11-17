
import pandas as pd
from pathlib import Path

from pdf_qa import (
    extract_pdf_text,
    split_text,
    create_vector_store
)

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent
PDF_PATH = PROJECT_DIR / "dataset" / "sample.pdf"
DATASET_DIR = PROJECT_DIR / "dataset"

# Check PDF
if not PDF_PATH.exists():
    raise FileNotFoundError("sample.pdf not found in dataset folder!")

# 1. Extract PDF text
print("\n1. Extracting PDF text...")

extracted_text, total_pages = extract_pdf_text(str(PDF_PATH))

(DATASET_DIR / "extracted_text.txt").write_text(
    extracted_text,
    encoding="utf-8"
)

print("PDF Pages:", total_pages)
print("Total Characters:", len(extracted_text))


# 2. Experiment with chunk sizes and overlap
print("\n2. Experimenting with chunk sizes...")

settings = [
    (300, 50),
    (500, 100),
    (800, 150),
    (1000, 200)
]

experiment_results = []

for chunk_size, overlap in settings:

    chunks = split_text(
        extracted_text,
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )

    experiment_results.append({
        "Chunk_Size": chunk_size,
        "Chunk_Overlap": overlap,
        "Total_Chunks": len(chunks)
    })

    print(
        f"Chunk Size: {chunk_size}, "
        f"Overlap: {overlap}, "
        f"Total Chunks: {len(chunks)}"
    )


# Save experiment results
experiment_df = pd.DataFrame(experiment_results)

experiment_df.to_csv(
    DATASET_DIR / "chunk_experiments.csv",
    index=False
)


# 3. Select chunk size 500 and overlap 100
print("\n3. Creating ChromaDB collection...")

selected_chunks = split_text(
    extracted_text,
    chunk_size=500,
    chunk_overlap=100
)

chunks_df = pd.DataFrame({
    "Chunk_ID": [
        f"chunk_{i}" for i in range(len(selected_chunks))
    ],
    "Content": selected_chunks
})

chunks_df.to_csv(
    DATASET_DIR / "chunks.csv",
    index=False
)

collection = create_vector_store(selected_chunks)

print("\nChromaDB collection created successfully!")
print("Collection Name: week14_pdf_collection")
print("Total Stored Chunks:", collection.count())

print("\nAll experiments completed successfully!")
