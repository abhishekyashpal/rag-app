from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mongodb import MongoDBAtlasVectorSearch

from app.db.mongodb import collection


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PDF_DIRECTORY = Path("data/raw")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

EMBEDDING_MODEL = "text-embedding-3-small"


# --------------------------------------------------
# Load PDFs
# --------------------------------------------------

def load_pdfs():
    documents = []

    pdf_files = list(PDF_DIRECTORY.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {PDF_DIRECTORY}"
        )

    for pdf_file in pdf_files:
        print(f"Loading: {pdf_file.name}")

        loader = PyPDFLoader(str(pdf_file))
        pdf_documents = loader.load()

        documents.extend(pdf_documents)

        print(f"  Pages loaded: {len(pdf_documents)}")

    return documents


# --------------------------------------------------
# Split documents into chunks
# --------------------------------------------------

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = splitter.split_documents(documents)

    return chunks


# --------------------------------------------------
# Add metadata
# --------------------------------------------------

def enrich_metadata(chunks):
    for index, chunk in enumerate(chunks):
        source = Path(chunk.metadata["source"]).name

        chunk.metadata["source"] = source
        chunk.metadata["chunk_id"] = index

    return chunks


# --------------------------------------------------
# Generate embeddings and store in MongoDB
# --------------------------------------------------

def store_documents(chunks):
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL
    )

    vector_store = MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embeddings,
        index_name="vector_index",
        text_key="text",
        embedding_key="embedding",
    )

    vector_store.add_documents(chunks)


# --------------------------------------------------
# Main ingestion pipeline
# --------------------------------------------------

def run_ingestion():
    print("\nStarting RAG ingestion...\n")

    # 1. Load PDFs
    documents = load_pdfs()

    print(f"\nTotal pages loaded: {len(documents)}")

    # 2. Split into chunks
    chunks = split_documents(documents)

    print(f"Total chunks created: {len(chunks)}")

    # 3. Add metadata
    chunks = enrich_metadata(chunks)

    # 4. Generate embeddings + store in MongoDB
    print("\nGenerating embeddings and storing in MongoDB...")

    store_documents(chunks)

    print("\nIngestion completed successfully!")
    print(f"Documents stored: {len(chunks)}")


if __name__ == "__main__":
    run_ingestion()