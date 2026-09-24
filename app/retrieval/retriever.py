from langchain_openai import OpenAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch

from app.db.mongodb import collection


# --------------------------------------------------
# Configuration
# --------------------------------------------------

EMBEDDING_MODEL = "text-embedding-3-small"
INDEX_NAME = "vector_index"

TOP_K = 5


# --------------------------------------------------
# Create vector store
# --------------------------------------------------

embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL
)

vector_store = MongoDBAtlasVectorSearch(
    collection=collection,
    embedding=embeddings,
    index_name=INDEX_NAME,
    text_key="text",
    embedding_key="embedding",
)


# --------------------------------------------------
# Retrieve relevant documents
# --------------------------------------------------

def retrieve(query: str, k: int = TOP_K):
    results = vector_store.similarity_search_with_score(
        query=query,
        k=k,
    )

    return results


# --------------------------------------------------
# Test retrieval
# --------------------------------------------------

def main():
    query = "What were the major features of Mauryan administration?"

    print("\nQuery:")
    print(query)

    print("\nRetrieved documents:\n")

    results = retrieve(query)

    for i, (document, score) in enumerate(results, start=1):
        print("=" * 80)
        print(f"Result #{i}")
        print("=" * 80)

        print(f"\nScore: {score}")

        print("\nContent:")
        print(document.page_content)

        print("\nMetadata:")
        print(document.metadata)

        print()


if __name__ == "__main__":
    main()