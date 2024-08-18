import chromadb
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
)
from llama_index.vector_stores.chroma import ChromaVectorStore

from dotenv import load_dotenv

# Load environment variables.
load_dotenv()

DATA_PATH: str = 'res/data'
CHROMA_PATH: str = 'res/chroma'
CHROMA_COLLECTION: str = 'paul_graham_story'


def main() -> None:

    # Initialize client, setting path to save data.
    db = chromadb.PersistentClient(CHROMA_PATH)

    # Check if collection exists.
    collection_exists = False
    if any(collection.name == CHROMA_COLLECTION for collection in db.list_collections()):
        collection_exists = True

    # Create collection.
    collection = db.get_or_create_collection(CHROMA_COLLECTION)

    # Assign chroma as the vector store.
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    if not collection_exists:
        # Load the documents
        print('Loading documents...')
        documents = SimpleDirectoryReader(DATA_PATH).load_data()

        # Create index.
        print('Creating index...')
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True,
        )
    else:
        # Load index from vector store.
        print('Loading index...')
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context,
        )

    # Create query engine.
    query_engine = index.as_query_engine()
    while (question := input('Ask questions about the essay (q to quit): ')) != 'q':
        response = query_engine.query(question)
        print(f'Answer: {response}\n')


if __name__ == '__main__':
    main()
