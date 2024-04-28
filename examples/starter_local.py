# import os
from dotenv import load_dotenv
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    VectorStoreIndex,
)

# StorageContext,
# load_index_from_storage,
from llama_index.core.embeddings import resolve_embed_model
from llama_index.llms.ollama import Ollama

load_dotenv()


LOCAL_MODEL: str = 'llama3'
# PERSIST_DIR: str = f'res/index/{LOCAL_MODEL}'


def main() -> None:
    """Starting point of the script."""
    # if not os.path.exists(PERSIST_DIR):
    # Load documents.
    documents = SimpleDirectoryReader('res/data').load_data()

    # bge embedding model.
    Settings.embed_model = resolve_embed_model('local:BAAI/bge-small-en-v1.5')
    # Use local ollama model.
    Settings.llm = Ollama(model=LOCAL_MODEL, request_timeout=30.0)

    # Create vector store from documents.
    index = VectorStoreIndex.from_documents(documents=documents)

    # Store index.
    # index.storage_context.persist(persist_dir=PERSIST_DIR)
    # else:
    #     # Load index.
    #     storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    #     index = load_index_from_storage(storage_context=storage_context)

    # Query engine from index.
    query_engine = index.as_query_engine()

    while (prompt := input('Ask question here (q to quit): ')) != 'q':
        response = query_engine.query(prompt)
        print(f'{response}\n')


if __name__ == '__main__':
    main()
