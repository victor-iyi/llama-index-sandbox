# import logging
# import sys
import os
from dotenv import load_dotenv

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage
)

load_dotenv()

# To view verbose output of what's happening in the background.
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Path to store index.
PERSIST_DIR: str = 'res/index'


def main() -> None:
    # storing index in "res/index/"
    if not os.path.exists(PERSIST_DIR):
        # Load documents and create the index.
        documents = SimpleDirectoryReader('res/data').load_data()
        index = VectorStoreIndex.from_documents(documents)

        # Store it for later.
        index.storage_context.persist(persist_dir=PERSIST_DIR)
    else:
        # Load the existing index.
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)

    # Query the index.
    query_engine = index.as_query_engine()
    while (question := input('Ask question about essay (q to quit): ')) != 'q':
        response = query_engine.query(question)
        print(f'{response}\n')


if __name__ == '__main__':
    main()
