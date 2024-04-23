# import logging
# import sys
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

load_dotenv()

# To view verbose output of what's happening in the background.
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


def main() -> None:
    documents = SimpleDirectoryReader('res/data').load_data()
    index = VectorStoreIndex.from_documents(documents)

    query_engine = index.as_query_engine()

    while (question := input('Ask question about essay (q to quit): ')) != 'q':
        response = query_engine.query(question)
        print(f'{response}\n')


if __name__ == '__main__':
    main()
