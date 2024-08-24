import sys
import os
from collections import defaultdict
from pathlib import Path
import logging
from datetime import datetime

from src.indexer_service import IndexerService
from src.retriever_service import RetrieverService
from src.utils import timer
from src import constants

if not os.path.exists(constants.LOG_FOLDER):
    os.mkdir(constants.LOG_FOLDER)
log_file_path = constants.LOG_FOLDER + "/" + datetime.now().strftime("%Y-%m-%d %H-%M-%S") + ".log"
logger = logging.getLogger()
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
file_handler = logging.FileHandler(log_file_path, encoding='utf8')
logger.addHandler(stream_handler)
logger.addHandler(file_handler)

@timer("building index completed in: ")
def rebuild_index():
    """this function rebuild index"""
    cis = IndexerService()
    all_docs = cis.load_data_folder(Path(constants.DATA_FOLDER))
    all_docs = cis.split_documents(all_docs)
    cis.create_save_embedding(all_docs)

if __name__ == "__main__":
    BUILD_INDEX=True
    if BUILD_INDEX:
        rebuild_index()

    rc = RetrieverService()
    test_result = rc.get_relevant_document("who developed the pyramid?", {
                                           "source": "general.pdf"})
    # print(test_result)
    chat_history = []
    query = None
    ch_re = rc.get_chroma_retriever()
    op_chain = rc.get_openai_retrivalchain(ch_re)
    while True:
        if not query:
            query = input("Prompt: ")
        if query in ['quit', 'q', 'exit']:
            sys.exit()
        with timer("getting answer took: "):
            result = op_chain({"question": query, "chat_history": chat_history})
        metadata_dict = defaultdict(list)
        logger.info(query)
        logger.info(result['answer'])
        for doc in result.get("source_documents"):
            page_no = doc.metadata.get("page", 1)
            metadata_dict[doc.metadata["source"]].append(page_no)
        logger.info(metadata_dict)

        chat_history.append((query, result['answer']))
        query = None
