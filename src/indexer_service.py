import os
import shutil
from pathlib import Path
from typing import List
import logging
from tqdm import tqdm
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.docstore.document import Document

from src import constants
from src.utils import EmbeddingModel
logger = logging.getLogger()

class IndexerService:
    """generate chroma db index
    """

    def __init__(self) -> None:
        emb=EmbeddingModel()
        self.embeddings = emb.get_model()

    def load_pdfs(self, folder_path: Path) -> List[Document]:
        """ Load the PDF file and split it into smaller chunks
        PyPDFDirectoryLoader can also load all pdf, 
        but in order to have control over metadata we are using loader manually
        """
        docs = []
        logger.info("Starting with loading pdfs and creating chunks")
        for file_path in tqdm(folder_path.glob("**/[!.]*.pdf")):
            if file_path.is_file():
                loader = PyPDFLoader(str(file_path))
                sub_docs = loader.load()
                for doc in sub_docs:
                    doc.metadata["source"] = file_path.name
                    # below is done because langchain start pdf page_no with 0
                    doc.metadata["page"] = doc.metadata["page"] + 1
                docs.extend(sub_docs)
        return docs

    def load_data_folder(self, folder_path: Path) -> List[Document]:
        """loads data folder there you can use multiple loader and extend for now we have only pdf loader"""
        all_docs = []
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
            raise RuntimeError(f"please add file in the {folder_path} folder")
        all_docs.extend(self.load_pdfs(folder_path))
        return all_docs

    def split_documents(self, list_docs: List[Document], chunk_size: int = 1500, separator: str = "\n") -> List[Document]:
        """split the document in 1500 character limit, embedding model maximum support 512 token"""
        if len(list_docs)==0:
            raise RuntimeError("could not load file.")
        text_splitter = CharacterTextSplitter(
            chunk_size=chunk_size, separator=separator)
        chunks = text_splitter.split_documents(list_docs)
        return chunks

    def create_save_embedding(self, list_docs: List[Document]) -> None:
        """Index the vector database by embedding then inserting document chunks"""
        db_folder = constants.DB_FOLDER
        if not os.path.exists(db_folder):
            os.mkdir(db_folder)
        vector_db_path = os.path.join(db_folder, constants.COLLECTION_NAME)
        if os.path.exists(vector_db_path) and constants.RECREATE_COLLECTION_FOLDER:
            shutil.rmtree(vector_db_path)
            logger.info("deleted folder at path: %s", str(vector_db_path))

        logger.info("Starting with creating embeddings")
        ch_db = Chroma.from_documents(list_docs,
                                   embedding=self.embeddings,
                                   persist_directory=vector_db_path)
        logger.info("Starting with storing embeddings")
        ch_db.persist()


if __name__ == "__main__":
    cis = IndexerService()
    all_docs = cis.load_data_folder(Path(constants.DATA_FOLDER))
    all_docs = cis.split_documents(all_docs)
    cis.create_save_embedding(all_docs)
