import os
import sys
from collections import defaultdict
from typing import Dict, Optional
import langchain
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Chroma
from langchain.indexes.vectorstore import VectorStoreIndexWrapper

from langchain.schema.vectorstore import VectorStoreRetriever
from langchain.chains.conversational_retrieval.base import BaseConversationalRetrievalChain

from src import constants
from src.utils import SingletonClass, EmbeddingModel


class RetrieverService(metaclass=SingletonClass):
    """this is to retrieve document from chroma db index"""

    def __init__(self) -> None:
        """ we made this class singleton , so we do not create multiple instance of chroma db"""
        self.verbose = False
        langchain.verbose = self.verbose
        emb=EmbeddingModel()
        self.embeddings = emb.get_model()
        vector_db_path = os.path.join(
            constants.DB_FOLDER, constants.COLLECTION_NAME)
        self.vectorstore = Chroma(embedding_function=self.embeddings,
                                  persist_directory=vector_db_path)
        self.index = VectorStoreIndexWrapper(vectorstore=self.vectorstore)

    def get_chroma_retriever(self, metadata_filter: Optional[Dict[str, str]] = None, number_of_match: int = 3) -> VectorStoreRetriever:
        """ this function return retriever from chroma db"""
        chroma_retriever = self.index.vectorstore.as_retriever(
            search_kwargs={"k": number_of_match, 'filter': metadata_filter})
        return chroma_retriever

    def get_openai_retrivalchain(self, retriever_obj: VectorStoreRetriever, model_name: str = "gpt-3.5-turbo") -> BaseConversationalRetrievalChain:
        """ this function returns openaichain, this automatically takes care of making query independent"""
        chain = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(model=model_name, temperature=0),
            retriever=retriever_obj, return_source_documents=True, verbose=self.verbose
        )
        return chain

    def get_relevant_document(self, query: str, metadata_filter: Optional[Dict[str, str]] = None):
        """getting most similar doc from chroma db"""
        result = self.vectorstore.similarity_search(
            query, k=4, filter=metadata_filter, include_metadata=True)
        return result


if __name__ == "__main__":
    rc = RetrieverService()
    test_result = rc.get_relevant_document("who developed the pyramid?", {
                                           "source": "general.pdf"})
    # print(test_result)
    chat_history = []
    query = None
    while True:
        if not query:
            query = input("Prompt: ")
        if query in ['quit', 'q', 'exit']:
            sys.exit()
        ch_re = rc.get_chroma_retriever()
        op_chain = rc.get_openai_retrivalchain(ch_re)
        result = op_chain({"question": query, "chat_history": chat_history})
        metadata_dict = defaultdict(list)
        print(result['answer'])
        for doc in result.get("source_documents"):
            page_no = doc.metadata.get("page", "1")
            metadata_dict[doc.metadata["source"]].append(page_no)
        print(metadata_dict)

        chat_history.append((query, result['answer']))
        query = None
