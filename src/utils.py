import time
from typing import Union
from contextlib import contextmanager
import logging
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings

from src import constants
logger = logging.getLogger()

@contextmanager
def timer(message):
    """to calculate timing ref: https://www.learndatasci.com/solutions/python-timer/"""
    t_0 = time.perf_counter()
    try:
        yield
    finally:
        t_1 = time.perf_counter()
        elapsed = t_1 - t_0
        logger.info('%s %0.4f',message ,elapsed)

class SingletonClass(type):
    """add this class as metaclass for make it singleton"""
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonClass, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class EmbeddingModel(metaclass=SingletonClass):
    """class to get embedding model"""
    def __init__(self) -> None:
        if constants.EMBEDDING_MODEL_TYPE=="openai":
            self.model=OpenAIEmbeddings(model="text-embedding-ada-002")
            self.max_seq_length=self.model.chunk_size
        elif constants.EMBEDDING_MODEL_TYPE=="huggingface":
            self.model=HuggingFaceEmbeddings(
                model_name=constants.EMBEDDING_MODEL_NAME,
                cache_folder=constants.SENTENCE_TRANSFORMERS_HOME
                #,model_kwargs = {"device": constants.TORCH_DEVICE}
            )
            # self.max_seq_length=self.model.client[0].max_seq_length
            self.max_seq_length=self.model.client.tokenizer.model_max_length
        else:
            raise RuntimeError("model type is not valid")

    def get_model(self) -> Union[HuggingFaceEmbeddings,OpenAIEmbeddings]:
        """to get embedding model"""
        return self.model
