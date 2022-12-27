"""Module chứa các lớp Datasets."""
from typing import Union, List
from transformers import (
    PreTrainedTokenizerFast,
    PreTrainedTokenizer
)
import numpy as np
from abc import ABC, abstractmethod


class QuestionAnsweringDataset(ABC):
    """Abstract class cho các lớp Question Answering Datasets."""
    use_wordsegment = False
    reader_class = None

    def __init__(
        self,
        tokenizer: Union[PreTrainedTokenizerFast, PreTrainedTokenizer],
        **kwargs,
    ):
        self.tokenizer = tokenizer

    @abstractmethod
    def get_train_dataset(self, main_process_first=None):
        pass

    @abstractmethod
    def get_eval_dataset(self, main_process_first=None):
        pass

    @abstractmethod
    def get_predict_dataset(self, main_process_first=None):
        pass

    @abstractmethod
    def prepare_train_features(self, examples: dict) -> dict:
        pass

    @abstractmethod
    def prepare_validation_features(self, examples: dict) -> dict:
        pass

    @abstractmethod
    def post_processing_function(self, examples, features, predictions, stage="eval"):
        pass

    @abstractmethod
    def preprocess_text(self, text: str) -> str:
        pass

    @abstractmethod
    def decode_model_outputs(
        self,
        context_map: list,
        start_scores: np.ndarray,
        end_scores: np.ndarray,
        has_answer_scores: np.ndarray,
        threshold: float
    ):
        pass

    @abstractmethod
    def batch_get_inputs_for_generation(self, questions: List[str], context: List[List[dict]]):
        pass
