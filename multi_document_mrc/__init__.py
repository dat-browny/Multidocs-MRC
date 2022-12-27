"""Module chứa các lớp models."""
from multi_document_mrc.models_map import get_model_version_classes
from multi_document_mrc.mydatasets import QuestionAnsweringDataset
from datetime import datetime
from typing import List
import numpy as np
import logging
import time
import torch
import json
import os


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiDocMRC:
    def __init__(
        self,
        models_path: str = "mrc_models",
        max_seq_length: int = 256,
    ):
        """Hàm tạo."""
        self.model_architecture_name = self.get_model_architecture(models_path)
        self.model_architecture = get_model_version_classes(self.model_architecture_name)
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        # Tokenizer Object
        self.tokenizer = self.model_architecture.tokenizer_class.from_pretrained(models_path)
        self.model_config = self.model_architecture.config_class.from_pretrained(models_path)
        self.model = self.model_architecture.model_class.from_pretrained(models_path)
        self.model.eval()
        self.model.to(self.device)

        self.dataset_obj: QuestionAnsweringDataset = self.model_architecture.dataset_class(
            tokenizer=self.tokenizer,
            max_seq_length=max_seq_length
        )

    @staticmethod
    def get_model_architecture(models_path):
        with open(os.path.join(models_path, "config.json"), "r") as file:
            config = json.load(file)
        return config["model_architecture"]

    def generate_responses(
        self,
        data: List[dict],
        n_contexts: int = 10,
        threshold: float = 0.5,
        silent: bool = True,
    ) -> List[str]:
        """Sinh câu trả lời theo batch. Hàm này dùng cho service knowledge_grounded_response_generator.

        Args:
            data (List[dict]): Danh sách các request data
                Mỗi request data là 1 dict:
                {
                    "query": câu hỏi,
                    "knowledge_retrieval": [Danh sách context],
                    "knowledge_search": [Danh sách context]
                }
                Mỗi context là 1 dict: {"passage_title": Tiêu đề context, "passage_content": Nội dung context}
            n_contexts (int, optional): Số contexts tối đa cho vào model. Defaults to 10.
            max_length (int, optional): Chiều dài tối đa của câu trả lời. Defaults to 64.
            threshold (float, optional): Ngưỡng chấp nhập một câu trả lời là đúng. Defaults to 0.5
            silent (bool, optional): Nếu giá trị là True, không in log thời gian xử lý của model. Defaults to True.

        Raises:
            ValueError: _description_

        Returns:
            List[str]: Danh sách các câu trả lời tương ứng cho từng request data
        """
        start_time = time.time()

        def _get_contexts(item):
            passage_items = item.get("knowledge_search", []) + item.get("knowledge_retrieval", [])
            return [self.preprocess_context_item(c) for c in passage_items[: n_contexts]]

        questions = [self.preprocess_text(item["query"]) for item in data]
        all_contexts = [_get_contexts(item) for item in data]
        inputs, context_map = self.dataset_obj.batch_get_inputs_for_generation(
            questions=questions,
            contexts=all_contexts,
        )
        model_outputs = self.model(input_ids=inputs["input_ids"].to(self.device),
                                   attention_mask=inputs["attention_mask"].to(self.device))
        start_scores, end_scores, has_answer_scores = self.post_process_model_outputs(model_outputs)
        answers = self.dataset_obj.decode_model_outputs(
            context_map=context_map,
            start_scores=start_scores,
            end_scores=end_scores,
            has_answer_scores=has_answer_scores,
            threshold=threshold,
        )
        if not silent:
            logger.info(
                f"{datetime.now()}   - GroundedKnowledgeResponseGenerator process {len(data)} (n_contexts={len(all_contexts[0])}) in {time.time()-start_time}s"
            )
        return answers

    def post_process_model_outputs(self, model_outputs) -> List[np.ndarray]:
        """Xử lý đầu ra của model và trả ra start scores, end scores và has answer scores."""
        start_scores = torch.softmax(model_outputs.start_logits, dim=-1).detach().cpu().numpy()
        end_scores = torch.softmax(model_outputs.end_logits, dim=-1).detach().cpu().numpy()
        if hasattr(model_outputs, "has_answer_logits"):
            has_answer_scores = torch.sigmoid(model_outputs.has_answer_logits).detach().cpu().numpy()
        else:
            has_answer_scores = np.ones(start_scores.shape[:1])
        return start_scores, end_scores, has_answer_scores

    def generate_answer(
        self,
        question: str,
        contexts: List[str],
    ) -> str:
        """Đọc các contexts và sinh ra các câu trả lời tương ứng cho câu hỏi.

        Args:
            question (str): Câu hỏi
            contexts (List[str]): Các câu context

        Returns:
            str: Câu trả lời
        """
        data = [{
            "query": question,
            "knowledge_retrieval": [{"passage_content": c, "passage_title": ""}for c in contexts]
        }]
        answer = self.generate_responses(
            data,
            n_contexts=len(contexts),
            silent=True
        )
        return answer[0]

    def preprocess_text(self, text: str):
        """Tiền xử lý dữ liệu trước khi cho vào model."""
        return self.dataset_obj.preprocess_text(text)

    def preprocess_context_item(self, context: dict):
        """Tiền xử lý context khi cho vào model.

        Context là 1 dict chứa 2 thông tin là passage_title và passage_content. Ví dụ:
                {
                    'passage_title': 'Đường Cao tốc Pháp Vân - Cầu Giẽ'
                    'passage_content': 'Cao tốc Mai Sơn - quốc lộ 45 gấp rút về đích cuối năm. 
                                    Dự án cao tốc Bắc Nam đoạn Mai Sơn - quốc lộ 45 gặp khó khăn do nhiều đoạn đất nền yếu,
                                    tuy nhiên nhà thầu cam kết về đích đúng hẹn cuối năm.'
                }
        """
        return {
            "passage_title": self.preprocess_text(context["passage_title"]),
            "passage_content": self.preprocess_text(context["passage_content"])
        }


def get_root_path():
    """Trả về đường dẫn đến thư mục gốc."""
    root_path = os.path.dirname(os.path.realpath(__file__))
    return root_path
