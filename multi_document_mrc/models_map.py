"""Map tên model đến các lớp model và dataset tương ứng."""
from multi_document_mrc.models.tokenization_phobert_fast import PhobertTokenizerFast
from transformers.modeling_utils import PreTrainedModel
from multi_document_mrc.mydatasets import QuestionAnsweringDataset
from multi_document_mrc.mydatasets.bert_datasets import MRCDatasetsForBERT
from multi_document_mrc.mydatasets.phobert_datasets import (
    ViMRCDatasetsForPhoBERTNoHap,
    ViMRCDatasetsForPhoBERT,
    ViMRCDatasetsV1bForPhoBERT,
    ViMRCDatasetsForPhoBERTNoHapReflection,
    ViMRCReflection
)
from multi_document_mrc.models.roberta_mrc import (
    RobertaForQuestionAnswering,
    RobertaForMRC
)
from multi_document_mrc.models.reflection_roberta_mrc import (
    RobertaForMRCReflection,
    ReflectionModel
)
from transformers import (
    RobertaConfig,
    PretrainedConfig,
    PreTrainedTokenizerFast,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    AutoConfig
)


class VersionClassesMap():
    def __init__(
        self,
        model_class: PreTrainedModel,
        dataset_class: QuestionAnsweringDataset,
        config_class: PretrainedConfig,
        tokenizer_class: PreTrainedTokenizerFast,
        description: str,
    ):
        self.model_class = model_class
        self.dataset_class = dataset_class
        self.config_class = config_class
        self.tokenizer_class = tokenizer_class
        self.description = description

    def __repr__(self) -> str:
        return self.description

    def __str__(self) -> str:
        return self.description


MODELS_MAP = {
    "huggingface-qa": VersionClassesMap(
        model_class=AutoModelForQuestionAnswering,
        dataset_class=MRCDatasetsForBERT,
        config_class=AutoConfig,
        tokenizer_class=AutoTokenizer,
        description="Huggingface Question Answering Auto Model"
    ),
    "phobert-qa-nohap":  VersionClassesMap(
        model_class=RobertaForQuestionAnswering,
        dataset_class=ViMRCDatasetsForPhoBERTNoHap,
        config_class=RobertaConfig,
        tokenizer_class=PhobertTokenizerFast,
        description="Phobert Question Answering model, No Has Answer Prediction Layer"
    ),
    "phobert-qa":  VersionClassesMap(
        model_class=RobertaForMRC,
        dataset_class=ViMRCDatasetsForPhoBERT,
        config_class=RobertaConfig,
        tokenizer_class=PhobertTokenizerFast,
        description="Phobert Question Answering model, Has a Answer Prediction Layer"
    ),
    "phobert-qa-v1b":  VersionClassesMap(
        model_class=RobertaForMRC,
        dataset_class=ViMRCDatasetsV1bForPhoBERT,
        config_class=RobertaConfig,
        tokenizer_class=PhobertTokenizerFast,
        description="Phobert Question Answering model, Has a Answer Prediction Layer, modify content title concatenation"
    ),
    "phobert-qa-mrc-block":VersionClassesMap(
        model_class=RobertaForMRCReflection,
        dataset_class=ViMRCDatasetsForPhoBERTNoHapReflection,
        config_class=RobertaConfig,
        tokenizer_class=PhobertTokenizerFast,
        description="Phobert Question Answering model, Has a Answer Prediction Layer, modify content title concatenation"
    ),
    "phobert-qa-reflection-block":VersionClassesMap(
        model_class=ReflectionModel,
        dataset_class=ViMRCReflection,
        config_class=RobertaConfig,
        tokenizer_class=PhobertTokenizerFast,
        description="Phobert Question Answering model, Has a Answer Prediction Layer, modify content title concatenation"
    ),
}
MODEL_VERSIONS = list(MODELS_MAP.keys())


def get_model_version_classes(model_architecture: str) -> VersionClassesMap:
    """Return VersionClassesMap Object."""
    if model_architecture not in MODELS_MAP:
        raise ValueError(f"model_architecture must be in {MODEL_VERSIONS}")
    else:
        return MODELS_MAP[model_architecture]
