from multi_document_mrc.mydatasets import ViMRCDatasetsForPhoBERTNoHap, ViMRCDatasetsForPhoBERT, ViMRCDatasetsForPhoBERTNoHapReflection
from datasets import Dataset
from multi_document_mrc.models.reflection_roberta_mrc import RobertaForMRCReflection
from typing import Union, Optional
from transformers import (
    RobertaConfig,
    PreTrainedTokenizerFast,
    PreTrainedTokenizer,
)
from dataclasses import dataclass
import torch

config = RobertaConfig.from_pretrained("vinai/phobert-base")

class ViMRCReflection(ViMRCDatasetsForPhoBERTNoHap):

    def __init__(self, tokenizer: Union[PreTrainedTokenizerFast, PreTrainedTokenizer], model_name_or_path: str = None, data_args:  Optional[dataclass] = None, cache_dir: Optional[str] = None, max_seq_length: Optional[int] = None, do_train: bool = False, do_eval: bool = False, do_predict: bool = False, **kwargs):
        super().__init__(tokenizer, data_args, cache_dir, max_seq_length, do_train, do_eval, do_predict, **kwargs)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model_name_or_path = model_name_or_path
        self.MRCModel = RobertaForMRCReflection.from_pretrained(self.model_name_or_path, config=config).to(self.device)

    def prepare_train_features(self, examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples[self.question_column_name] = [q.lstrip() for q in examples[self.question_column_name]]
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = self.tokenizer(
            examples[self.question_column_name if self.pad_on_right else self.context_column_name],
            examples[self.context_column_name if self.pad_on_right else self.question_column_name],
            truncation="only_second",
            max_length=self.max_seq_length,
            stride=self.data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if self.data_args.pad_to_max_length else False,
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples["offset_mapping"]

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []
        tokenized_examples["has_answer_labels"] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(self.tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples[self.answer_column_name][sample_index]
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
                # has_answer_labels==0 tương ứng với câu hỏi không có câu trả lời, ngược lại
                tokenized_examples["has_answer_labels"].append(0)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if self.pad_on_right else 0):
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if self.pad_on_right else 0):
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                    tokenized_examples["has_answer_labels"].append(0)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)
                    tokenized_examples["has_answer_labels"].append(1)
        
        for k, v in tokenized_examples.items():
            tokenized_examples[k] = torch.tensor(v, device=self.device)
        with torch.no_grad(): 
        #Dungf postprocess cua model MRC de gen instance training cho model nay
            predictions = self.MRCModel(input_ids=tokenized_examples['input_ids'], 
                                start_positions=tokenized_examples['start_positions'], 
                                end_positions=tokenized_examples['end_positions'], 
                                has_answer_labels=tokenized_examples['has_answer_labels'], 
                                return_dict=True)

        predictions = (predictions['start_logits'],predictions['end_logits'],predictions['has_answer_probs'],predictions['score'],predictions['head_features'])

        x = Dataset.from_dict(dict(examples))
        features = x.map(ViMRCDatasetsForPhoBERT(self.tokenizer).prepare_validation_features_reflection,
                        batched=True,
                        remove_columns=x.features)

        instance_training = ViMRCDatasetsForPhoBERTNoHapReflection(self.tokenizer, model_name_or_path=self.model_name_or_path).postprocess_qa_predictions(examples=x, 
                            features=features, 
                            predictions=predictions,
                            version_2_with_negative=True,
                            is_training_reflection=True)

        head_features = [value['head_features'] for key, value in instance_training.items()]
        feature_index = [value['feature_index'] for key, value in instance_training.items()]
        tokenized_examples['has_answer_labels'] = tokenized_examples['has_answer_labels'].tolist()
        tokenized_examples_ = {}
        tokenized_examples_['input_ids'] = []
        tokenized_examples_['ans_type_ids'] = []
        tokenized_examples_['has_answer_labels'] = []
        tokenized_examples_['attention_mask'] = []
        tokenized_examples_['head_features'] = []
        for id, feature_slice in enumerate(feature_index):
            tokenized_examples_['input_ids'].append(tokenized_examples['input_ids'][feature_slice])
            tokenized_examples_['has_answer_labels'].append(tokenized_examples['has_answer_labels'][feature_slice])
            tokenized_examples_['attention_mask'].append(tokenized_examples['attention_mask'][feature_slice])
            tokenized_examples_['head_features'].append(head_features[id])
            start_position = tokenized_examples['start_positions'][feature_slice]
            end_position = tokenized_examples['end_positions'][feature_slice]
            ans_type_id = torch.tensor([0]*self.max_seq_length)
            if tokenized_examples_['has_answer_labels'][-1] == 0:
                ans_type_id[0] = 1
            else:
                ans_type_id[0] = 2
            if start_position < end_position:
                ans_type_id[start_position] = 3
                ans_type_id[start_position+1:end_position+1] = 4
            tokenized_examples_['ans_type_ids'].append(ans_type_id)
        
        return tokenized_examples_
    
    def prepare_validation_features(self, examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples[self.question_column_name] = [q.lstrip() for q in examples[self.question_column_name]]
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = self.tokenizer(
            examples[self.question_column_name if self.pad_on_right else self.context_column_name],
            examples[self.context_column_name if self.pad_on_right else self.question_column_name],
            truncation="only_second",
            max_length=self.max_seq_length,
            stride=self.data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if self.data_args.pad_to_max_length else False,
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples["offset_mapping"]

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []
        tokenized_examples["has_answer_labels"] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(self.tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples[self.answer_column_name][sample_index]
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
                # has_answer_labels==0 tương ứng với câu hỏi không có câu trả lời, ngược lại
                tokenized_examples["has_answer_labels"].append(0)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if self.pad_on_right else 0):
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if self.pad_on_right else 0):
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                    tokenized_examples["has_answer_labels"].append(0)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)
                    tokenized_examples["has_answer_labels"].append(1)
        
        for k, v in tokenized_examples.items():
            tokenized_examples[k] = torch.tensor(v, device=self.device)
        with torch.no_grad(): 

            predictions = self.MRCModel(input_ids=tokenized_examples['input_ids'], 
                                start_positions=tokenized_examples['start_positions'], 
                                end_positions=tokenized_examples['end_positions'], 
                                has_answer_labels=tokenized_examples['has_answer_labels'], 
                                return_dict=True)

        predictions = (predictions['start_logits'],predictions['end_logits'],predictions['has_answer_probs'],predictions['score'],predictions['head_features'])

        x = Dataset.from_dict(dict(examples))
        features = x.map(ViMRCDatasetsForPhoBERT(self.tokenizer).prepare_validation_features_reflection,
                        batched=True,
                        remove_columns=x.features)
        instance_training = ViMRCDatasetsForPhoBERTNoHapReflection(self.tokenizer, model_name_or_path=self.model_name_or_path).postprocess_qa_predictions(examples=x, 
                            features=features, 
                            predictions=predictions,
                            version_2_with_negative=True,
                            is_training_reflection=True)

        head_features = [value['head_features'] for key, value in instance_training.items()]
        feature_index = [value['feature_index'] for key, value in instance_training.items()]

        tokenized_examples['has_answer_labels'] = tokenized_examples['has_answer_labels'].tolist()

        tokenized_examples_ = {}
        tokenized_examples_['input_ids'] = []
        tokenized_examples_['ans_type_ids'] = []
        tokenized_examples_['has_answer_labels'] = []
        tokenized_examples_['attention_mask'] = []
        tokenized_examples_['head_features'] = []

        for id, feature_slice in enumerate(feature_index):
            tokenized_examples_['input_ids'].append(tokenized_examples['input_ids'][feature_slice])
            tokenized_examples_['has_answer_labels'].append(tokenized_examples['has_answer_labels'][feature_slice])
            tokenized_examples_['attention_mask'].append(tokenized_examples['attention_mask'][feature_slice])
            tokenized_examples_['head_features'].append(head_features[id])
            start_position = tokenized_examples['start_positions'][feature_slice]
            end_position = tokenized_examples['end_positions'][feature_slice]
            ans_type_id = torch.tensor([0]*self.max_seq_length)
            if tokenized_examples_['has_answer_labels'][-1] == 0:
                ans_type_id[0] = 1
            else:
                ans_type_id[0] = 2
            if start_position < end_position:
                ans_type_id[start_position] = 3
                ans_type_id[start_position+1:end_position+1] = 4
            tokenized_examples_['ans_type_ids'].append(ans_type_id)
        
        return tokenized_examples_

def main():
    
