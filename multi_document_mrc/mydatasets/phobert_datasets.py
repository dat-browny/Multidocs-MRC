from multi_document_mrc.utils.read_data import SquadReader, SquadReaderV1b
from multi_document_mrc.mydatasets.bert_datasets import MRCDatasetsForBERT
from transformers import (
    RobertaConfig,
    PreTrainedTokenizerFast,
    PreTrainedTokenizer,
    EvalPrediction,
    PreTrainedModel
)
from multi_document_mrc.models.reflection_roberta_mrc import RobertaForMRCReflection, ReflectionModel
from dataclasses import dataclass
from typing import Union, Optional, Tuple, List
from datasets import Dataset
import logging
import torch
import numpy as np
import collections
from tqdm import tqdm
import os
import json

logger = logging.getLogger(__name__)
config = RobertaConfig.from_pretrained("vinai/phobert-base")


class ViMRCDatasetsForPhoBERTNoHap(MRCDatasetsForBERT):
    use_wordsegment = True
    reader_class = SquadReader


class ViMRCDatasetsForPhoBERT(ViMRCDatasetsForPhoBERTNoHap):

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
        offset_mapping = tokenized_examples.pop("offset_mapping")

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

        return tokenized_examples
        
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
            truncation="only_second" if self.pad_on_right else "only_first",
            max_length=self.max_seq_length,
            stride=self.data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if self.data_args.pad_to_max_length else False,
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if self.pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples

    def prepare_validation_features_reflection(self, examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        question_column_name = "question"
        context_column_name = "context"
        max_seq_length = 256
        pad_to_max_length = True
        doc_stride = 128
        pad_on_right = True
        examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = self.tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if pad_to_max_length else False,
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if self.pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples

    def post_processing_function(
        self,
        examples,
        features,
        predictions,
        output_dir,
        log_level,
        stage="eval"
    ):
        # Post-processing: we match the start logits and end logits to answers in the original context.
        predictions = self.postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            version_2_with_negative=self.data_args.version_2_with_negative,
            n_best_size=self.data_args.n_best_size,
            max_answer_length=self.data_args.max_answer_length,
            null_score_diff_threshold=self.data_args.null_score_diff_threshold,
            output_dir=output_dir,
            log_level=log_level,
            prefix=stage,
        )
        # Format the result to the format the metric expects.
        if self.data_args.version_2_with_negative:
            formatted_predictions = [
                {"id": k, "prediction_text": v["text"], "no_answer_probability": v["na_prob"]} for k, v in predictions.items()
            ]
        else:
            formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]

        references = [{"id": ex["id"], "answers": ex[self.answer_column_name]} for ex in examples]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)

    @staticmethod
    def postprocess_qa_predictions(
        examples,
        features,
        predictions: Tuple[np.ndarray, np.ndarray],
        version_2_with_negative: bool = False,
        n_best_size: int = 20,
        max_answer_length: int = 30,
        null_score_diff_threshold: float = 0.0,
        output_dir: Optional[str] = None,
        prefix: Optional[str] = None,
        log_level: Optional[int] = logging.WARNING,
    ):
        """
        Post-processes the predictions of a question-answering model to convert them to answers that are substrings of the
        original contexts. This is the base postprocessing functions for models that only return start and end logits.

        Args:
            examples: The non-preprocessed dataset (see the main script for more information).
            features: The processed dataset (see the main script for more information).
            predictions (:obj:`Tuple[np.ndarray, np.ndarray]`):
                The predictions of the model: two arrays containing the start logits and the end logits respectively. Its
                first dimension must match the number of elements of :obj:`features`.
            version_2_with_negative (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not the underlying dataset contains examples with no answers.
            n_best_size (:obj:`int`, `optional`, defaults to 20):
                The total number of n-best predictions to generate when looking for an answer.
            max_answer_length (:obj:`int`, `optional`, defaults to 30):
                The maximum length of an answer that can be generated. This is needed because the start and end predictions
                are not conditioned on one another.
            null_score_diff_threshold (:obj:`float`, `optional`, defaults to 0):
                The threshold used to select the null answer: if the best answer has a score that is less than the score of
                the null answer minus this threshold, the null answer is selected for this example (note that the score of
                the null answer for an example giving several features is the minimum of the scores for the null answer on
                each feature: all features must be aligned on the fact they `want` to predict a null answer).

                Only useful when :obj:`version_2_with_negative` is :obj:`True`.
            output_dir (:obj:`str`, `optional`):
                If provided, the dictionaries of predictions, n_best predictions (with their scores and logits) and, if
                :obj:`version_2_with_negative=True`, the dictionary of the scores differences between best and null
                answers, are saved in `output_dir`.
            prefix (:obj:`str`, `optional`):
                If provided, the dictionaries mentioned above are saved with `prefix` added to their names.
            log_level (:obj:`int`, `optional`, defaults to ``logging.WARNING``):
                ``logging`` log level (e.g., ``logging.WARNING``)
        """
        if len(predictions) != 3:
            raise ValueError(
                "`predictions` should be a tuple with two elements (start_logits, end_logits, has_answer_logits).")
        all_start_logits, all_end_logits, has_answer_logits = predictions
        has_answer_probs = 1/(1 + np.exp(-has_answer_logits))
        no_answer_probs = 1 - has_answer_probs

        if len(predictions[0]) != len(features):
            raise ValueError(f"Got {len(predictions[0])} predictions and {len(features)} features.")

        # Build a map example to its corresponding features.
        example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
        features_per_example = collections.defaultdict(list)
        for i, feature in enumerate(features):
            features_per_example[example_id_to_index[feature["example_id"]]].append(i)

        # The dictionaries we have to fill.
        all_predictions = collections.OrderedDict()
        all_nbest_json = collections.OrderedDict()
        if version_2_with_negative:
            scores_diff_json = collections.OrderedDict()

        # Logging.
        logger.setLevel(log_level)
        logger.info(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

        # Let's loop over all the examples!
        for example_index, example in enumerate(tqdm(examples)):
            # Those are the indices of the features associated to the current example.
            feature_indices = features_per_example[example_index]

            min_null_prediction = None
            prelim_predictions = []

            # Looping through all the features associated to the current example.
            for feature_index in feature_indices:
                # We grab the predictions of the model for this feature.
                start_logits = all_start_logits[feature_index]
                end_logits = all_end_logits[feature_index]
                na_prob = float(no_answer_probs[feature_index])
                # This is what will allow us to map some the positions in our logits to span of texts in the original
                # context.
                offset_mapping = features[feature_index]["offset_mapping"]
                # Optional `token_is_max_context`, if provided we will remove answers that do not have the maximum context
                # available in the current feature.
                token_is_max_context = features[feature_index].get("token_is_max_context", None)

                # Update minimum null prediction.
                feature_null_score = start_logits[0] + end_logits[0]
                if min_null_prediction is None or min_null_prediction["score"] > feature_null_score:
                    min_null_prediction = {
                        "offsets": (0, 0),
                        "score": feature_null_score,
                        "start_logit": start_logits[0],
                        "end_logit": end_logits[0],
                        "na_prob": na_prob,
                    }

                # Go through all possibilities for the `n_best_size` greater start and end logits.
                start_indexes = np.argsort(start_logits)[-1: -n_best_size - 1: -1].tolist()
                end_indexes = np.argsort(end_logits)[-1: -n_best_size - 1: -1].tolist()
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                        # to part of the input_ids that are not in the context.
                        if (
                            start_index >= len(offset_mapping)
                            or end_index >= len(offset_mapping)
                            or offset_mapping[start_index] is None
                            or len(offset_mapping[start_index]) < 2
                            or offset_mapping[end_index] is None
                            or len(offset_mapping[end_index]) < 2
                        ):
                            continue
                        # Don't consider answers with a length that is either < 0 or > max_answer_length.
                        if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                            continue
                        # Don't consider answer that don't have the maximum context available (if such information is
                        # provided).
                        if token_is_max_context is not None and not token_is_max_context.get(str(start_index), False):
                            continue

                        prelim_predictions.append(
                            {
                                "offsets": (offset_mapping[start_index][0], offset_mapping[end_index][1]),
                                "score": start_logits[start_index] + end_logits[end_index],
                                "start_logit": start_logits[start_index],
                                "end_logit": end_logits[end_index],
                                "na_prob": na_prob,
                            }
                        )
            if version_2_with_negative and min_null_prediction is not None:
                # Add the minimum null prediction
                prelim_predictions.append(min_null_prediction)
                null_score = min_null_prediction["score"]

            # Only keep the best `n_best_size` predictions.
            predictions = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[:n_best_size]

            # Add back the minimum null prediction if it was removed because of its low score.
            if (
                version_2_with_negative
                and min_null_prediction is not None
                and not any(p["offsets"] == (0, 0) for p in predictions)
            ):
                predictions.append(min_null_prediction)

            # Use the offsets to gather the answer text in the original context.
            context = example["context"]
            for pred in predictions:
                offsets = pred.pop("offsets")
                pred["text"] = context[offsets[0]: offsets[1]]

            # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
            # failure.
            if len(predictions) == 0 or (len(predictions) == 1 and predictions[0]["text"] == ""):
                predictions.insert(0, {"text": "empty", "start_logit": 0.0,
                                   "end_logit": 0.0, "score": 0.0, "na_prob": 0.0})

            # Compute the softmax of all scores (we do it with numpy to stay independent from torch/tf in this file, using
            # the LogSumExp trick).
            scores = np.array([pred.pop("score") for pred in predictions])
            exp_scores = np.exp(scores - np.max(scores))
            probs = exp_scores / exp_scores.sum()

            # Include the probabilities in our predictions.
            for prob, pred in zip(probs, predictions):
                pred["probability"] = prob

            # Pick the best prediction. If the null answer is not possible, this is easy.
            if not version_2_with_negative:
                all_predictions[example["id"]] = predictions[0]["text"]
            else:
                NO_ANSWER_THRESHOLD = 0.5
                predictions = [p for p in predictions if p["na_prob"] < NO_ANSWER_THRESHOLD or p["text"] == ""]
                # Otherwise we first need to find the best non-empty prediction.
                for i in range(len(predictions)):
                    if predictions[i]["text"] != "":
                        break
                best_non_null_pred = predictions[i]

                # Then we compare to the null prediction using the threshold.
                score_diff = null_score - best_non_null_pred["start_logit"] - best_non_null_pred["end_logit"]
                scores_diff_json[example["id"]] = float(score_diff)  # To be JSON-serializable.
                if score_diff > null_score_diff_threshold:
                    all_predictions[example["id"]] = {"text": "", "na_prob": 1.0}
                else:
                    all_predictions[example["id"]] = {
                        "text": best_non_null_pred["text"],
                        "na_prob": best_non_null_pred["na_prob"]
                    }

            # Make `predictions` JSON-serializable by casting np.float back to float.
            all_nbest_json[example["id"]] = [
                {k: (float(v) if isinstance(v, (np.float16, np.float32, np.float64)) else v) for k, v in pred.items()}
                for pred in predictions
            ]

        # If we have an output_dir, let's save all those dicts.
        if output_dir is not None:
            if not os.path.isdir(output_dir):
                raise EnvironmentError(f"{output_dir} is not a directory.")

            prediction_file = os.path.join(
                output_dir, "predictions.json" if prefix is None else f"{prefix}_predictions.json"
            )
            nbest_file = os.path.join(
                output_dir, "nbest_predictions.json" if prefix is None else f"{prefix}_nbest_predictions.json"
            )
            if version_2_with_negative:
                null_odds_file = os.path.join(
                    output_dir, "null_odds.json" if prefix is None else f"{prefix}_null_odds.json"
                )

            logger.info(f"Saving predictions to {prediction_file}.")
            with open(prediction_file, "w") as writer:
                writer.write(json.dumps(all_predictions, indent=4) + "\n")
            logger.info(f"Saving nbest_preds to {nbest_file}.")
            with open(nbest_file, "w") as writer:
                writer.write(json.dumps(all_nbest_json, indent=4) + "\n")
            if version_2_with_negative:
                logger.info(f"Saving null_odds to {null_odds_file}.")
                with open(null_odds_file, "w") as writer:
                    writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

        return all_predictions


class ViMRCDatasetsV1bForPhoBERT(ViMRCDatasetsForPhoBERT):
    use_wordsegment = True
    reader_class = SquadReaderV1b

    def __init__(
        self,
        tokenizer: Union[PreTrainedTokenizerFast, PreTrainedTokenizer],
        data_args: Optional[dataclass] = None,
        cache_dir: Optional[str] = None,
        max_seq_length: Optional[int] = None,
        do_train: bool = False,
        do_eval: bool = False,
        do_predict: bool = False,
        **kwargs,
    ):
        super().__init__(
            tokenizer=tokenizer,
            data_args=data_args,
            cache_dir=cache_dir,
            max_seq_length=max_seq_length,
            do_train=do_train,
            do_eval=do_eval,
            do_predict=do_predict,
            **kwargs
        )
        if data_args is not None:
            assert "title" in self.column_names, "Lỗi dataset không chứa cột answers"
            self.title_column_name = "title"

    def prepare_train_features(self, examples):
        prefix_contexts = [
            f"{question} {self.tokenizer.sep_token} Tiêu_đề : {title} . Nội_dung :"
            for question, title in zip(examples[self.question_column_name], examples[self.title_column_name])
        ]
        contexts = [
            f"{prefix_s} {context}"
            for prefix_s, context in zip(prefix_contexts, examples[self.context_column_name])
        ]

        tokenized_examples = self.tokenizer(
            contexts,
            truncation=True,
            max_length=self.max_seq_length,
            return_offsets_mapping=True,
            padding="max_length" if self.data_args.pad_to_max_length else False,
        )

        prefix_lens = [len(s)+1 for s in prefix_contexts]
        prefix_ids_lens = [len(self.tokenizer.tokenize(s))+1 for s in prefix_contexts]
        prefix_mask = [
            [-10000] * x + [0] * (self.max_seq_length-x)
            for x in prefix_ids_lens
        ]
        tokenized_examples["prefix_mask"] = prefix_mask
        tokenized_examples["new_context"] = contexts

        offset_mapping = tokenized_examples.pop("offset_mapping")

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []
        tokenized_examples["has_answer_labels"] = []

        for sample_index, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][sample_index]
            cls_index = input_ids.index(self.tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(sample_index)

            # One example can give several spans, this is the index of the example containing this span of text.
            answers = examples[self.answer_column_name][sample_index]
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
                # has_answer_labels==0 tương ứng với câu hỏi không có câu trả lời, ngược lại
                tokenized_examples["has_answer_labels"].append(0)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0] + prefix_lens[sample_index]
                end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text.
                token_start_index = prefix_ids_lens[sample_index]
                while sequence_ids[token_start_index] != 0:
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != 0:
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

        return tokenized_examples

    def prepare_validation_features(self, examples):
        prefix_contexts = [
            f"{question} {self.tokenizer.sep_token} Tiêu_đề : {title} . Nội_dung :"
            for question, title in zip(examples[self.question_column_name], examples[self.title_column_name])
        ]
        contexts = [
            f"{prefix_s} {context}"
            for prefix_s, context in zip(prefix_contexts, examples[self.context_column_name])
        ]

        tokenized_examples = self.tokenizer(
            contexts,
            truncation=True,
            max_length=self.max_seq_length,
            return_offsets_mapping=True,
            padding="max_length" if self.data_args.pad_to_max_length else False,
        )

        prefix_ids_lens = [len(self.tokenizer.tokenize(s))+1 for s in prefix_contexts]
        prefix_mask = [
            [-10000] * x + [0] * (self.max_seq_length-x)
            for x in prefix_ids_lens
        ]
        tokenized_examples["prefix_mask"] = prefix_mask
        tokenized_examples["new_context"] = contexts
        tokenized_examples["example_id"] = examples["id"]

        return tokenized_examples

    @staticmethod
    def postprocess_qa_predictions(
        examples,
        features,
        predictions: Tuple[np.ndarray, np.ndarray],
        version_2_with_negative: bool = False,
        n_best_size: int = 20,
        max_answer_length: int = 30,
        null_score_diff_threshold: float = 0.0,
        output_dir: Optional[str] = None,
        prefix: Optional[str] = None,
        log_level: Optional[int] = logging.WARNING,
    ):
        if len(predictions) != 3:
            raise ValueError(
                "`predictions` should be a tuple with two elements (start_logits, end_logits, has_answer_logits).")
        all_start_logits, all_end_logits, has_answer_logits = predictions
        prefix_mask = np.array(features["prefix_mask"], dtype=all_start_logits.dtype)
        all_start_logits += prefix_mask
        all_end_logits += prefix_mask

        has_answer_probs = 1/(1 + np.exp(-has_answer_logits))
        no_answer_probs = 1 - has_answer_probs

        if len(predictions[0]) != len(features):
            raise ValueError(f"Got {len(predictions[0])} predictions and {len(features)} features.")

        # The dictionaries we have to fill.
        all_predictions = collections.OrderedDict()
        all_nbest_json = collections.OrderedDict()
        if version_2_with_negative:
            scores_diff_json = collections.OrderedDict()

        # Logging.
        logger.setLevel(log_level)
        logger.info(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

        # Let's loop over all the examples!
        for feature_index, example in enumerate(tqdm(examples)):
            # Those are the indices of the features associated to the current example.

            prelim_predictions = []

            # We grab the predictions of the model for this feature.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            na_prob = float(no_answer_probs[feature_index])
            # This is what will allow us to map some the positions in our logits to span of texts in the original
            # context.
            offset_mapping = features[feature_index]["offset_mapping"]
            # Optional `token_is_max_context`, if provided we will remove answers that do not have the maximum context
            # available in the current feature.
            token_is_max_context = features[feature_index].get("token_is_max_context", None)

            # null prediction.
            null_prediction = {
                "offsets": (0, 0),
                "score": start_logits[0] + end_logits[0],
                "start_logit": start_logits[0],
                "end_logit": end_logits[0],
                "na_prob": na_prob,
            }

            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = np.argsort(start_logits)[-1: -n_best_size - 1: -1].tolist()
            end_indexes = np.argsort(end_logits)[-1: -n_best_size - 1: -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                    ):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue
                    # Don't consider answer that don't have the maximum context available (if such information is
                    # provided).
                    if token_is_max_context is not None and not token_is_max_context.get(str(start_index), False):
                        continue

                    prelim_predictions.append(
                        {
                            "offsets": (offset_mapping[start_index][0], offset_mapping[end_index][1]),
                            "score": start_logits[start_index] + end_logits[end_index],
                            "start_logit": start_logits[start_index],
                            "end_logit": end_logits[end_index],
                            "na_prob": na_prob,
                        }
                    )

            if version_2_with_negative:
                # Add the minimum null prediction
                prelim_predictions.append(null_prediction)
                null_score = null_prediction["score"]

            # Only keep the best `n_best_size` predictions.
            predictions = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[:n_best_size]

            # Add back the minimum null prediction if it was removed because of its low score.
            if (
                version_2_with_negative
                and not any(p["offsets"] == (0, 0) for p in predictions)
            ):
                predictions.append(null_prediction)

            # Use the offsets to gather the answer text in the original context.
            context = features[feature_index]["new_context"]
            for pred in predictions:
                offsets = pred.pop("offsets")
                pred["text"] = context[offsets[0]: offsets[1]]

            # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
            # failure.
            if len(predictions) == 0 or (len(predictions) == 1 and predictions[0]["text"] == ""):
                predictions.insert(0, {"text": "empty", "start_logit": 0.0,
                                   "end_logit": 0.0, "score": 0.0, "na_prob": 0.0})

            # Compute the softmax of all scores (we do it with numpy to stay independent from torch/tf in this file, using
            # the LogSumExp trick).
            scores = np.array([pred.pop("score") for pred in predictions])
            exp_scores = np.exp(scores - np.max(scores))
            probs = exp_scores / exp_scores.sum()

            # Include the probabilities in our predictions.
            for prob, pred in zip(probs, predictions):
                pred["probability"] = prob

            # Pick the best prediction. If the null answer is not possible, this is easy.
            if not version_2_with_negative:
                all_predictions[example["id"]] = predictions[0]["text"]
            else:
                NO_ANSWER_THRESHOLD = 0.5
                predictions = [p for p in predictions if p["na_prob"] < NO_ANSWER_THRESHOLD or p["text"] == ""]
                # Otherwise we first need to find the best non-empty prediction.
                for i in range(len(predictions)):
                    if predictions[i]["text"] != "":
                        break
                best_non_null_pred = predictions[i]

                # Then we compare to the null prediction using the threshold.
                score_diff = null_score - best_non_null_pred["start_logit"] - best_non_null_pred["end_logit"]
                scores_diff_json[example["id"]] = float(score_diff)  # To be JSON-serializable.
                if score_diff > null_score_diff_threshold:
                    all_predictions[example["id"]] = {"text": "", "na_prob": 1.0}
                else:
                    all_predictions[example["id"]] = {
                        "text": best_non_null_pred["text"],
                        "na_prob": best_non_null_pred["na_prob"]
                    }

            # Make `predictions` JSON-serializable by casting np.float back to float.
            all_nbest_json[example["id"]] = [
                {k: (float(v) if isinstance(v, (np.float16, np.float32, np.float64)) else v) for k, v in pred.items()}
                for pred in predictions
            ]

        # If we have an output_dir, let's save all those dicts.
        if output_dir is not None:
            if not os.path.isdir(output_dir):
                raise EnvironmentError(f"{output_dir} is not a directory.")

            prediction_file = os.path.join(
                output_dir, "predictions.json" if prefix is None else f"{prefix}_predictions.json"
            )
            nbest_file = os.path.join(
                output_dir, "nbest_predictions.json" if prefix is None else f"{prefix}_nbest_predictions.json"
            )
            if version_2_with_negative:
                null_odds_file = os.path.join(
                    output_dir, "null_odds.json" if prefix is None else f"{prefix}_null_odds.json"
                )

            logger.info(f"Saving predictions to {prediction_file}.")
            with open(prediction_file, "w") as writer:
                writer.write(json.dumps(all_predictions, indent=4) + "\n")
            logger.info(f"Saving nbest_preds to {nbest_file}.")
            with open(nbest_file, "w") as writer:
                writer.write(json.dumps(all_nbest_json, indent=4) + "\n")
            if version_2_with_negative:
                logger.info(f"Saving null_odds to {null_odds_file}.")
                with open(null_odds_file, "w") as writer:
                    writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

        return all_predictions

    def batch_get_inputs_for_generation(
        self,
        questions: List[str],
        contexts: List[List[dict]],
    ) -> List:
        """Chuyển đổi raw inputs thành tensor inputs cho model."""
        def _get_prefix_context(question: str, title: str):
            if title != "":
                return f"{question} {self.tokenizer.sep_token} Tiêu_đề : {title} . Nội_dung :"
            else:
                return f"{question} {self.tokenizer.sep_token}"

        def _concat_title_context(question: str, passage_item: dict) -> str:
            content = passage_item.get("passage_content")
            title = passage_item.get("passage_title")
            prefix_s = _get_prefix_context(question, title)
            return (f"{prefix_s} {content}", prefix_s)

        contexts = [
            [_concat_title_context(question, passage_item) for passage_item in subcontexts]
            for question, subcontexts in zip(questions, contexts)
        ]

        flatten_contexts = []
        prefix_lens = []
        context_idx = 0
        context_map = {
            "contexts": {},
            "n_sample": len(contexts),
        }
        for sample_idx, subcontexts in enumerate(contexts):
            for context, prefix_s in subcontexts:
                context_map["contexts"][context_idx] = {
                    "text": context,
                    "sample": sample_idx
                }
                flatten_contexts.append(context)
                prefix_lens.append(len(self.tokenizer.tokenize(prefix_s))+1)
                context_idx += 1

        inputs = self.tokenizer(
            flatten_contexts,
            truncation=True,
            padding="max_length",
            max_length=self.max_seq_length,
            return_offsets_mapping=True,
            return_tensors="pt"
        )

        offset_mapping = inputs.pop("offset_mapping")
        context_map["non_context_mask"] = np.array([
            [
                -10000.0 if (x is None or i < pl) else 0.0
                for i, x in enumerate(inputs.sequence_ids(sample_idx))
            ]
            for sample_idx, pl in enumerate(prefix_lens)
        ])
        context_map["offset_mapping"] = offset_mapping.detach().cpu().numpy()

        return inputs, context_map

class ViMRCDatasetsForPhoBERTNoHapReflection(ViMRCDatasetsForPhoBERT):
    def __init__(self, tokenizer: Union[PreTrainedTokenizerFast, PreTrainedTokenizer], data_args: Optional[dataclass] = None, cache_dir: Optional[str] = None, max_seq_length: Optional[int] = None, do_train: bool = False, do_eval: bool = False, reflection_path: str = None, do_predict: bool = False, is_training_reflection: bool = False, **kwargs):
        super().__init__(tokenizer, data_args, cache_dir, max_seq_length, do_train, do_eval, do_predict, **kwargs)
        if do_eval:
            self.model = ReflectionModel.from_pretrained(reflection_path, config=config)
            self.is_training_reflection = is_training_reflection
    # def post_processing_function(self, examples, features, predictions, output_dir, log_level, stage="eval"):
    @staticmethod
    def postprocess_qa_predictions(
        examples,
        features,
        predictions: Tuple[np.ndarray, np.ndarray],
        version_2_with_negative: bool = False,
        n_best_size: int = 20,
        max_answer_length: int = 30,
        log_level: Optional[int] = logging.WARNING,
        model: PreTrainedModel = None,
        is_training_reflection = True
    ):

        if len(predictions) != 5:
            raise ValueError(
                "`predictions` should be a tuple with five elements (start_logits, end_logits, has_answer_logits, score, head_features).")
        all_start_logits, all_end_logits, has_answer_probs, scores, head_features = predictions
        no_answer_probs = has_answer_probs[:,0]

        if len(predictions[0]) != len(features):
            raise ValueError(f"Got {len(predictions[0])} predictions and {len(features)} features.")
        
        example_id_to_index = {k: i for i, k in enumerate(examples["id"])}

        features_per_example = collections.defaultdict(list)
        for i, feature in enumerate(features):
            features_per_example[example_id_to_index[feature["example_id"]]].append(i)

        all_predictions = collections.OrderedDict()
        all_nbest_json = collections.OrderedDict()
        if version_2_with_negative:
            scores_diff_json = collections.OrderedDict()

        logger.setLevel(log_level)
        logger.info(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

        for example_index, example in enumerate(tqdm(examples)):

            n = len(all_predictions)
            # Those are the indices of the features associated to the current example.
            feature_indices = features_per_example[example_index]

            feature_index_with_best_score = collections.UserList([index, scores[index]] for index in feature_indices)

            feature_index = sorted(feature_index_with_best_score, key=lambda x: x[1], reverse=True)[0][0]

            prelim_predictions = []
            min_null_prediction = None
            # Looping through all the features associated to the current example.
            # for feature_index in feature_indices:
                # We grab the predictions of the model for this feature.

            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            na_prob = float(no_answer_probs[feature_index])

            feature_null_score = start_logits[0] + end_logits[0]
            if min_null_prediction is None or min_null_prediction["score"] > feature_null_score:
                min_null_prediction = {
                    "offsets": (0, 0),
                    "score": feature_null_score,
                    "start_logit": start_logits[0],
                    "end_logit": end_logits[0],
                    "na_prob": na_prob,
                }
            
            head_feature = head_features[feature_index]
            offset_mapping = features[feature_index]["offset_mapping"]
            token_is_max_context = features[feature_index].get("token_is_max_context", None)

            if is_training_reflection:
                start_indexes = torch.argsort(start_logits)[-n_best_size:: 1].tolist()
                end_indexes = torch.argsort(end_logits)[-n_best_size:: 1].tolist()
            else: 
                start_indexes = np.argsort(start_logits)[-n_best_size:: 1].tolist()
                end_indexes = np.argsort(end_logits)[-n_best_size:: 1].tolist()

            start_indexes.reverse()
            end_indexes.reverse()
                
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or len(offset_mapping[start_index]) < 2
                        or offset_mapping[end_index] is None
                        or len(offset_mapping[end_index]) < 2
                    ):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue
                    # Don't consider answer that don't have the maximum context available (if such information is
                    # provided).
                    if token_is_max_context is not None and not token_is_max_context.get(str(start_index), False):
                        continue

                    prelim_predictions.append(
                        {
                            "offsets": (offset_mapping[start_index][0], offset_mapping[end_index][1]),
                            "score": start_logits[start_index] + end_logits[end_index],
                            "start_logit": start_logits[start_index],
                            "end_logit": end_logits[end_index],
                            "feature_index": feature_index,
                            "head_features": head_feature,
                            "start_index": start_index,
                            "end_index": end_index
                        }
                    )        


            if version_2_with_negative and min_null_prediction is not None:
                # Add the minimum null prediction
                prelim_predictions.append(min_null_prediction)
                null_score = min_null_prediction["score"]

            # Only keep the best `n_best_size` predictions.
            predictions = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[:n_best_size]

            # Add back the minimum null prediction if it was removed because of its low score.
            if (
                version_2_with_negative
                and min_null_prediction is not None
                and not any(p["offsets"] == (0, 0) for p in predictions)
            ):
                predictions.append(min_null_prediction)

            # Use the offsets to gather the answer text in the original context.

            context = example["context"]
            for pred in predictions:
                offsets = pred.pop("offsets")
                pred["text"] = context[offsets[0]: offsets[1]]

            # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
            # failure.
            if len(predictions) == 0 or (len(predictions) == 1 and predictions[0]["text"] == "") and is_training_reflection:
                all_predictions[example["id"]] = {
                            "start_positions": 0,
                            "end_positions": 0,
                            "head_features": head_feature,
                            "feature_index": feature_index
                        }
            elif len(predictions) == 0 or (len(predictions) == 1 and predictions[0]["text"] == "") and not is_training_reflection:
                all_predictions[example["id"]] = {"text": "empty", "start_logit": 0.0,
                                   "end_logit": 0.0, "score": 0.0, "na_prob": 0.0}
            else:
            # Pick the best prediction. If the null answer is not possible, this is easy.
                if not version_2_with_negative:
                    all_predictions[example["id"]] = predictions[0]["text"]
                else:
                    # Otherwise we first need to find the best non-empty prediction.
                    for i in range(len(predictions)):
                        if predictions[i]["text"] != "":
                            break
                    best_non_null_pred = predictions[i]
                    head_feature =  best_non_null_pred['head_features']
                    feature_index =  best_non_null_pred['feature_index']

                    if is_training_reflection:
                        all_predictions[example["id"]] = {
                            "start_positions": best_non_null_pred['start_index'],
                            "end_positions": best_non_null_pred['end_index'],
                            "head_features": head_feature,
                            "feature_index": feature_index
                        }
                    else:
                        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
                        start_index = best_non_null_pred['start_index']
                        end_index = best_non_null_pred['end_index']

                        input_ids = torch.tensor([features[feature_index]['input_ids']], device=device)
                        head_feature=torch.tensor([head_feature], device=device)
                        ans_type_ids = torch.tensor([[0]*len(input_ids[0])], device=device)
                        if no_answer_probs[feature_index] < 0.5:
                            ans_type_ids[0][0] = 2
                        else:
                            ans_type_ids[0][0] = 1
                        ans_type_ids[0][start_index] = 3
                        ans_type_ids[0][start_index+1:end_index+1] = 4
                        if input_ids.device == ans_type_ids.device == head_feature.device:
                            print(111111111111111111111111111111111111111111111111111111111111111111111111)
                        na_probs_ = model(input_ids=input_ids, 
                                          ans_type_ids=ans_type_ids, 
                                          head_features=head_feature,
                                          return_dict=True)['ans_type_probs']

                        # Then we compare to the null prediction using the threshold.

                        if na_probs_ > 0.5:
                            all_predictions[example["id"]] = {"text": "", "na_prob": 1.0}
                        else:
                            all_predictions[example["id"]] = {
                                "text": best_non_null_pred["text"],
                                "na_prob": na_probs_
                            }

            # Make `predictions` JSON-serializable by casting np.float back to float.
            all_nbest_json[example["id"]] = [
                {k: (float(v) if isinstance(v, (np.float16, np.float32, np.float64)) else v) for k, v in pred.items()}
                for pred in predictions
            ]

        return all_predictions

    def post_processing_function(
        self,
        examples,
        features,
        predictions,
        output_dir,
        log_level,
        stage="eval",
    ): 

        # Post-processing: we match the start logits and end logits to answers in the original context.
        predictions = self.postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            version_2_with_negative=self.data_args.version_2_with_negative,
            n_best_size=self.data_args.n_best_size,
            max_answer_length=self.data_args.max_answer_length,
            log_level=log_level,
            model=self.model,
            is_training_reflection=self.is_training_reflection
        )

        # Format the result to the format the metric expects.

        if self.data_args.version_2_with_negative:
            formatted_predictions = [
                {"id": k, "prediction_text": v["text"], "no_answer_probability": v["na_prob"]} for k, v in predictions.items()
            ]
        else:
            formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]

        references = [{"id": ex["id"], "answers": ex[self.answer_column_name]} for ex in examples]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)

class ViMRCReflection(ViMRCDatasetsForPhoBERTNoHap):

    # def __init__(self, tokenizer: Union[PreTrainedTokenizerFast, PreTrainedTokenizer], data_args:  Optional[dataclass] = None, cache_dir: Optional[str] = None, max_seq_length: Optional[int] = None, do_train: bool = False, do_eval: bool = False, do_predict: bool = False, **kwargs):
    #     super().__init__(tokenizer, data_args, cache_dir, max_seq_length, do_train, do_eval, do_predict, **kwargs)
    #     self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    #     self.model_name_or_path = model_name_or_path
    #     self.model = model
    #     self.MRCModel = model.to(self.device)
    # def __init__(self, tokenizer: Union[PreTrainedTokenizerFast, PreTrainedTokenizer], model_name_or_path: str = None, data_args:  Optional[dataclass] = None, cache_dir: Optional[str] = None, max_seq_length: Optional[int] = None, do_train: bool = False, do_eval: bool = False, do_predict: bool = False, **kwargs):
    #     super().__init__(tokenizer, data_args, cache_dir, max_seq_length, do_train, do_eval, do_predict, **kwargs)
    #     self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    #     self.model_name_or_path = model_name_or_path
    #     self.MRCModel = RobertaForMRCReflection.from_pretrained(self.model_name_or_path, config=config).to(self.device)
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
        
        return tokenized_examples
    
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
        
        return tokenized_examples