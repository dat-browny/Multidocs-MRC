from multi_document_mrc.mydatasets import QuestionAnsweringDataset
from multi_document_mrc.utils.read_data import SquadReader
from multi_document_mrc.utils.preprocess import normalize
from typing import Union, Optional, Tuple, List
from dataclasses import dataclass
from transformers import (
    PreTrainedTokenizerFast,
    PreTrainedTokenizer,
    EvalPrediction
)
from datasets import load_dataset
from pyvi import ViTokenizer
import numpy as np
import logging
import collections
from tqdm import tqdm
import os
import json


logger = logging.getLogger(__name__)


class MRCDatasetsForBERT(QuestionAnsweringDataset):
    use_wordsegment = False
    # SquadReader sử dụng khi đọc dữ liệu từ file json định dạng Squad
    # Cần chỉ định train_file, validation_file trong lúc run_train
    reader_class = SquadReader

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
        self.data_args = data_args
        self.tokenizer = tokenizer
        self.cache_dir = cache_dir
        self.do_train = do_train
        self.do_eval = do_eval
        self.do_predict = do_predict

        self.pad_on_right = tokenizer.padding_side == "right"
        if max_seq_length is not None:
            self.max_seq_length = max_seq_length
        elif data_args is not None:
            if data_args.max_seq_length > tokenizer.model_max_length:
                logger.warning(
                    f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                    f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
                )
            self.max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
        else:
            self.max_seq_length = tokenizer.model_max_length
        self.data_reader = self.reader_class(use_wordsegment=self.use_wordsegment,
                                             max_seq_length=self.max_seq_length)
        if data_args is not None:
            self.load_data()

    def load_data(self):
        # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
        # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
        # (the dataset will be downloaded automatically from the datasets Hub).
        #
        # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
        # 'text' is found. You can easily tweak this behavior (see below).
        #
        # In distributed training, the load_dataset function guarantee that only one local process can concurrently
        # download the dataset.
        if self.data_args.dataset_name is not None:
            # Downloading and loading a dataset from the hub.
            self.raw_datasets = load_dataset(
                self.data_args.dataset_name,
                self.data_args.dataset_config_name,
                cache_dir=self.cache_dir,
            )
        else:
            self.raw_datasets = self._load_data_from_file()

        # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
        # https://huggingface.co/docs/datasets/loading_datasets.html.
        # Preprocessing the datasets.
        # Preprocessing is slighlty different for training and evaluation.
        if self.do_train:
            column_names = self.raw_datasets["train"].column_names
        elif self.do_eval:
            column_names = self.raw_datasets["validation"].column_names
        else:
            column_names = self.raw_datasets["test"].column_names

        assert "question" in column_names, "Lỗi dataset không chứa cột question"
        self.question_column_name = "question"
        assert "context" in column_names, "Lỗi dataset không chứa cột context"
        self.context_column_name = "context"
        assert "answers" in column_names, "Lỗi dataset không chứa cột answers"
        self.answer_column_name = "answers"
        self.column_names = column_names
        return

    def _load_data_from_file(self):
        data_files = {}
        if self.do_train and self.data_args.train_file is not None:
            data_files["train"] = self.data_args.train_file
        if self.do_eval and self.data_args.validation_file is not None:
            data_files["validation"] = self.data_args.validation_file
        if self.do_predict and self.data_args.test_file is not None:
            data_files["test"] = self.data_args.test_file

        return self.data_reader.load_datasets(data_files)

    def get_train_dataset(self, main_process_first):
        train_dataset = None
        train_examples = None
        if self.do_train:
            if "train" not in self.raw_datasets:
                raise ValueError("--do_train requires a train dataset")
            train_examples = self.raw_datasets["train"]
            train_dataset = self.raw_datasets["train"]
            if self.data_args.max_train_samples is not None:
                # We will select sample from whole data if argument is specified
                max_train_samples = min(len(train_dataset), self.data_args.max_train_samples)
                train_dataset = train_dataset.select(range(max_train_samples))
            # Create train feature from dataset
            with main_process_first(desc="train dataset map pre-processing"):
                train_dataset = train_dataset.map(
                    self.prepare_train_features,
                    batched=True,
                    num_proc=self.data_args.preprocessing_num_workers,
                    remove_columns=self.column_names,
                    load_from_cache_file=not self.data_args.overwrite_cache,
                    desc="Running tokenizer on train dataset",
                )
            if self.data_args.max_train_samples is not None:
                # Number of samples might increase during Feature Creation, We select only specified max samples
                max_train_samples = min(len(train_dataset), self.data_args.max_train_samples)
                train_dataset = train_dataset.select(range(max_train_samples))
        return train_dataset, train_examples

    def get_eval_dataset(self, main_process_first):
        eval_dataset, eval_examples = None, None
        if self.do_eval:
            if "validation" not in self.raw_datasets:
                raise ValueError("--do_eval requires a validation dataset")
            eval_examples = self.raw_datasets["validation"]
            if self.data_args.max_eval_samples is not None:
                # We will select sample from whole data
                max_eval_samples = min(len(eval_examples), self.data_args.max_eval_samples)
                eval_examples = eval_examples.select(range(max_eval_samples))
            # Validation Feature Creation
            with main_process_first(desc="validation dataset map pre-processing"):
                eval_dataset = eval_examples.map(
                    self.prepare_validation_features,
                    batched=True,
                    num_proc=self.data_args.preprocessing_num_workers,
                    remove_columns=self.column_names,
                    load_from_cache_file=not self.data_args.overwrite_cache,
                    desc="Running tokenizer on validation dataset",
                )
            if self.data_args.max_eval_samples is not None:
                # During Feature creation dataset samples might increase, we will select required samples again
                max_eval_samples = min(len(eval_dataset), self.data_args.max_eval_samples)
                eval_dataset = eval_dataset.select(range(max_eval_samples))
        return eval_dataset, eval_examples

    def get_predict_dataset(self, main_process_first):
        predict_dataset, predict_examples = None, None
        if self.do_predict:
            if "test" not in self.raw_datasets:
                raise ValueError("--do_predict requires a test dataset")
            predict_examples = self.raw_datasets["test"]
            if self.data_args.max_predict_samples is not None:
                # We will select sample from whole data
                predict_examples = predict_examples.select(range(self.data_args.max_predict_samples))
            # Predict Feature Creation
            with main_process_first(desc="prediction dataset map pre-processing"):
                predict_dataset = predict_examples.map(
                    self.prepare_validation_features,
                    batched=True,
                    num_proc=self.data_args.preprocessing_num_workers,
                    remove_columns=self.column_names,
                    load_from_cache_file=not self.data_args.overwrite_cache,
                    desc="Running tokenizer on prediction dataset",
                )
            if self.data_args.max_predict_samples is not None:
                # During Feature creation dataset samples might increase, we will select required samples again
                max_predict_samples = min(len(predict_dataset), self.data_args.max_predict_samples)
                predict_dataset = predict_dataset.select(range(max_predict_samples))

        return predict_dataset, predict_examples

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
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

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
                {"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()
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
        if len(predictions) != 2:
            raise ValueError("`predictions` should be a tuple with two elements (start_logits, end_logits).")
        all_start_logits, all_end_logits = predictions

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
                predictions.insert(0, {"text": "empty", "start_logit": 0.0, "end_logit": 0.0, "score": 0.0})

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
                # Otherwise we first need to find the best non-empty prediction.
                i = 0
                while predictions[i]["text"] == "":
                    i += 1
                best_non_null_pred = predictions[i]

                # Then we compare to the null prediction using the threshold.
                score_diff = null_score - best_non_null_pred["start_logit"] - best_non_null_pred["end_logit"]
                scores_diff_json[example["id"]] = float(score_diff)  # To be JSON-serializable.
                if score_diff > null_score_diff_threshold:
                    all_predictions[example["id"]] = ""
                else:
                    all_predictions[example["id"]] = best_non_null_pred["text"]

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

    def preprocess_text(self, text: str):
        """Tiền xử lý dữ liệu trước khi cho vào model."""
        if self.use_wordsegment is not None:
            return ViTokenizer.tokenize(normalize(text))
        else:
            return normalize(text)

    def decode_model_outputs(
        self,
        context_map: list,
        start_scores: np.ndarray,
        end_scores: np.ndarray,
        has_answer_scores: np.ndarray,
        threshold: float = 0.5,
    ) -> List[str]:
        """Decode đầu ra của model và trả về câu trả lời."""
        answers = [[] for _ in range(context_map["n_sample"])]

        start_scores += context_map["non_context_mask"]
        end_scores += context_map["non_context_mask"]
        for cidx, (start_score, end_score) in enumerate(zip(start_scores, end_scores)):
            if has_answer_scores[cidx] < threshold:
                continue
            sidx, sscore = np.argmax(start_score), np.max(start_score)
            eidx, escore = np.argmax(end_score[sidx:]), np.max(end_score[sidx:])
            eidx = sidx + eidx
            score = (sscore+escore)/2

            eidx2, escore2 = np.argmax(end_score), np.max(end_score)
            sidx2, sscore2 = np.argmax(start_score[:eidx2]), np.max(start_score[:eidx2])
            score2 = (sscore2+escore2)/2
            if score < score2:
                sidx, eidx = sidx2, eidx2
                score = score2

            start_offset = context_map["offset_mapping"][cidx][sidx][0]
            end_offse = context_map["offset_mapping"][cidx][eidx][1]
            answers[context_map["contexts"][cidx]["sample"]].append({
                "text": context_map["contexts"][cidx]["text"][start_offset:end_offse].replace("_", " "),
                "confidence": (score+has_answer_scores[cidx])/2
            })

        def _best_answer(list_answer):
            """Tìm câu trả lời tốt nhất khi mô hình tìm được nhiều câu trả lời."""
            if len(list_answer) == 0:
                return {"text": "", "confidence": 1.0}
            return max(list_answer, key=lambda c: c["confidence"])
        return [_best_answer(list_answer) for list_answer in answers]

    def batch_get_inputs_for_generation(
        self,
        questions: List[str],
        contexts: List[List[dict]],
    ) -> List:
        """Chuyển đổi raw inputs thành tensor inputs cho model."""
        def _concat_title_context(passage_item):
            content = passage_item.get("passage_content")
            title = passage_item.get("passage_title")
            if passage_item.get("passage_title").strip() != "":
                return f"{title} . {content}"
            else:
                return content

        flatten_contexts = []
        context_idx = 0
        context_map = {
            "contexts": {},
            "n_sample": len(contexts),
        }
        new_questions = []
        for sample_idx, subcontexts in enumerate(contexts):
            for passage_item in subcontexts:
                context = _concat_title_context(passage_item)
                context_map["contexts"][context_idx] = {
                    "text": context,
                    "sample": sample_idx
                }
                flatten_contexts.append(context)
                new_questions.append(questions[sample_idx])
                context_idx += 1

        inputs = self.tokenizer(
            new_questions if self.pad_on_right else flatten_contexts,
            flatten_contexts if self.pad_on_right else new_questions,
            truncation="only_second" if self.pad_on_right else "only_first",
            padding="max_length",
            max_length=self.max_seq_length,
            return_offsets_mapping=True,
            return_tensors="pt"
        )

        offset_mapping = inputs.pop("offset_mapping")
        context_index = 1 if self.pad_on_right else 0
        context_map["non_context_mask"] = np.array([
            [0.0 if x == context_index else -10000.0 for x in inputs.sequence_ids(i)]
            for i in range(len(new_questions))
        ])
        context_map["offset_mapping"] = offset_mapping.detach().cpu().numpy().tolist()

        return inputs, context_map
