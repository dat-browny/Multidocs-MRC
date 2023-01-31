#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for question answering using a slightly adapted version of the 🤗 Trainer.
"""
from metrics import compute_f1

import logging
import os
import sys
import datasets
import evaluate
import transformers
import torch
from multi_document_mrc.models.reflection_roberta_mrc import ReflectionModel
from multi_document_mrc.trainer import QuestionAnsweringTrainer
from transformers import (
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
import json

from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.versions import require_version
from multi_document_mrc.arguments import ModelArguments, DataTrainingArguments
from multi_document_mrc.models_map import get_model_version_classes

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/question-answering/requirements.txt")

logger = logging.getLogger(__name__)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model_architecture = get_model_version_classes(model_args.model_architecture)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    config = model_architecture.config_class.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    # add properties to config
    config.model_architecture = model_args.model_architecture

    tokenizer = model_architecture.tokenizer_class.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = model_architecture.model_class.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    model_reflection = ReflectionModel.from_pretrained(model_args.reflection_path, config=config)

    # Tokenizer check: this script requires a fast tokenizer.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models at"
            " https://huggingface.co/transformers/index.html#supported-frameworks to find the model types that meet"
            " this requirement"
        )

    # Datasets
    dataset_obj = model_architecture.dataset_class(
        tokenizer=tokenizer,
        data_args=data_args,
        cache_dir=model_args.cache_dir,
        do_train=training_args.do_train,
        do_eval=training_args.do_eval,
        do_predict=training_args.do_predict,
        reflection_path=model_args.reflection_path
    )

    train_dataset, train_examples = dataset_obj.get_train_dataset(
        main_process_first=training_args.main_process_first
    )
    eval_dataset, eval_examples = dataset_obj.get_eval_dataset(
        main_process_first=training_args.main_process_first
    )
    predict_dataset, predict_examples = dataset_obj.get_predict_dataset(
        main_process_first=training_args.main_process_first
    )

    # Data collator 
    # We have already padded to max length if the corresponding flag is True, otherwise we need to pad in the data
    # collator.
    data_collator = (
        default_data_collator
        if data_args.pad_to_max_length
        else DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)
    )

    metric = evaluate.load("squad_v2" if data_args.version_2_with_negative else "squad")

    def compute_metrics(p: EvalPrediction):
        with open(training_args.output_dir + 'prediction', 'w') as f:
            json.dump(p.predictions, f)
        if data_args.version_2_with_negative:
            predict_data = {}
            ids = []
            text = []
            predict_data['input_ids'] = []
            predict_data['head_feature'] = []
            predict_data['ans_type_ids'] = []
            formated_prediction = []

            for key, value in p.predictions.items():
                if len(value) != 4:
                    formated_prediction.append({"id": key, "prediction_text": value["text"], "no_answer_probability": value["na_prob"]})
                else:
                    ids.append(key)
                    text.append(value['text'])
                    predict_data['input_ids'].append(value['input_ids'])
                    predict_data['head_feature'].append(value['head_feature'])
                    predict_data['ans_type_ids'].append(value['ans_type_ids'])

            na_prob = []
            predict_data = datasets.Dataset.from_dict(predict_data)
            batch_data = DataLoader(predict_data.with_format("torch"), batch_size=16, shuffle=False)
            model_reflection.to(device)
            for batch in tqdm(batch_data):
                batch_na_probs = model_reflection(input_ids=batch['input_ids'].to(device),  head_features=batch['head_feature'].to(device), ans_type_ids=batch['ans_type_ids'].to(device))['ans_type_probs'].tolist()
                na_prob += batch_na_probs

            for id, prob in enumerate(na_prob):
                if prob > 0.5:
                    formated_prediction.append({"id": ids[id], "prediction_text": text[id], "no_answer_probability": 1-prob})
                else: 
                    formated_prediction.append({"id": ids[id], "prediction_text": "", "no_answer_probability": 1-prob})

            p.label_ids = sorted(p.label_ids,key=lambda x: x['id'])

            formated_prediction = sorted(formated_prediction, key=lambda x: x['id'])

            # remove if run actually
            # precision , recall = [], []
            # wrong = []
            # for id, sample in enumerate(formated_prediction):
            #     predicted_answer = sample['prediction_text']
            #     truth_answer = label_ids[id]['answers']['text']
            #     if len(truth_answer) == 0:
            #         if predicted_answer == "":
            #             precision.append(1)
            #             recall.append(1)
            #         else:
            #             precision.append(0)
            #             recall.append(0)
            #     else:
            #         score = [compute_f1(predicted_answer, answer) for answer in truth_answer]
            #         try:
            #             precision.append(max(score[:][0]))
            #             recall.append(max(score[:][1]))
            #         except:
            #             wrong.append([predicted_answer, truth_answer])

            
            # print(precision)
            # print(recall)
            # print(wrong)
            # print(sum(precision)/len(formated_prediction))
            # print(sum(recall)/len(formated_prediction))
            # print(formated_prediction[:10])
            # print(p.label_ids[:10])
            return metric.compute(predictions=formated_prediction, references=p.label_ids)

        return metric.compute(predictions=p.predictions, references=p.label_ids)
    # Initialize our Trainer
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        eval_examples=eval_examples if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=dataset_obj.post_processing_function,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Prediction
    if training_args.do_predict:
        logger.info("*** Predict ***")
        results = trainer.predict(predict_dataset, predict_examples)
        metrics = results.metrics

        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "question-answering"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name


if __name__ == "__main__":
    main()
