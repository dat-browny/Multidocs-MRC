# import os
# import sys
import torch
# import logging
# import json
# import datasets
# import evaluate
# import transformers
# from tqdm import tqdm
# from transformers import (
#     RobertaConfig,
#     HfArgumentParser,
#     PreTrainedTokenizerFast,
#     TrainingArguments,
#     DataCollatorWithPadding,
#     EvalPrediction,
#     default_data_collator,
#     set_seed,
# )
# from multi_document_mrc.mydatasets.phobert_datasets import (
#     ViMRCReflection
# )
# from torch.utils.data.dataloader import default_collate
# from torch.utils.data import DataLoader
# from transformers.trainer_utils import get_last_checkpoint
# from multi_document_mrc.models.tokenization_phobert_fast import PhobertTokenizerFast
# from multi_document_mrc.arguments import ModelArguments, DataTrainingArguments
# from multi_document_mrc.models_map import get_model_version_classes
# from dataclasses import dataclass
# from multi_document_mrc.models.reflection_roberta_mrc import RobertaForMRCReflection
# from multi_document_mrc.mydatasets.phobert_datasets import ViMRCDatasetsForPhoBERT, ViMRCDatasetsForPhoBERTNoHapReflection
# from multi_document_mrc.trainer import ReflectionTrainer

# from transformers.trainer_utils import get_last_checkpoint
# from transformers.utils.versions import require_version
# from multi_document_mrc.arguments import ModelArguments, DataTrainingArguments
# from multi_document_mrc.models_map import get_model_version_classes

import logging
import os
import sys
# import datasets
import evaluate
import transformers
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
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.versions import require_version
from multi_document_mrc.arguments import ModelArguments, DataTrainingArguments
from multi_document_mrc.models_map import get_model_version_classes

logger = logging.getLogger(__name__)

def convert_to_instance(model, tokenizer, examples, tokenized_data, device, batch_size, model_name_or_path, max_seq_length):

    tokenized_data_dict = tokenized_data.to_dict()
    for k, v in tokenized_data_dict.items():
        tokenized_data_dict[k] = torch.tensor(v, device=device)

    infer_data = DataLoader(tokenized_data.with_format("torch"), batch_size=batch_size)
    start_logits = []
    end_logits = []
    has_answer_probs = []
    score = []
    head_features = []

    with torch.no_grad():
        for batch in tqdm(infer_data):
            output = model(input_ids=batch['input_ids'].to(device), 
                                    start_positions=batch['start_positions'].to(device), 
                                    end_positions=batch['end_positions'].to(device), 
                                    has_answer_labels=batch['has_answer_labels'].to(device), 
                                    return_dict=True)

            start_logits += output['start_logits'].tolist()
            end_logits += output['end_logits'].tolist()
            has_answer_probs += output['has_answer_probs'].tolist()
            score += output['score'].tolist()
            head_features += output['head_features'].tolist()
            
    predictions = tuple(torch.tensor(i) for i in (start_logits, end_logits, has_answer_probs, score, head_features))
    # x = datasets.Dataset.from_dict(dict(examples))

    features = examples.map(ViMRCDatasetsForPhoBERT(tokenizer).prepare_validation_features_reflection,
                    batched=True,
                    remove_columns=examples.features)

    instance_training = ViMRCDatasetsForPhoBERTNoHapReflection(tokenizer, model_name_or_path=model_name_or_path).postprocess_qa_predictions(examples=examples, 
                    features=features, 
                    predictions=predictions,
                    version_2_with_negative=True,
                    is_training_reflection=True)

    start_positions = [value['start_positions'] for key, value in instance_training.items()]
    end_positions = [value['end_positions'] for key, value in instance_training.items()]
    head_features = [value['head_features'] for key, value in instance_training.items()]
    feature_index = [value['feature_index'] for key, value in instance_training.items()]

    tokenized_examples_ = {}
    tokenized_examples_['input_ids'] = []
    tokenized_examples_['ans_type_ids'] = []
    tokenized_examples_['has_answer_labels'] = []
    tokenized_examples_['attention_mask'] = []
    tokenized_examples_['head_features'] = []

    for id, feature_slice in tqdm(enumerate(feature_index)):
        tokenized_examples_['input_ids'].append(tokenized_data_dict['input_ids'][feature_slice].tolist())
        tokenized_examples_['has_answer_labels'].append(tokenized_data_dict['has_answer_labels'][feature_slice].tolist())
        tokenized_examples_['attention_mask'].append(tokenized_data_dict['attention_mask'][feature_slice].tolist())
        tokenized_examples_['head_features'].append(head_features[id].tolist())
        start_position = start_positions[id]
        end_position = end_positions[id]
        ans_type_id = [0]* max_seq_length
        if tokenized_examples_['has_answer_labels'][-1] == 1 and start_position<end_position:
            ans_type_id[0] = 2
        else:
            ans_type_id[0] = 1
        if start_position < end_position:
            ans_type_id[start_position] = 3
            for i in range(start_position+1, end_position+1):
                ans_type_id[i] = 4
        tokenized_examples_['ans_type_ids'].append(ans_type_id)

    return tokenized_examples_

# def save_datasets(datasets, dir):
#     if not os.path.isdir(dir):
#         os.mkdir(dir)
#     dataset_name_root = dir.split("/")[-1] 
#     if len(datasets) == 1:
#         if datasets is not None:
#             dataset_name += '.json'
#             path = os.path.join(dir, dataset_name)
#             with open(path, 'w') as fp:
#                 json.dump(datasets, fp)
#         else: 
#             logger.warn("For step Training Reflection, Training dataset must required, please add train_file, and do_train arguments")
#     else:
#         for id, dataset in enumerate(datasets):
#             if id == 0:
#                 dataset_name = dataset_name_root + '_datasets.json'
#                 if dataset is not None:
#                     path = os.path.join(dir, dataset_name_root)
#                     with open(path, 'w') as fp:
#                         json.dump(datasets, fp)
#                 else: 
#                     logger.warn(f"For step {dataset_name_root} Reflection, {dataset_name_root} dataset must required")          
#             else: 
#                 dataset_name = dataset_name_root + '_datasets.json'
#                 if dataset is not None:
#                     path = os.path.join(dir, dataset_name_root)
#                     with open(path, 'w') as fp:
#                         json.dump(datasets, fp)
#                 else: 
#                     logger.warn(f"For step {dataset_name_root} Reflection, {dataset_name_root} dataset must required")     
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

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    model_architecture = get_model_version_classes(model_args.model_architecture)

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

    config = RobertaConfig.from_pretrained(
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

    model_ = RobertaForMRCReflection.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # Tokenizer check: this script requires a fast tokenizer.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models at"
            " https://huggingface.co/transformers/index.html#supported-frameworks to find the model types that meet"
            " this requirement"
        )

    # Datasets
    dataset_obj = ViMRCReflection(
        tokenizer=tokenizer,
        data_args=data_args,
        cache_dir=model_args.cache_dir,
        do_train=training_args.do_train,
        do_eval=training_args.do_eval,
        do_predict=training_args.do_predict,
        model=model,
        model_name_or_path=model_args.model_name_or_path
    )
    
    train_dataset, train_examples = dataset_obj.get_train_dataset(
        main_process_first=training_args.main_process_first
    )

    eval_dataset, eval_examples = dataset_obj.get_eval_dataset(
        main_process_first=training_args.main_process_first
    )

    predict_dataset, predict_examples = dataset_obj.get_predict_dataset(
        main_process_first=training_args.main_process_first)

    model_.to(device)

    if training_args.do_train:
        train_dataset = convert_to_instance(model=model_, 
                                        tokenizer=tokenizer, 
                                        examples=train_examples, 
                                        tokenized_data=train_dataset, 
                                        device=device, batch_size=32, 
                                        model_name_or_path=model_args.model_name_or_path, 
                                        max_seq_length=data_args.max_seq_length)
    if training_args.do_eval: 
        eval_dataset = convert_to_instance(model=model_, 
                                        tokenizer=tokenizer, 
                                        examples=eval_examples, 
                                        tokenized_data=eval_dataset, 
                                        device=device, batch_size=32, 
                                        model_name_or_path=model_args.model_name_or_path, 
                                        max_seq_length=data_args.max_seq_length)
     
    if training_args.do_predict: 
        predict_dataset = convert_to_instance(model=model_, 
                                        tokenizer=tokenizer, 
                                        examples=predict_examples, 
                                        tokenized_data=predict_dataset, 
                                        device=device, batch_size=32, 
                                        model_name_or_path=model_args.model_name_or_path, 
                                        max_seq_length=data_args.max_seq_length)

    dataset_name = ["train", "eval", "predict"]
    dataset = [train_dataset,  (eval_dataset, eval_examples), (predict_dataset, predict_examples)]

    for id, name in enumerate(dataset_name):
        path = os.path.join(training_args.output_dir, name)
        save_datasets(dataset[id], path)
        logger.info(f'Saving {name} dataset for step Reflection at {path}')

    # data_collator = (
    #     default_data_collator
    #     if data_args.pad_to_max_length
    #     else DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)
    # )

    # metric = evaluate.load("f1")

    # def compute_metrics(p: EvalPrediction):
    #     return metric.compute(predictions=p.predictions, references=p.label_ids)

    # # Initialize our Trainer
    # trainer = ReflectionTrainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_dataset if training_args.do_train else None,
    #     eval_dataset=eval_dataset if training_args.do_eval else None,
    #     eval_examples=eval_examples if training_args.do_eval else None,
    #     tokenizer=tokenizer,
    #     data_collator=data_collator,
    #     post_process_function=dataset_obj.post_processing_function,
    #     compute_metrics=compute_metrics
    # )

    # # Training
    # if training_args.do_train:
    #     checkpoint = None
    #     if training_args.resume_from_checkpoint is not None:
    #         checkpoint = training_args.resume_from_checkpoint
    #     elif last_checkpoint is not None:
    #         checkpoint = last_checkpoint
    #     train_result = trainer.train(resume_from_checkpoint=checkpoint)
    #     trainer.save_model()  # Saves the tokenizer too for easy upload

    #     metrics = train_result.metrics
    #     max_train_samples = (
    #         data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
    #     )
    #     metrics["train_samples"] = min(max_train_samples, len(train_dataset))

    #     trainer.log_metrics("train", metrics)
    #     trainer.save_metrics("train", metrics)
    #     trainer.save_state()

    # # Evaluation
    # if training_args.do_eval:
    #     logger.info("*** Evaluate ***")
    #     metrics = trainer.evaluate()

    #     max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
    #     metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

    #     trainer.log_metrics("eval", metrics)
    #     trainer.save_metrics("eval", metrics)

    # # Prediction
    # if training_args.do_predict:
    #     logger.info("*** Predict ***")
    #     results = trainer.predict(predict_dataset, predict_examples)
    #     metrics = results.metrics

    #     max_predict_samples = (
    #         data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
    #     )
    #     metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

    #     trainer.log_metrics("predict", metrics)
    #     trainer.save_metrics("predict", metrics)

    # kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "question-answering"}
    # if data_args.dataset_name is not None:
    #     kwargs["dataset_tags"] = data_args.dataset_name
    #     if data_args.dataset_config_name is not None:
    #         kwargs["dataset_args"] = data_args.dataset_config_name
    #         kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
    #     else:
    #         kwargs["dataset"] = data_args.dataset_name


if __name__ == "__main__":
    main()
