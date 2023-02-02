from sklearn.metrics import classification_report
from transformers import (
    RobertaConfig,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    TrainingArguments,
    set_seed,
)
from multi_document_mrc.mydatasets.phobert_datasets import (
    ViMRCReflection
)
from torch.utils.data import DataLoader
from multi_document_mrc.arguments import ModelArguments, DataTrainingArguments
from multi_document_mrc.models_map import get_model_version_classes
from multi_document_mrc.models.reflection_roberta_mrc import RobertaForMRCReflection
from multi_document_mrc.mydatasets.phobert_datasets import ViMRCDatasetsForPhoBERT, ViMRCDatasetsForPhoBERTNoHapReflection
import os
import sys
import torch
import logging
import transformers
import datasets
from tqdm import tqdm
import json
import evaluate
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
from multi_document_mrc.trainer import QuestionAnsweringTrainer


logger = logging.getLogger(__name__)

def convert_to_instance(trainer, tokenizer, examples, tokenized_data, model_name_or_path, max_seq_length):

    tokenized_data_dict = tokenized_data.to_dict()
    tokenized_data = tokenized_data.map(remove_columns=['offset_mapping', 'has_answer_labels'])
    # 'start_positions', 'end_positions',
    predictions = trainer.inference(dataset=tokenized_data)

    features = examples.map(ViMRCDatasetsForPhoBERT(tokenizer).prepare_validation_features_reflection,
                    batched=True,
                    remove_columns=examples.features)

    instance_training = ViMRCDatasetsForPhoBERTNoHapReflection(tokenizer, model_name_or_path=model_name_or_path).postprocess_qa_predictions(examples=examples, 
                    features=features, 
                    predictions=predictions[1:],
                    version_2_with_negative=True,
                    is_training_reflection=True)

    start_positions = [value['start_positions'] for key, value in instance_training.items()]
    end_positions = [value['end_positions'] for key, value in instance_training.items()]
    head_features = [value['head_features'] for key, value in instance_training.items()]
    feature_index = [value['feature_index'] for key, value in instance_training.items()]
    na_probs = [value['na_probs'] for key, value in instance_training.items()]

    tokenized_examples_ = {}
    tokenized_examples_['input_ids'] = []
    tokenized_examples_['ans_type_ids'] = []
    tokenized_examples_['has_answer_labels'] = []
    tokenized_examples_['attention_mask'] = []
    tokenized_examples_['head_features'] = []

    for id, feature_slice in tqdm(enumerate(feature_index)):
        tokenized_examples_['input_ids'].append(tokenized_data_dict['input_ids'][feature_slice])
        tokenized_examples_['has_answer_labels'].append(tokenized_data_dict['has_answer_labels'][feature_slice])
        tokenized_examples_['attention_mask'].append(tokenized_data_dict['attention_mask'][feature_slice])
        tokenized_examples_['head_features'].append(head_features[id].tolist())
        start_position = start_positions[id]
        end_position = end_positions[id]
        ans_type_id = [0]* max_seq_length
        if na_probs[id] > 0.5 and start_position < end_position:
            ans_type_id[0] = 2
        else:
            ans_type_id[0] = 1
        if start_position < end_position:
            ans_type_id[start_position] = 3
            for i in range(start_position+1, end_position+1):
                ans_type_id[i] = 4
        tokenized_examples_['ans_type_ids'].append(ans_type_id)

    return tokenized_examples_
        
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
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    model_architecture = get_model_version_classes(model_args.model_architecture)

    log_level = training_args.get_process_log_level()
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

    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

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
    
    data_collator = (
        default_data_collator
        if data_args.pad_to_max_length
        else DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)
    )

    # model_.to(device) 

    trainer = QuestionAnsweringTrainer(
        model=model_,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        eval_examples=eval_examples if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=dataset_obj.post_processing_function
    )

    if training_args.do_train:
        train_dataset = convert_to_instance(trainer=trainer, 
                                        tokenizer=tokenizer, 
                                        examples=train_examples, 
                                        tokenized_data=train_dataset, 
                                        model_name_or_path=model_args.model_name_or_path, 
                                        max_seq_length=data_args.max_seq_length)

        train_path = os.path.join(training_args.output_dir, "train")
        if not os.path.isdir(train_path):
            os.mkdir(train_path)
        with open(os.path.join(train_path, "train_dataset.json"), 'w') as fp:
            json.dump(train_dataset, fp)
            fp.close()

    if training_args.do_eval: 
        eval_dataset = convert_to_instance(trainer=trainer, 
                                        tokenizer=tokenizer, 
                                        examples=eval_examples, 
                                        tokenized_data=eval_dataset, 
                                        model_name_or_path=model_args.model_name_or_path, 
                                        max_seq_length=data_args.max_seq_length)

        eval_path = os.path.join(training_args.output_dir, "eval")
        if not os.path.isdir(eval_path):
            os.mkdir(eval_path)
        with open(os.path.join(eval_path, "eval_dataset.json"), 'w') as fp:
            json.dump(eval_dataset, fp)
            fp.close()
        with open(os.path.join(eval_path, "eval_examples.json"), 'w') as fp:
            json.dump(eval_examples.to_dict(), fp)
            fp.close()

    if training_args.do_predict: 
        predict_dataset = convert_to_instance(trainer=trainer, 
                                        tokenizer=tokenizer, 
                                        examples=predict_examples, 
                                        tokenized_data=predict_dataset, 
                                        model_name_or_path=model_args.model_name_or_path, 
                                        max_seq_length=data_args.max_seq_length)

        predict_path = os.path.join(training_args.output_dir, "predict")
        if not os.path.isdir(predict_path):
            os.mkdir(predict_path)
        with open(os.path.join(predict_path, "predict_dataset.json"), 'w') as fp:
            json.dump(predict_dataset, fp)
            fp.close()
        with open(os.path.join(predict_path, "predict_examples.json"), 'w') as fp:
            json.dump(predict_examples.to_dict(), fp)
            fp.close()

if __name__ == "__main__":
    main()
