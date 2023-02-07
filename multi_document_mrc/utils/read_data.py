from multi_document_mrc.utils.preprocess import normalize, sylabelize, sentence_tokenize
from datasets.dataset_dict import DatasetDict, Dataset
from typing import Dict, List
from pyvi import ViTokenizer
import pandas as pd
import json
import tqdm
import os


class SquadReader():
    def __init__(self, use_wordsegment: bool = False, max_seq_length: int = 256):
        self.use_wordsegment = use_wordsegment
        self.max_seq_length = max_seq_length

    def load_datasets(self, data_files: Dict[(str, str)]) -> DatasetDict:
        datasets = DatasetDict()
        for key, data_file_path in data_files.items():
            if os.path.isfile(data_file_path):
                data = self._read(data_file_path)
                datasets[key] = Dataset.from_pandas(pd.DataFrame(data))
        return datasets

    def _read(self, data_file_path: str) -> List[dict]:
        print(f"Đọc file dữ liệu: {data_file_path}")
        raw_data = self._read_squad_format(data_file_path)
        for sample in raw_data:
            for qa in sample["qas"]:
                self._redefine_answer_char_start(sample["context"], qa)
            sample["qas"] = [qa for qa in sample["qas"] if self._check_true_index_answer(sample["context"], qa)]

        data = [self.tokenize_and_convert_sample(sample) for sample in tqdm.tqdm(raw_data)]
        dict_data = self.convert_dict_data(data)

        print("     - Total qa: " + str(len(dict_data["id"])))
        print("     - Negative qa: "+str(len([x for x in dict_data["answers"] if len(x["text"]) == 0])))
        return dict_data

    @staticmethod
    def convert_dict_data(data):
        dict_data = {
            "id": [],
            "title": [],
            "context": [],
            "question": [],
            "answers": []
        }
        idx = 0
        for sample in data:
            for qa in sample["qas"]:
                dict_data["title"].append(sample["title"])
                dict_data["context"].append(sample["context"])
                dict_data["question"].append(qa["question"])
                if not qa["is_positive"]:
                    dict_data["answers"].append({
                        "text": [],
                        "answer_start": []
                    })
                else:
                    dict_data["answers"].append({
                        "text": [answer["text"] for answer in qa["answers"]],
                        "answer_start": [answer["answer_start"] for answer in qa["answers"]],
                    })
                dict_data["id"].append(f"squad-{idx}")
                idx += 1
        return dict_data

    @staticmethod
    def _check_true_index_answer(context: str, qa: dict) -> bool:
        if not qa["is_positive"]:
            return True

        for i, answer in enumerate(qa["answers"]):
            if answer["text"] == context[answer["answer_start"]:answer["answer_start"]+len(answer["text"])]:
                continue
            else:
                qa["answers"][i] = None
        qa["answers"] = [answer for answer in qa["answers"] if answer is not None]
        if len(qa["answers"]) == 0:
            return False
        else:
            return True

    def _preprocess_text(self, text: str) -> str:
        """Tiền xử lý dữ liệu text."""
        text = text.replace("–", "-")
        if self.use_wordsegment:
            return normalize(ViTokenizer.tokenize(text))
        else:
            return text
            # return normalize(sylabelize(text))

    @staticmethod
    def _read_squad_format(data_file_path: str) -> List[dict]:
        with open(data_file_path, "r") as file:
            raw_data = json.load(file)["data"]
        data = []
        for page in raw_data:
            for paragraph in page["paragraphs"]:
                context = paragraph["context"]
                qas = [
                    {
                        "question": qa["question"],
                        "answers": [] if qa.get("is_impossible", False) else qa["answers"],
                        "is_positive": not qa.get("is_impossible", False),
                        "id": qa["id"],
                    }
                    for qa in paragraph["qas"]
                ]
                data.append({
                    "title": page["title"],
                    "context": context,
                    "qas": qas
                })
        return data

    @staticmethod
    def _redefine_answer_char_start(context: str, qa: dict) -> dict:
        """Xác định lại answert start.

        Args:
            context (str): Đoạn context chứa câu trả lời
            qa (dict): Dict chứa câu hỏi và câu trả lời
        """
        if not qa["is_positive"]:
            return

        context2 = context.replace("_", " ")
        for answer in qa["answers"]:
            answer_text = answer["text"].replace("_", " ")

            len_answer = len(answer_text)
            start = answer["answer_start"]

            if context2[start:start+len_answer] == answer_text:
                answer["text"] = context[start:start+len_answer].strip()
                continue

            for i in range(max(start-20, 0), start+20):
                if context2[i:i+len_answer] == answer_text:
                    answer["answer_start"] = i
                    answer["text"] = context[i:i+len_answer].strip()
                    break
        return

    def tokenize_and_convert_sample(self, sample: dict) -> dict:
        title = self._preprocess_text(sample["title"])
        context = sample["context"]
        negative_qas = [qa for qa in sample["qas"] if not qa["is_positive"]]
        qas = [qa for qa in sample["qas"] if qa["is_positive"]]

        for qa in negative_qas:
            qa["question"] = self._preprocess_text(qa["question"])

        # Word segment context và map lại index của câu trả lời với context mới
        new_context = self._preprocess_text(context)
        try:
            mapidx = MapNewIndex(context, new_context)

            for qa in qas:
                qa["question"] = self._preprocess_text(qa["question"])
                for answer in qa["answers"]:
                    answer["text"] = self._preprocess_text(answer["text"])
                    answer["answer_start"] = mapidx.map_new_char_index(answer["answer_start"])
                self._redefine_answer_char_start(new_context, qa)

            qas = [qa for qa in qas if self._check_true_index_answer(new_context, qa)]

            return {
                "title": title,
                "context": new_context,
                "qas": qas+negative_qas
            }
        except:
            return {
                "title": title,
                "context": new_context,
                "qas": []
            }


class MapNewIndex():
    def __init__(self, context: str, new_context: str):
        oldtext_2_ziptext = {}
        zipidx = 0
        for oldidx, c in enumerate(context):
            oldtext_2_ziptext[oldidx] = zipidx
            if c != " " and c != "_":
                zipidx += 1

        ziptext_2_newtext = {}
        zipidx = 0
        for newidx, c in enumerate(new_context):
            ziptext_2_newtext[zipidx] = newidx
            if c != " " and c != "_":
                zipidx += 1

        self._oldtext_2_ziptext = oldtext_2_ziptext
        self._ziptext_2_newtext = ziptext_2_newtext

        assert len(self.ziptext(new_context)) == len(self.ziptext(context))

    @staticmethod
    def ziptext(text):
        return "".join([c for c in text if c != " " and c != "_"])

    def map_new_char_index(self, oldidx: int) -> int:
        zipidx = self._oldtext_2_ziptext[oldidx]
        newidx = self._ziptext_2_newtext[zipidx]
        return newidx


class SquadReaderV1b(SquadReader):
    def _read(self, data_file_path: str) -> List[dict]:
        print(f"Đọc file dữ liệu: {data_file_path}")
        raw_data = self._read_squad_format(data_file_path)
        for sample in raw_data:
            for qa in sample["qas"]:
                self._redefine_answer_char_start(sample["context"], qa)
            sample["qas"] = [qa for qa in sample["qas"] if self._check_true_index_answer(sample["context"], qa)]

        data = [self.tokenize_and_convert_sample(sample) for sample in tqdm.tqdm(raw_data)]
        data = self.split_long_data(data, max_len=200)
        dict_data = self.convert_dict_data(data)

        print("     - Total qa: " + str(len(dict_data["id"])))
        print("     - Negative qa: "+str(len([x for x in dict_data["answers"] if len(x["text"]) == 0])))
        return dict_data

    @staticmethod
    def split_context(title, context, max_len):
        max_len -= len(title.split())
        sentences = sentence_tokenize(context)
        assert len(context) == sum([len(s) + 1 for s in sentences]) - 1
        all_context = []
        index = 0
        cur_len = 0
        cur_context = []
        num_sentences = 0
        for line in sentences:
            num_words = len(line.split())
            cur_len += num_words
            cur_context.append(line)
            num_sentences += 1
            if cur_len >= max_len:
                cur_context_s = " ".join(cur_context)
                all_context.append({
                    "title": title,
                    "context": cur_context_s,
                    "qas": [],
                    "start": index,
                    "end": index + len(cur_context_s)
                })

                index = index + len(cur_context_s) - len(line)
                cur_len = len(line)
                cur_context = [line]
                num_sentences = 1

        if num_sentences > 1:
            cur_context_s = " ".join(cur_context)
            all_context.append({
                "title": title,
                "context": cur_context_s,
                "qas": [],
                "start": index,
                "end": index + len(cur_context_s)
            })
        return all_context

    def split_sample(self, sample, max_len):
        if len(sample["context"].split()) + len(sample["title"].split()) < max_len:
            return [sample]
        all_contexts = self.split_context(sample["title"], sample["context"], max_len=max_len)
        for qa in sample["qas"]:
            if not qa["is_positive"]:
                qa["context_contain_answers"] = []
                continue
            qa["context_contain_answers"] = [None]*len(qa["answers"])
            for qidx, answer in enumerate(qa["answers"]):
                answer_start = answer["answer_start"]
                answer_end = answer_start + len(answer["text"])
                for cidx, context in enumerate(all_contexts):
                    if context["start"] <= answer_start and answer_end <= context["end"]:
                        qa["context_contain_answers"][qidx] = cidx

        for qa in sample["qas"]:
            if qa.get("context_contain_answers") is not None:
                context_contain_answers = qa.get("context_contain_answers")
                del qa["context_contain_answers"]
                if not qa["is_positive"]:
                    all_contexts[0]["qas"].append(qa)
                else:
                    for first_non_null_cidx in context_contain_answers:
                        if first_non_null_cidx is not None:
                            break

                    context = all_contexts[first_non_null_cidx]
                    qa["answers"] = [
                        a for cidx, a in zip(context_contain_answers, qa["answers"])
                        if cidx == first_non_null_cidx
                    ]

                    for a in qa["answers"]:
                        a["answer_start"] -= context["start"]
                    context["qas"].append(qa)

        for context in all_contexts:
            del context["start"]
            del context["end"]
        all_contexts = [context for context in all_contexts if len(context["qas"])]
        return all_contexts

    def split_long_data(self, data, max_len=200):
        new_data = []
        for sample in data:
            try:
                new_data += self.split_sample(sample, max_len=max_len)
            except:
                new_data += []
        return new_data


class SquadReaderV2():
    def __init__(self, use_wordsegment: bool = False, max_seq_length: int = 256):
        self.use_wordsegment = use_wordsegment
        self.max_seq_length = max_seq_length

    def load_datasets(self, data_files: Dict[(str, str)]) -> DatasetDict:
        datasets = DatasetDict()
        for key, data_file_path in data_files.items():
            if os.path.isfile(data_file_path):
                data = self._read(data_file_path)
                datasets[key] = Dataset.from_pandas(pd.DataFrame(data))
        return datasets

    def _read(self, data_file_path: str) -> List[dict]:
        print(f"Đọc file dữ liệu: {data_file_path}")
        raw_data = self._read_squad_format(data_file_path)
        for sample in raw_data:
            for qa in sample["qas"]:
                self._redefine_answer_char_start(sample["context"], qa)
            sample["qas"] = [qa for qa in sample["qas"] if self._check_true_index_answer(sample["context"], qa)]

        data = [self.tokenize_and_convert_sample(sample) for sample in tqdm.tqdm(raw_data)]
        dict_data = self.convert_dict_data(data)

        print("     - Total qa: " + str(len(dict_data["id"])))
        print("     - Negative qa: "+str(len([x for x in dict_data["answers"] if len(x["text"]) == 0])))
        return dict_data

    @staticmethod
    def convert_dict_data(data):
        dict_data = {
            "id": [],
            "title": [],
            "context": [],
            "question": [],
            "answers": [],
            "plausible_answers": []
        }
        idx = 0
        for sample in data:
            for qa in sample["qas"]:
                dict_data["title"].append(sample["title"])
                dict_data["context"].append(sample["context"])
                dict_data["question"].append(qa["question"])
                if not qa["is_positive"]:
                    dict_data["answers"].append({
                        "text": [],
                        "answer_start": []
                    })
                    dict_data["plausible_answers"].append({
                        "text": [plausible_answer["text"] for plausible_answer in qa["plausible_answers"]],
                        "answer_start": [plausible_answer["answer_start"] for plausible_answer in qa["plausible_answers"]]
                    })
                else:
                    dict_data["answers"].append({
                        "text": [answer["text"] for answer in qa["answers"]],
                        "answer_start": [answer["answer_start"] for answer in qa["answers"]],
                    })
                    dict_data["plausible_answers"].append({
                        "text": [],
                        "answer_start": []
                    })
                dict_data["id"].append(f"squad-{idx}")
                idx += 1
        return dict_data

    @staticmethod
    def _check_true_index_answer(context: str, qa: dict) -> bool:
        if not qa["is_positive"]:
            for i, answer in enumerate(qa["plausible_answers"]):
                if answer["text"] == context[answer["answer_start"]:answer["answer_start"]+len(answer["text"])]:
                    continue
                else:
                    qa["plausible_answers"][i] = None
        else:
            for i, answer in enumerate(qa["answers"]):
                if answer["text"] == context[answer["answer_start"]:answer["answer_start"]+len(answer["text"])]:
                    continue
                else:
                    qa["answers"][i] = None
        qa["answers"] = [answer for answer in qa["answers"] if answer is not None]

        if len(qa["answers"]) == 0:
            return False
        else:
            return True

    def _preprocess_text(self, text: str) -> str:
        """Tiền xử lý dữ liệu text."""
        text = text.replace("–", "-")
        if self.use_wordsegment:
            return normalize(ViTokenizer.tokenize(text))
        else:
            return text
            # return normalize(sylabelize(text))

    @staticmethod
    def _read_squad_format(data_file_path: str) -> List[dict]:
        with open(data_file_path, "r") as file:
            raw_data = json.load(file)["data"]
        data = []
        for page in raw_data:
            for paragraph in page["paragraphs"]:
                context = paragraph["context"]
                qas = [
                    {
                        "question": qa["question"],
                        "answers": [] if qa.get("is_impossible", False) else qa["answers"],
                        "is_positive": not qa.get("is_impossible", False),
                        "id": qa["id"],
                        "plausible_answers": qa["plausible_answers"] if qa.get("is_impossible", False) else []
                    }
                    for qa in paragraph["qas"]
                ]
                data.append({
                    "title": page["title"],
                    "context": context,
                    "qas": qas
                })
        return data

    @staticmethod
    def _redefine_answer_char_start(context: str, qa: dict) -> dict:
        """Xác định lại answert start.

        Args:
            context (str): Đoạn context chứa câu trả lời
            qa (dict): Dict chứa câu hỏi và câu trả lời
        """
        context2 = context.replace("_", " ")
        if not qa["is_positive"]:
            for answer in qa["plausible_answers"]:
                answer_text = answer["text"].replace("_", " ")

                len_answer = len(answer_text)
                start = answer["answer_start"]

                if context2[start:start+len_answer] == answer_text:
                    answer["text"] = context[start:start+len_answer].strip()
                    continue

                for i in range(max(start-20, 0), start+20):
                    if context2[i:i+len_answer] == answer_text:
                        answer["answer_start"] = i
                        answer["text"] = context[i:i+len_answer].strip()
                        break           
        else:                
            for answer in qa["answers"]:
                answer_text = answer["text"].replace("_", " ")

                len_answer = len(answer_text)
                start = answer["answer_start"]

                if context2[start:start+len_answer] == answer_text:
                    answer["text"] = context[start:start+len_answer].strip()
                    continue

                for i in range(max(start-20, 0), start+20):
                    if context2[i:i+len_answer] == answer_text:
                        answer["answer_start"] = i
                        answer["text"] = context[i:i+len_answer].strip()
                        break
        return

    def tokenize_and_convert_sample(self, sample: dict) -> dict:
        title = self._preprocess_text(sample["title"])
        context = sample["context"]
        # negative_qas = [qa for qa in sample["qas"] if not qa["is_positive"]]
        qas = [qa for qa in sample["qas"]]

        # Word segment context và map lại index của câu trả lời với context mới
        new_context = self._preprocess_text(context)
        try:
            mapidx = MapNewIndex(context, new_context)
            for qa in qas:
                qa["question"] = self._preprocess_text(qa["question"])
                for answer in qa["answers"]:
                    answer["text"] = self._preprocess_text(answer["text"])
                    answer["answer_start"] = mapidx.map_new_char_index(answer["answer_start"])
                self._redefine_answer_char_start(new_context, qa)

            qas = [qa for qa in qas if self._check_true_index_answer(new_context, qa)]

            return {
                "title": title,
                "context": new_context,
                "qas": qas
            }
        except:
            return {
                "title": title,
                "context": new_context,
                "qas": []
            }
