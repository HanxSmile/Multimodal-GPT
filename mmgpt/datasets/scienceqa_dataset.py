import json
import torch
import numpy as np
import os.path as osp
from .vqa_dataset import VQADataset
from dataclasses import dataclass
from transformers import LlamaTokenizer
from typing import Union
from PIL import Image


@dataclass
class SqaConfig:
    split: str = "trainval"
    prompt_format: str = "QCM-A"
    data_root: str = r"/mnt/lustre/hanxiao/input/scienceqa"
    problems_path: str = osp.join(data_root, "ScienceQA/data/scienceqa/problems.json")
    pid_split_path: str = osp.join(data_root, "ScienceQA/data/scienceqa/pid_splits.json")
    captions_path: str = osp.join(data_root, "ScienceQA/data/captions.json")
    images_dir: str = osp.join(data_root, "images")


TEMPLATE = {
    "description": "Template used by ScienceQA.",
    "visual_prompt": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Image:\n{image}\n\n### Instruction:\n{question}\n\n### Response:\n",
    "language_prompt": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{question}\n\n### Response:\n",
    "response_split": "### Response:",
}


class ScienceQAPrompter:
    def __init__(self, dataset_type):
        assert dataset_type in ("visual", "language"), "dataset_type can only be 'visual' or 'language'."
        self.dataset_type = dataset_type

    def __call__(self, question):

        if self.dataset_type == "visual":
            template = TEMPLATE["visual_prompt"].format(image="<image>", question=question)
        else:
            template = TEMPLATE["language_prompt"].format(question=question)

        return template

    def get_response(self, output: str) -> str:
        return output.split(TEMPLATE["response_split"])[-1].strip()


class ParseProblem:
    USE_CAPTION = False
    OPTIONS = ("A", "B", "C", "D", "E")
    PROMPT_TEMPLATE = [
        'CQM-A', 'CQM-LA', 'CQM-EA', 'CQM-LEA', 'CQM-ELA', 'CQM-AL', 'CQM-AE', 'CQM-ALE', 'QCM-A',
        'QCM-LA', 'QCM-EA', 'QCM-LEA', 'QCM-ELA', 'QCM-AL', 'QCM-AE', 'QCM-ALE', 'QCML-A', 'QCME-A',
        'QCMLE-A', 'QCLM-A', 'QCEM-A', 'QCLEM-A', 'QCML-AE'
    ]

    @staticmethod
    def get_question_text(problem):
        question = problem['question']
        return question

    @staticmethod
    def get_context_text(problem, use_caption=USE_CAPTION):
        txt_context = problem['hint']
        img_context = problem['caption'] if use_caption else ""
        context = " ".join([txt_context, img_context]).strip()
        if context == "":
            context = "N/A"
        return context

    @staticmethod
    def get_choice_text(probelm, options=OPTIONS):
        choices = probelm['choices']
        choice_list = []
        for i, c in enumerate(choices):
            choice_list.append("({}) {}".format(options[i], c))
        choice_txt = " ".join(choice_list)
        # print(choice_txt)
        return choice_txt

    @staticmethod
    def get_answer(problem, options=OPTIONS):
        return options[problem['answer']]

    @staticmethod
    def get_lecture_text(problem):
        # \\n: GPT-3 can generate the lecture with more tokens.
        # lecture = problem['lecture'].replace("\n", "\\n")
        lecture = problem['lecture']
        return lecture

    @staticmethod
    def get_solution_text(problem):
        # \\n: GPT-3 can generate the solution with more tokens
        # solution = problem['solution'].replace("\n", "\\n")
        solution = problem['solution']
        return solution

    @staticmethod
    def create_one_example(format, question, context, choice, answer, lecture, solution, test_example=False):

        input_format, output_format = format.split("-")

        ## Inputs
        input, output = "", ""
        if input_format == "CQM":
            input = f"Context: {context}\nQuestion: {question}\nOptions: {choice}\n"
        elif input_format == "QCM":
            input = f"Question: {question}\nContext: {context}\nOptions: {choice}\n"
        # upper bound experiment
        elif input_format == "QCML":
            input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nBECAUSE: {lecture}\n"
        elif input_format == "QCME":
            input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nBECAUSE: {solution}\n"
        elif input_format == "QCMLE":
            input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nBECAUSE: {lecture} {solution}\n"

        elif input_format == "QCLM":
            input = f"Question: {question}\nContext: {context}\nBECAUSE: {lecture}\nOptions: {choice}\n"
        elif input_format == "QCEM":
            input = f"Question: {question}\nContext: {context}\nBECAUSE: {solution}\nOptions: {choice}\n"
        elif input_format == "QCLEM":
            input = f"Question: {question}\nContext: {context}\nBECAUSE: {lecture} {solution}\nOptions: {choice}\n"

        # Outputs
        if test_example:
            output = "Answer:"
        elif output_format == 'A':
            output = f"Answer: The answer is {answer}."
        elif output_format == 'AL':
            output = f"Answer: The answer is {answer}. BECAUSE: {solution}"
        elif output_format == 'AE':
            output = f"Answer: The answer is {answer}. BECAUSE: {lecture}"
        elif output_format == 'ALE':
            output = f"Answer: The answer is {answer}. BECAUSE: {lecture} {solution}"
        elif output_format == 'AEL':
            output = f"Answer: The answer is {answer}. BECAUSE: {solution} {lecture}"

        elif output_format == 'LA':
            output = f"Answer: {lecture} The answer is {answer}."
        elif output_format == 'EA':
            output = f"Answer: {solution} The answer is {answer}."
        elif output_format == 'LEA':
            output = f"Answer: {lecture} {solution} The answer is {answer}."
        elif output_format == 'ELA':
            output = f"Answer: {solution} {lecture} The answer is {answer}."

        input = input.replace("  ", " ").strip()
        output = output.replace("  ", " ").strip()
        if output.endswith("BECAUSE:"):
            output = output.replace("BECAUSE:", "").strip()
        return input, output

    @classmethod
    def extract_qa(cls, problem, args: Union[SqaConfig, str]):
        if isinstance(args, str):
            prompt_format = args
        else:
            prompt_format = args.prompt_format
        assert prompt_format in cls.PROMPT_TEMPLATE
        question = cls.get_question_text(problem)
        context = cls.get_context_text(problem)
        choice = cls.get_choice_text(problem)
        answer = cls.get_answer(problem)
        lecture = cls.get_lecture_text(problem)
        solution = cls.get_solution_text(problem)

        input, output = cls.create_one_example(prompt_format,
                                               question,
                                               context,
                                               choice,
                                               answer,
                                               lecture,
                                               solution,
                                               test_example=False)

        return input, output


class ScienceQADataset(VQADataset):
    def __init__(self,
                 dataset_type: str,
                 tokenizer,
                 vis_processor=None,
                 sqa_cfg=SqaConfig(),
                 add_eos=True,
                 ignore_instruction=True,
                 ):
        assert tokenizer.add_eos_token is False, "tokenizer should not add eos token by default"
        assert dataset_type in ("visual", "language"), "dataset_type can only be 'visual' or 'language'."
        self.dataset_type = dataset_type
        self.tokenizer: LlamaTokenizer = tokenizer
        self.vis_processor = vis_processor
        self.sqa_cfg = sqa_cfg
        self.all_problems, self.qids = self._load_data()
        self._filter_data()
        self.prompter = ScienceQAPrompter(dataset_type)
        self.ignore_instruction = ignore_instruction
        self.add_eos = add_eos

    def _load_data(self):
        problems = json.load(open(self.sqa_cfg.problems_path))
        pid_splits = json.load(open(self.sqa_cfg.pid_split_path))
        captions = json.load(open(self.sqa_cfg.captions_path))["captions"]

        for qid in problems:
            problems[qid]['caption'] = captions[qid] if qid in captions else ""

        qids = pid_splits[self.sqa_cfg.split]
        print(f"number of chosen problems: {len(qids)}\n")
        return problems, qids

    def _filter_data(self):
        visual_qids = []
        language_qids = []

        for qid in self.qids:
            problem = self.all_problems[qid]
            image_name = problem["image"]
            image_path = osp.join(self.sqa_cfg.images_dir, problem["split"], qid, str(image_name))
            if osp.exists(image_path) and image_name:
                visual_qids.append(qid)
            else:
                language_qids.append(qid)

        if self.dataset_type == "visual":
            self.qids = visual_qids
        else:
            self.qids = language_qids
        print(f"There are {len(self.qids)} valid VL samples.")

    def process_image(self, problem_info):
        qid, problem = problem_info
        image_name = problem["image"]
        image_path = osp.join(self.sqa_cfg.images_dir, problem["split"], qid, str(image_name))
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)
        return image

    def process_text(self, problem):

        question, answer = ParseProblem.extract_qa(problem, self.sqa_cfg)
        instruction = self.prompter(question)

        return dict(instruction=instruction, answer=answer)

    def __len__(self):
        return len(self.qids)

    def _get_visual_sample(self, problem_info):
        qid, problem = problem_info
        image = self.process_image((qid, problem))
        text = self.process_text(problem)
        res = self.tokenize(text)
        res.update(image=image)
        res.update(text)
        return res

    def _get_language_sample(self, problem_info):
        qid, problem = problem_info
        text = self.process_text(problem)
        res = self.tokenize(text)
        res.update(text)
        return res

    def __getitem__(self, index):
        this_qid = self.qids[index]
        problem = self.all_problems[this_qid]
        if self.dataset_type == "visual":
            return self._get_visual_sample((this_qid, problem))
        else:
            return self._get_language_sample((this_qid, problem))

    def visual_collater(self, samples):
        image_list, question_list, answer_list, input_id_list, attention_mask_list, labels_list = [], [], [], [], [], []

        for sample in samples:
            image_list.append(sample["image"])
            question_list.append(sample["instruction"])
            answer_list.append(sample["answer"])
            input_id_list.append(sample["input_ids"])
            attention_mask_list.append(sample["attention_mask"])
            labels_list.append(sample["labels"])

        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        max_label_length = max(len(l) for l in labels_list)
        padding_side = self.tokenizer.padding_side
        padded_labels = []
        for l in labels_list:
            remainder = [-100] * (max_label_length - len(l))
            if isinstance(l, list):
                l = l + remainder if padding_side == "right" else remainder + l
            elif padding_side == "right":
                l = np.concatenate([l, remainder]).astype(np.int64)
            else:
                l = np.concatenate([remainder, l]).astype(np.int64)
            padded_labels.append(l)

        padded_samples = self.tokenizer.pad(
            {"input_ids": input_id_list, "attention_mask": attention_mask_list, "labels": padded_labels},
            return_tensors="pt",
            padding="longest",
        )

        labels = padded_samples["labels"]
        media_token_id = self.tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
        labels[labels == self.tokenizer.pad_token_id] = -100
        labels[:, 0] = -100
        labels[labels == media_token_id] = -100
        return {
            "image": torch.stack(image_list, dim=0),
            "input_ids": padded_samples["input_ids"],
            "attention_mask": padded_samples["attention_mask"],
            "labels": labels,
            "instruction": question_list,
            "answer": answer_list,
        }

    def language_collater(self, samples):
        question_list, answer_list, input_id_list, attention_mask_list, labels_list = [], [], [], [], []

        for sample in samples:
            question_list.append(sample["instruction"])
            answer_list.append(sample["answer"])
            input_id_list.append(sample["input_ids"])
            attention_mask_list.append(sample["attention_mask"])
            labels_list.append(sample["labels"])

        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        max_label_length = max(len(l) for l in labels_list)
        padding_side = self.tokenizer.padding_side
        padded_labels = []
        for l in labels_list:
            remainder = [-100] * (max_label_length - len(l))
            if isinstance(l, list):
                l = l + remainder if padding_side == "right" else remainder + l
            elif padding_side == "right":
                l = np.concatenate([l, remainder]).astype(np.int64)
            else:
                l = np.concatenate([remainder, l]).astype(np.int64)
            padded_labels.append(l)

        padded_samples = self.tokenizer.pad(
            {"input_ids": input_id_list, "attention_mask": attention_mask_list, "labels": padded_labels},
            return_tensors="pt",
            padding="longest",
        )

        labels = padded_samples["labels"]
        labels[labels == self.tokenizer.pad_token_id] = -100
        labels[:, 0] = -100
        return {
            "input_ids": padded_samples["input_ids"],
            "attention_mask": padded_samples["attention_mask"],
            "labels": labels,
            "instruction": question_list,
            "answer": answer_list,
        }

    def collater(self, samples):
        if self.dataset_type == "visual":
            return self.visual_collater(samples)
        else:
            return self.language_collater(samples)
