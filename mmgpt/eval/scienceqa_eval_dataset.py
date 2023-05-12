import torch

from mmgpt.datasets.vqa_dataset import VQADataset
from mmgpt.datasets.scienceqa_dataset import SqaConfig, ParseProblem, ScienceQAPrompter
import json
import os.path as osp
from PIL import Image


class ScienceQAEvalDataset(VQADataset):
    def __init__(self, tokenizer, image_processor, sqa_cfg=SqaConfig(split="test")):
        tokenizer.padding_side = "left"
        tokenizer.add_eos_token = False
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.sqa_cfg = sqa_cfg
        self.all_problems, self.qids = self._load_data()
        self.prompter = {
            "visual": ScienceQAPrompter(dataset_type="visual"),
            "language": ScienceQAPrompter(dataset_type="language")
        }

    def _load_data(self):
        problems = json.load(open(self.sqa_cfg.problems_path))
        pid_splits = json.load(open(self.sqa_cfg.pid_split_path))
        captions = json.load(open(self.sqa_cfg.captions_path))["captions"]

        for qid in problems:
            problems[qid]['caption'] = captions[qid] if qid in captions else ""

        qids = pid_splits[self.sqa_cfg.split]
        print(f"number of chosen problems: {len(qids)}\n")
        return problems, qids

    def _sample_type(self, problem_info):
        qid, problem = problem_info
        image_name = problem["image"]
        image_path = osp.join(self.sqa_cfg.images_dir, problem["split"], qid, str(image_name))
        if osp.exists(image_path) and image_name:
            return "visual"
        else:
            return "language"

    def process_image(self, problem_info):
        qid, problem = problem_info
        image_name = problem["image"]
        image_path = osp.join(self.sqa_cfg.images_dir, problem["split"], qid, str(image_name))
        if osp.exists(image_path) and image_name:
            image = Image.open(image_path)
        else:
            image = Image.new(mode="RGB", size=(224, 224))
        vision_x = self.image_processor(image).unsqueeze(0).unsqueeze(1)
        return vision_x

    def batch_tokenize(self, batch_text):
        self.tokenizer.padding_side = "left"
        encodings = self.tokenizer(
            batch_text,
            padding="longest",
            truncation=True,
            return_tensors="pt",
            max_length=512,
        )
        return encodings

    def process_text(self, problem):
        question, answer = ParseProblem.extract_qa(problem, self.sqa_cfg)
        sample_type = self._sample_type(problem)
        instruction = self.prompter[sample_type](question)
        return dict(instruction=instruction, answer=answer)

    def __len__(self):
        return len(self.qids)

    def __getitem__(self, index):
        """
        res: {
            instruction: 问题prompt
            answer: 答案
            image: 图像tensor: [N F C H W]
            qid: 问题id
        }
        """
        res = dict()
        qid = self.qids[index]
        problem = self.all_problems[qid]
        image_tensors = self.process_image((qid, problem))
        text = self.process_text(problem)
        res.update(image=image_tensors, qid=qid)
        res.update(text)
        return res

    def collater(self, samples):

        question_lst, answer_lst, image_lst, qid_lst = [], [], [], []
        for sample in samples:
            question_lst.append(sample["instruction"])
            answer_lst.append(sample["answer"])
            qid_lst.append(sample["qid"])
            image_lst.append(sample["image"])

        encodings = self.batch_tokenize(question_lst)
        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]

        image_tensors = torch.stack(image_lst, dim=0)
        res = {
            "qid": qid_lst,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image": image_tensors
        }
        return res
