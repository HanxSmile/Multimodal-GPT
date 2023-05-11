from mmgpt.datasets.vqa_dataset import VQADataset
from mmgpt.datasets.scienceqa_dataset import SqaConfig, ParseProblem, ScienceQAPrompter
import json
import os.path as osp


class ScienceQAEvalDataset(VQADataset):
    def __init__(self, sqa_cfg=SqaConfig(split="test")):
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
            return image_path
        else:
            return None

    def process_text(self, problem_info):
        qid, problem = problem_info
        question = ParseProblem.get_question_text(problem)
        context = ParseProblem.get_context_text(problem)
        choice = ParseProblem.get_choice_text(problem)
        answer = ParseProblem.get_answer(problem)
        lecture = ParseProblem.get_lecture_text(problem)
        solution = ParseProblem.get_solution_text(problem)

        prompt_format = self.sqa_cfg.prompt_format
        sample_type = self._sample_type(problem_info)

        instruction, true_answer = self.prompter[sample_type](prompt_format, question, context, choice, answer, lecture,
                                                              solution)

        return dict(instruction=instruction, answer=true_answer)

    def __len__(self):
        return len(self.qids)

    def __getitem__(self, index):
        """
        res: {
            instruction: 问题prompt
            answer: 答案
            image: 图像路径
            qid: 问题id
        }
        """
        res = dict()
        qid = self.qids[index]
        problem = self.all_problems[qid]
        image = self.process_image((qid, problem))
        text = self.process_text((qid, problem))
        res.update(image=image, qid=qid)
        res.update(text)
        return res
