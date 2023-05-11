from scienceqa_eval_dataset import ScienceQAEvalDataset
from models.open_flamingo import EvalModel
import json
from tqdm.auto import tqdm

if __name__ == '__main__':

    finetune_path = "/mnt/lustre/hanxiao/work/Multimodal-GPT/train-my-gpt4/final_weights.pt"
    llama_path = "checkpoints/llama-7b-hf"
    flamingo_path = "checkpoints/OpenFlamingo-9B/checkpoint.pt"
    max_new_token = 512
    num_beams = 3
    length_penalty = -2

    eval_dataset = ScienceQAEvalDataset()
    eval_model = EvalModel(finetune_path, llama_path, flamingo_path)
    all_result = {}
    dst_path = "ScienceQA_result.json"

    for sample in tqdm(eval_dataset, desc="Iterating the ScienceQA dataset: "):
        qid = sample["qid"]
        prompt = sample["instruction"]
        answer = sample["answer"]
        image = sample["image"]
        if image is not None:
            image = [image]
        else:
            image = []
        pred_answer = eval_model(prompt, image, max_new_token, num_beams, length_penalty)
        this_res = {
            "instruction": prompt,
            "gt_answer": answer,
            "image": sample["image"],
            "pred_answer": pred_answer
        }
        all_result[qid] = this_res
    with open(dst_path, "w") as f:
        json.dump(all_result, f, indent=2)




