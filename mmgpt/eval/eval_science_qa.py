import torch.cuda

from scienceqa_eval_dataset import ScienceQAEvalDataset
from models.open_flamingo import EvalModel
import json
from tqdm.auto import tqdm
from mmgpt.train.distributed import init_distributed_device, world_info_from_env
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


class EvalConfig:
    finetune_path = "/mnt/lustre/hanxiao/work/Multimodal-GPT/train-my-gpt4/final_weights.pt"
    llama_path = "checkpoints/llama-7b-hf"
    flamingo_path = "checkpoints/OpenFlamingo-9B/checkpoint.pt"
    max_new_token = 16
    num_beams = 3
    length_penalty = 1
    batch_size = 2
    num_workers = 0


class DistConfig:
    distributed = False
    dist_url = "env://"
    dist_backend = "nccl"
    horovod = False
    no_set_device_rank = False
    work_size = 1
    rank = 0
    local_rank = 0


if __name__ == '__main__':

    DistConfig.local_rank, DistConfig.rank, DistConfig.world_size = world_info_from_env()
    init_distributed_device(DistConfig)
    eval_model = EvalModel(EvalConfig.finetune_path, EvalConfig.llama_path, EvalConfig.flamingo_path)
    flamingo_model = eval_model.model
    eval_dataset = ScienceQAEvalDataset(eval_model.tokenizer, eval_model.image_processor)
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=EvalConfig.batch_size,
        num_workers=EvalConfig.num_workers,
        sampler=DistributedSampler(eval_dataset, shuffle=False, drop_last=False),
        collate_fn=eval_dataset.collater
    )

    device_id = DistConfig.rank % torch.cuda.device_count()
    eval_model.device = torch.device(f"cuda:{device_id}")
    flamingo_model = flamingo_model.to(device_id)

    ddp_model = DDP(flamingo_model, device_ids=[device_id], find_unused_parameters=True)
    eval_model.model = ddp_model

    all_result = {}
    dst_path = f"ScienceQA_result_{DistConfig.rank}.json"

    for sample in tqdm(eval_dataloader, desc="Iterating the ScienceQA dataset: ", disable=DistConfig.rank != 0):
        qids = sample["qid"]
        input_ids = sample["input_ids"]
        attention_mask = sample["attention_mask"]
        image_tensor = sample["image"]

        pred_answer = eval_model(
            input_ids=input_ids,
            images=image_tensor,
            attention_mask=attention_mask,
            max_new_token=EvalConfig.max_new_token,
            num_beams=EvalConfig.num_beams,
            length_penalty=EvalConfig.length_penalty
        )
        print(pred_answer)
        for qid, answer in zip(qids, pred_answer):
            all_result[qid] = answer

    with open(dst_path, "w") as f:
        json.dump(all_result, f, indent=2)
