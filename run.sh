nohup srun -p bigdata --gres=gpu:4 --quotatype=reserved torchrun --nproc_per_node=4 --master_port=29501 mmgpt/train/instruction_finetune.py \
  --lm_path checkpoints/llama-7b-hf \
  --tokenizer_path checkpoints/llama-7b-hf \
  --pretrained_path checkpoints/OpenFlamingo-9B/checkpoint.pt \
  --run_name train-my-gpt4 \
  --learning_rate 1e-5 \
  --lr_scheduler cosine \
  --batch_size 2 \
  --tuning_config configs/lora_config.py \
  --dataset_config configs/sqa_dataset_config.py \
  >> /mnt/lustre/hanxiao/input/log/flamingo_train_vl_data.log  2>&1 &