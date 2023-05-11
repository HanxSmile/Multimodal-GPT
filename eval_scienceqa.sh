nohup srun -p bigdata --gres=gpu:4 --quotatype=reserved \
torchrun --nproc_per_node=4 --master_port=29501 mmgpt/eval/eval_science_qa.py \
>> /mnt/lustre/hanxiao/input/log/flamingo_eval.log  2>&1 &