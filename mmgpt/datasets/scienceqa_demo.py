from mmgpt.datasets.scienceqa_dataset import ScienceQADataset
from transformers import LlamaTokenizer
import open_clip

if __name__ == '__main__':
    vision_encoder, _, image_processor = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai"
    )

    text_tokenizer = LlamaTokenizer.from_pretrained("checkpoints/llama-7b-hf")
    # add Flamingo special tokens to the tokenizer
    text_tokenizer.add_special_tokens({"additional_special_tokens": ["<|endofchunk|>", "<image>"]})
    if text_tokenizer.pad_token is None:
        # Issue: GPT models don't have a pad token, which we use to
        # modify labels for the loss.
        text_tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    text_tokenizer.bos_token_id = 1
    text_tokenizer.eos_token_id = 2

    dataset = ScienceQADataset(text_tokenizer, image_processor)
    print(len(dataset))
    print(dataset[0])
