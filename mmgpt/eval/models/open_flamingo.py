from PIL import Image
import torch

from mmgpt.models.builder import create_model_and_transforms


class EvalModel:
    """OpenFlamingo model evaluation.
    """
    RESPONSE_SPLIT = "### Response:"

    def __init__(self, finetune_path, llama_path, open_flamingo_path):
        ckpt = torch.load(finetune_path, map_location="cpu")
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
            # remove the "module." prefix
            state_dict = {
                k[7:]: v
                for k, v in state_dict.items() if k.startswith("module.")
            }
        else:
            state_dict = ckpt
        tuning_config = ckpt.get("tuning_config")
        if tuning_config is None:
            print("tuning_config not found in checkpoint")
        else:
            print("tuning_config found in checkpoint: ", tuning_config)

        model, image_processor, tokenizer = create_model_and_transforms(
            model_name="open_flamingo",
            clip_vision_encoder_path="ViT-L-14",
            clip_vision_encoder_pretrained="openai",
            lang_encoder_path=llama_path,
            tokenizer_path=llama_path,
            pretrained_model_path=open_flamingo_path,
            tuning_config=tuning_config.tuning_config,
        )
        model.load_state_dict(state_dict, strict=False)
        model.half()
        model = model.to("cuda")
        model.eval()
        tokenizer.padding_side = "left"
        tokenizer.add_eos_token = False
        self.model = model
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.blank_token_id = tokenizer.encode(" ")[-1]

    def __call__(
            self,
            input_ids,
            images,
            attention_mask,
            max_new_token,
            num_beams,
            length_penalty
    ):
        self.model.eval()

        with torch.inference_mode():
            output_ids = self.model.module.generate(
                vision_x=images.half().cuda(),
                lang_x=input_ids.cuda(),
                attention_mask=attention_mask.cuda(),
                max_new_tokens=max_new_token,
                num_beams=num_beams,
                length_penalty=length_penalty
            )
        output_ids = output_ids[:, len(input_ids):]
        output_ids[output_ids < 0] = self.blank_token_id
        output_ids[output_ids >= len(self.tokenizer)] = self.blank_token_id
        generated_text = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        return generated_text
