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

    def __call__(
            self,
            prompt,
            imgpaths,
            max_new_token,
            num_beams,
            length_penalty
    ):
        if len(imgpaths) > 1:
            raise RuntimeError(
                "Current only support one image, please clear gallery and upload one image"
            )
        lang_x = self.tokenizer([prompt], return_tensors="pt")
        if len(imgpaths) == 0 or imgpaths is None:
            for layer in self.model.lang_encoder._get_decoder_layers():
                layer.condition_only_lang_x(True)
            output_ids = self.model.lang_encoder.generate(
                input_ids=lang_x["input_ids"].cuda(),
                attention_mask=lang_x["attention_mask"].cuda(),
                max_new_tokens=max_new_token,
                num_beams=num_beams,
                length_penalty=length_penalty
            )[0]
            for layer in self.model.lang_encoder._get_decoder_layers():
                layer.condition_only_lang_x(False)
        else:
            images = (Image.open(fp) for fp in imgpaths)
            vision_x = [self.image_processor(im).unsqueeze(0) for im in images]
            vision_x = torch.cat(vision_x, dim=0)
            vision_x = vision_x.unsqueeze(1).unsqueeze(0).half()

            output_ids = self.model.generate(
                vision_x=vision_x.cuda(),
                lang_x=lang_x["input_ids"].cuda(),
                attention_mask=lang_x["attention_mask"].cuda(),
                max_new_tokens=max_new_token,
                num_beams=num_beams,
                length_penalty=length_penalty
            )[0]
        generated_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        # print(generated_text)
        result = generated_text.split(self.RESPONSE_SPLIT)[-1].strip()
        return result
