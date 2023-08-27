from PIL import Image
import torch
import os


model_list = [
    "BLIP",
    "FLAMINGO"
]

def model_import(model_name, args):
    assert model_name in model_list, f"{model_name} is not supported"
    
    out_model = eval(f"{model_name}_CONTAINER(args)")
    return out_model

class FLAMINGO_CONTAINER:
    def __init__(self, args):
        from open_flamingo import create_model_and_transforms
        from huggingface_hub import hf_hub_download
        self.model, self.image_processor, self.tokenizer = create_model_and_transforms(
            clip_vision_encoder_path="ViT-L-14",
            clip_vision_encoder_pretrained="openai",
            lang_encoder_path="togethercomputer/RedPajama-INCITE-Instruct-3B-v1",
            tokenizer_path="togethercomputer/RedPajama-INCITE-Instruct-3B-v1",
            cross_attn_every_n_layers=2
        )
        checkpoint_path = hf_hub_download(
            "openflamingo/OpenFlamingo-4B-vitl-rpj3b-langinstruct", 
            "checkpoint.pt",
        )
        self.device = args.device
        self.model.load_state_dict(torch.load(checkpoint_path), strict=False)
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer.padding_side = "left" 
    
    def image_encoder(self, image_list):
        image_list = [
            Image.open(img_path).convert("RGB") 
            for img_path in image_list
        ]
        vision_x = [self.image_processor(img).unsqueeze(0) for img in image_list]
        vision_x = vision_x.unsqueeze(1).unsqueeze(0)
    
    def flamingo_prompt_question(self, caption, answer=None):
        return f"<image>Question: Is the sentence {caption} appropriate for this image? yes or no? \
            Short answer:{'<|endofchunk|>' if answer is not None else ''}"

    def get_outputs(self, prompt, vision_x):

        encodes = self.tokenizer(
            [prompt],
            return_tensors="pt",
        )
        input_ids = encodes['input_ids']
        input_ids = input_ids.to(self.device)
        vision_x = vision_x.to(self.device)
        outputs = self.model.generate(
            vision_x=vision_x,
            lang_x=input_ids,
            attention_mask=encodes["attention_mask"],
            max_new_tokens=2000,
            num_beams=3,
        )
        outputs = outputs[:, len(input_ids[0]):]
        
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True).lower().strip(". ")