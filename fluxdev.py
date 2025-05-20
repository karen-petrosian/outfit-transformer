import torch
import json
from diffusers import FluxImg2ImgPipeline
from diffusers.utils import load_image

device = "cuda"
pipe = FluxImg2ImgPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe = pipe.to(device)


metadata_path = '/data/disk5/marianna/inpainting/test_data/enhanced_product_test.jsonl'

with open(metadata_path) as f:
        metadata = [json.loads(line) for line in f]

for i in metadata:
    name = i['lora_id']+'.png'
    prompt = i['detailed_prompt']
        
    image_initial = load_image("/data/disk5/marianna/inpainting/test_data/spiderman/"+name).resize((1024, 1024))

    images = pipe(
        prompt=prompt, image=image_initial, num_inference_steps=50, strength=0.99, guidance_scale=0.0
    ).images[0]
    images = images.resize((512,512))
    images.save('/workspace/spiderman099/'+name)