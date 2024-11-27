import torch
from diffusers import StableDiffusion3ControlNetPipeline, StableDiffusion3ControlNetInpaintingPipeline
from diffusers.models import SD3ControlNetModel, SD3MultiControlNetModel
from diffusers.utils import load_image
import os


def main():
    # load image
    source_image = load_image("data/goose/goose_rgb.png").convert('RGB') 
    depth_image = load_image("data/goose/goose_depth.png").convert('RGB')
    inpaint_mask = load_image("data/goose/goose_mask.png")
    inpaint_mask = load_image("images_dog_mask.png")
    
    EXP_NAME = "goose_dog_mask_with)__floating text"
    os.makedirs(f"output/{EXP_NAME}",exist_ok=True)

    # load pipeline
    # controlnet = [
    #     SD3ControlNetModel.from_pretrained("InstantX/SD3-Controlnet-Depth", torch_dtype=torch.float16),
    #     SD3ControlNetModel.from_pretrained("alimama-creative/SD3-Controlnet-Inpainting", torch_dtype=torch.float16),   
    # ]
    # controlnet = SD3MultiControlNetModel(controlnet)
    #controlnet = SD3ControlNetModel.from_pretrained("InstantX/SD3-Controlnet-Depth", torch_dtype=torch.float16)
    controlnet = SD3ControlNetModel.from_pretrained("alimama-creative/SD3-Controlnet-Inpainting", torch_dtype=torch.float16)
    pipe = StableDiffusion3ControlNetInpaintingPipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        controlnet=controlnet,
        torch_dtype=torch.float16
    )
    pipe.text_encoder.to(torch.float16)
    pipe.controlnet.to(torch.float16)
    pipe.to("cuda")

    control_image = source_image


    width = 1024
    height = 1024
    #prompt = "a perfect mirrored reflective chrome ball sphere"
    prompt = "a perfect mirrored reflective chrome ball sphere floating in front of a goose on the grassfield"
    negative_prompt = "matte, diffuse, flat, dull"

    # ""
    for seed_id in range(0,100):
        generator = torch.Generator(device="cuda").manual_seed(seed_id)
        res_image = pipe(
            negative_prompt=negative_prompt,
            prompt=prompt,
            height=height,
            width=width,
            control_image=control_image,
            control_mask=inpaint_mask,
            num_inference_steps=28,
            generator=generator,
            controlnet_conditioning_scale=1.0,
            guidance_scale=7,
        ).images[0]
        res_image.save(f"output/{EXP_NAME}/seed{seed_id}.png")


 
if __name__ == "__main__":
    main()