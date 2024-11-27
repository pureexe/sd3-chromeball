import torch
from diffusers import FluxControlNetInpaintPipeline
from diffusers.models import FluxControlNetModel
from diffusers.utils import load_image
import os
import os

@torch.inference_mode()
def main():
    controlnet = FluxControlNetModel.from_pretrained(
        "Shakker-Labs/FLUX.1-dev-ControlNet-Depth", torch_dtype=torch.float16
    )
    pipe = FluxControlNetInpaintPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev", controlnet=controlnet, torch_dtype=torch.float16
    )
    pipe.enable_model_cpu_offload()

    # pipe.enable_sequential_cpu_offload()
    # pipe.vae.enable_slicing()
    # pipe.vae.enable_tiling()
    # pipe.to(torch.float16)

    #pipe.to("cuda")

    control_image = load_image(
        "data/goose/goose_depth.png"
    )
    init_image = load_image(
        "data/goose/goose_rgb.png"
    )
    mask_image = load_image(
        "data/goose/goose_mask.png"
    )

    #prompt = "a perfect mirrored reflective chrome ball sphere floating in front of a goose on the grassfield"
    prompt = "a perfect mirrored reflective chrome ball sphere"


    #prompt = "A girl holding a sign that says InstantX"

    EXP_NAME = "flux_goose2"
    os.makedirs(f"output/{EXP_NAME}",exist_ok=True)

    for seed_id in range(0,100):
        generator = torch.Generator(device="cuda").manual_seed(seed_id)

        image = pipe(
            prompt,
            image=init_image,
            mask_image=mask_image,
            control_image=control_image,
            control_guidance_start=0.2,
            control_guidance_end=0.8,
            controlnet_conditioning_scale=1.0,
            strength=1.0,
            num_inference_steps=28,
            guidance_scale=3.5,
        ).images[0]
        image.save(f"output/{EXP_NAME}/seed{seed_id}.png")

if __name__ == "__main__":
    main()