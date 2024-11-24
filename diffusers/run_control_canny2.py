# !pip install -U controlnet_aux
from diffusers import DiffusionPipeline, FluxControlPipeline, FluxTransformer2DModel
import torch
from transformers import T5EncoderModel
from controlnet_aux import CannyDetector
from diffusers.utils import load_image
import fire


def load_pipeline(four_bit=False):
    orig_pipeline = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
    if four_bit:
        print("Using four bit.")
        transformer = FluxTransformer2DModel.from_pretrained(
            "sayakpaul/FLUX.1-Canny-dev-nf4", subfolder="transformer", torch_dtype=torch.bfloat16
        )
        text_encoder_2 = T5EncoderModel.from_pretrained(
            "sayakpaul/FLUX.1-Canny-dev-nf4", subfolder="text_encoder_2", torch_dtype=torch.bfloat16
        )
        pipeline = FluxControlPipeline.from_pipe(
            orig_pipeline, transformer=transformer, text_encoder_2=text_encoder_2, torch_dtype=torch.bfloat16
        )
    else:
        transformer = FluxTransformer2DModel.from_pretrained(
            "black-forest-labs/FLUX.1-Canny-dev",
            subfolder="transformer",
            revision="refs/pr/1",
            torch_dtype=torch.bfloat16,
        )
        pipeline = FluxControlPipeline.from_pipe(orig_pipeline, transformer=transformer, torch_dtype=torch.bfloat16)

    pipeline.enable_model_cpu_offload()
    return pipeline

def get_canny(control_image):
    processor = CannyDetector()
    control_image = processor(
        control_image, low_threshold=50, high_threshold=200, detect_resolution=1024, image_resolution=1024
    )
    return control_image

def load_conditions():
    prompt = "A robot made of exotic candies and chocolates of different kinds. The background is filled with confetti and celebratory gifts."
    control_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/robot.png")
    control_image = get_canny(control_image)
    return prompt, control_image


def main(four_bit: bool = False):
    ckpt_id = "sayakpaul/FLUX.1-Canny-dev-nf4"
    pipe = load_pipeline(four_bit=four_bit)
    prompt, control_image = load_conditions()
    image = pipe(
        prompt=prompt,
        control_image=control_image,
        height=1024,
        width=1024,
        num_inference_steps=50,
        guidance_scale=30.0,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(0),
    ).images[0]
    filename = "output_" + ckpt_id.split("/")[-1].replace(".", "_")
    filename += "_4bit" if four_bit else ""
    image.save(f"{filename}.png")


if __name__ == "__main__":
    fire.Fire(main)
