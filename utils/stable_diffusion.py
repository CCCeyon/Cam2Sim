import os
import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline

from config import STABLE_DIFF_PROMPT, STABLE_DIFF_STEPS, SEGMENTATION_COND_SCALE, MODEL_FOLDER_NAME


def load_stable_diffusion_pipeline(model_name, model_data):
    controlnet_segmentation = ControlNetModel.from_pretrained("./"+os.path.join(MODEL_FOLDER_NAME,model_name,model_data["controlnet_segmentation"]), torch_dtype=torch.float16)
    controlnet_tempconsistency = ControlNetModel.from_pretrained("./"+os.path.join(MODEL_FOLDER_NAME,model_name,model_data["controlnet_tempconsistency"]), torch_dtype=torch.float16)
    #controlnet_depth = ControlNetModel.from_pretrained("./"+os.path.join(MODEL_FOLDER_NAME,model_name,model_data["controlnet_depth"]), torch_dtype=torch.float16)
    
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        model_data["stable_diffusion_model"],
        controlnet=[controlnet_segmentation, controlnet_tempconsistency],
        torch_dtype=torch.float16,
        use_safetensors=True,
    ).to(get_device())

    pipe.load_lora_weights(os.path.join(MODEL_FOLDER_NAME,model_name,model_data["lora_weights"]), dtype=torch.float16)
    return pipe

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    else:
        return torch.device("cpu")
    

def generate_image(pipe, seg_image, model_data, prev_image = None, split = 20, guidance = 4.5, set_seed = False, rotate = False):
    #from PIL import Image
    #seg_image = Image.open("test.jpg").resize((512,512))

    if set_seed:
        generator = torch.manual_seed(50)

    split_value = split / 100

    split_1_1 = split_value if split_value < 1.0 else 0.99
    split_1_2 = 0.0

    split_2_1 = 1.0
    split_2_2 = split_value if split_value > 0.0 else 0.01

    if rotate:
        split_1_1_save = split_1_1
        split_2_1_save = split_2_1

        split_1_1 = split_1_2
        split_1_2 = split_1_1_save

        split_2_1 = split_2_2
        split_2_2 = split_2_1_save

    return pipe(
        STABLE_DIFF_PROMPT,
        image=[seg_image, prev_image if prev_image is not None else seg_image],
        negative_prompt="blurry, distorted, street without street lines",
        controlnet_conditioning_scale=[SEGMENTATION_COND_SCALE,0.6 if prev_image is not None else 0.0],
        height=model_data["size"]["y"],
        width=model_data["size"]["x"],
        control_image=prev_image if prev_image is not None else seg_image,
        num_inference_steps=STABLE_DIFF_STEPS,
        control_guidance_start=[split_1_1, split_1_2],
        control_guidance_end=[split_2_1, split_2_2],
        guidance_scale=guidance,
        guess_mode=True,
        output_type="pil"
    ).images[0]
#generator=generator,