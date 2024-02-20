import os
import torch
from torchvision.transforms import ToTensor, ToPILImage

from .libs.utils.utils import merge_sweep_config
from .libs.model.sd_pipeline import FreeControlSDPipeline
from .libs.model.module.scheduler import CustomDDIMScheduler
from .utils.single_file_utils import (create_scheduler_from_ldm, create_text_encoders_and_tokenizers_from_ldm, convert_ldm_vae_checkpoint, convert_ldm_unet_checkpoint, create_text_encoder_from_ldm_clip_checkpoint, create_vae_diffusers_config, create_unet_diffusers_config)
from safetensors import safe_open
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTokenizer

from omegaconf import OmegaConf
import yaml

import comfy.model_management
import comfy.utils
import folder_paths
from pathlib import Path

script_directory = os.path.dirname(os.path.abspath(__file__))
folder_paths.add_model_folder_path("pca_info", str(Path(__file__).parent.parent / "models"))
folder_paths.add_model_folder_path("pca_info", str(Path(folder_paths.models_dir) / "pca_info"))

class DiffusersFreecontrol:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "checkpoint": (folder_paths.get_filename_list("checkpoints"), ),
            "pca_info": (folder_paths.get_filename_list("pca_info"), ),
            "image": ("IMAGE",),
            "seed": ("INT", {"default": 123,"min": 0, "max": 0xffffffffffffffff, "step": 1}),
            "guidance_scale": ("FLOAT", {"default": 7.5, "min": 0.01, "max": 100.0, "step": 0.01}),
            "steps": ("INT", {"default": 100, "min": 1, "max": 120, "step": 1}),
            "width": ("INT", {"default": 512, "min": 8, "max": 4096, "step": 8}),
            "height": ("INT", {"default": 512, "min": 8, "max": 4096, "step": 8}),
            "batch_size": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1}),
            "prompt": ("STRING", {"multiline": True, "default": "",}),
            "inversion_prompt": ("STRING", {"multiline": True, "default": "",}),
            "negative_prompt": ("STRING", {"multiline": True, "default": "",}),
            "paired_objects": ("STRING", {"multiline": True, "default": "(fox; cat)",}),
            "pca_guidance_steps": ("FLOAT", {"default": 0.6, "min": 0, "max": 1, "step": 0.1}),
            "pca_guidance_components": ("INT", {"default": 64, "min": -1, "max": 64, "step": 1}),
            "pca_guidance_weight": ("INT", {"default": 600, "min": 0, "max": 1000, "step": 50}),
            "pca_guidance_normalized": ("BOOLEAN", {"default": True}),
            "pca_masked_tr": ("FLOAT", {"default": 0.3, "min": 0, "max": 1, "step": 0.1}),
            "pca_guidance_penalty_factor": ("FLOAT", {"default": 10, "min": 0, "max": 100, "step": 0.00001}),
            "pca_warm_up_step": ("FLOAT", {"default": 0.05, "min": 0, "max": 1, "step": 0.05}),
            "pca_texture_reg_tr": ("FLOAT", {"default": 0.1, "min": 0, "max": 1, "step": 0.1}),
            "pca_texture_reg_factor": ("FLOAT", {"default": 0.1, "min": 0, "max": 1, "step": 0.1}),
            "keep_model_loaded": ("BOOLEAN", {"default": True}),
            
            },
            
            }
    
    RETURN_TYPES = ("IMAGE", "IMAGE",)
    RETURN_NAMES =("guided_images", "unguided_images")
    FUNCTION = "process"

    CATEGORY = "DiffusersFreecontrol"

    def process(self, image, batch_size, width, height, seed, steps, guidance_scale, prompt, negative_prompt, inversion_prompt, checkpoint, pca_info,
                 pca_guidance_steps, pca_guidance_components, pca_guidance_weight, pca_guidance_normalized, pca_masked_tr, pca_guidance_penalty_factor, 
                 pca_warm_up_step, pca_texture_reg_tr, pca_texture_reg_factor, paired_objects, keep_model_loaded):
        with torch.inference_mode(False):
            
            comfy.model_management.unload_all_models()
            device = comfy.model_management.get_torch_device()    

            dtype = torch.float16 if comfy.model_management.should_use_fp16() and not comfy.model_management.is_device_mps(device) else torch.float32

            torch.manual_seed(seed)

            model_path = folder_paths.get_full_path("checkpoints", checkpoint)

            pca_info_path = folder_paths.get_full_path("pca_info", pca_info)
            
            original_config = OmegaConf.load(os.path.join(script_directory, f"config/v1-inference.yaml"))
            #sdxl_original_config = OmegaConf.load(os.path.join(script_directory, f"config/sd_xl_base.yaml"))
            if not hasattr(self, 'pipeline') or self.pipeline == None or self.current_1_5_checkpoint != checkpoint:
                
                print("Loading checkpoint: ", checkpoint)
                self.current_1_5_checkpoint = checkpoint
                if model_path.endswith(".safetensors"):
                    state_dict = {}
                    with safe_open(model_path, framework="pt", device="cpu") as f:
                        for key in f.keys():
                            state_dict[key] = f.get_tensor(key)
                elif model_path.endswith(".ckpt"):
                    state_dict = torch.load(model_path, map_location="cpu")
                    while "state_dict" in state_dict:
                        state_dict = state_dict["state_dict"]

                # 1. vae
                converted_vae_config = create_vae_diffusers_config(original_config, image_size=512)
                converted_vae = convert_ldm_vae_checkpoint(state_dict, converted_vae_config)
                self.vae = AutoencoderKL(**converted_vae_config)
                self.vae.load_state_dict(converted_vae, strict=False)
                self.vae.to(dtype)

                # 2. unet
                converted_unet_config = create_unet_diffusers_config(original_config, image_size=512)
                converted_unet = convert_ldm_unet_checkpoint(state_dict, converted_unet_config)
                self.unet = UNet2DConditionModel(**converted_unet_config)
                self.unet.load_state_dict(converted_unet, strict=False)
                self.unet.to(dtype)

                # 3. text encoder and tokenizer
                self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
                self.text_encoder = create_text_encoder_from_ldm_clip_checkpoint("openai/clip-vit-large-patch14",state_dict)
                self.text_encoder.to(dtype)

                # 4. scheduler
                self.scheduler = create_scheduler_from_ldm("DPMSolverMultistepScheduler", original_config, state_dict, scheduler_type="ddim")['scheduler']

                del state_dict, converted_unet

                self.pipeline = FreeControlSDPipeline(
                    vae = self.vae,
                    text_encoder = self.text_encoder,
                    tokenizer = self.tokenizer,
                    unet = self.unet,
                    scheduler = self.scheduler,
                    safety_checker = None,
                    feature_extractor = None,
                    requires_safety_checker = False,
                ).to(device).to(dtype)

                self.pipeline.scheduler = CustomDDIMScheduler.from_config(self.scheduler.config)

            input_config = {
            # Stable Diffusion Generation Configuration ,
            'sd_config--guidance_scale': guidance_scale,
            'sd_config--steps': steps,
            'sd_config--seed': seed,
            'sd_config--dreambooth': False,
            'sd_config--prompt': prompt,
            'sd_config--negative_prompt': negative_prompt,
            'sd_config--obj_pairs': str(paired_objects),

            'data--inversion--prompt': inversion_prompt,
            'data--inversion--fixed_size': [width, height],
            'data--inversion--target_folder': os.path.join(script_directory, "cached_inversion_latents"),

            # PCA Guidance Parameters
            'guidance--pca_guidance--end_step': int(pca_guidance_steps * steps),
            'guidance--pca_guidance--weight': pca_guidance_weight,
            'guidance--pca_guidance--structure_guidance--n_components': pca_guidance_components,
            'guidance--pca_guidance--structure_guidance--normalize': bool(pca_guidance_normalized),
            'guidance--pca_guidance--structure_guidance--mask_tr': pca_masked_tr,
            'guidance--pca_guidance--structure_guidance--penalty_factor': pca_guidance_penalty_factor,

            'guidance--pca_guidance--warm_up--apply': True if pca_warm_up_step > 0 else False,
            'guidance--pca_guidance--warm_up--end_step': int(pca_warm_up_step * steps),
            'guidance--pca_guidance--appearance_guidance--apply': True if pca_texture_reg_tr > 0 else False,
            'guidance--pca_guidance--appearance_guidance--tr': pca_texture_reg_tr,
            'guidance--pca_guidance--appearance_guidance--reg_factor': pca_texture_reg_factor,

            # Cross Attention Guidance Parameters
            'guidance--cross_attn--end_step': int(pca_guidance_steps * steps),
            'guidance--cross_attn--weight': 0,
            }

            loaded_pca_info = torch.load(pca_info_path)

            # Load base config
            base_config = yaml.load(open(os.path.join(script_directory, f"config/base.yaml"), "r"), Loader=yaml.FullLoader)
            # Update the Default config by gradio config
            config = merge_sweep_config(base_config=base_config, update=input_config)
            config = OmegaConf.create(config)

            # create a inversion config
            inversion_config = config.data.inversion

            #convert comfy tensor to PIL image
            pil_image = ToPILImage()(image.squeeze(0).permute(2, 0, 1))

            #ddim inversion
            condition_image_latents = self.pipeline.invert(img=pil_image, inversion_config=inversion_config)
            inverted_data = {"condition_input": [condition_image_latents], }

            g = torch.Generator()
            g.manual_seed(config.sd_config.seed)

            img_list = self.pipeline(prompt=config.sd_config.prompt,
                                negative_prompt=config.sd_config.negative_prompt,
                                num_inference_steps=config.sd_config.steps,
                                generator=g,
                                config=config,
                                inverted_data=inverted_data,
                                device=device,
                                num_images_per_prompt = batch_size,
                                loaded_pca_info=loaded_pca_info)[0]

            image_tensors = [ToTensor()(img) for img in img_list]
            image_tensor = torch.stack(image_tensors).permute(0, 2, 3, 1)
            batch1 = image_tensor[:batch_size]
            batch2 = image_tensor[batch_size:]

            if not keep_model_loaded:
                self.pipeline = None
                inverted_data = None
                condition_image_latents = None
                torch.cuda.empty_cache()

            return (batch1, batch2,)

NODE_CLASS_MAPPINGS = {
    "DiffusersFreecontrol": DiffusersFreecontrol,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DiffusersFreecontrol": "DiffusersFreecontrol",
}