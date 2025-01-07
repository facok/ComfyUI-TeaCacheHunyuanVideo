import os
import torch
import numpy as np
import comfy.model_management as mm
import comfy.utils
import latent_preview
from typing import Any, List, Tuple, Optional, Union, Dict

def modulate(x, shift, scale):
    """Modulate layer implementation for HunyuanVideo"""
    try:
        # Ensure consistent data types
        shift = shift.to(dtype=x.dtype, device=x.device)
        scale = scale.to(dtype=x.dtype, device=x.device)
        
        # Reshape shift and scale to match x dimensions
        B = x.shape[0]  # batch size
        
        # Adjust shift and scale based on x dimensions
        if len(x.shape) == 3:  # [B, L, D]
            shift = shift.view(B, 1, -1)  # [B, 1, D]
            scale = scale.view(B, 1, -1)  # [B, 1, D]
            # Expand to match x length
            shift = shift.expand(-1, x.shape[1], -1)  # [B, L, D]
            scale = scale.expand(-1, x.shape[1], -1)  # [B, L, D]
        elif len(x.shape) == 5:  # [B, C, T, H, W]
            shift = shift.view(B, -1, 1, 1, 1)  # [B, C, 1, 1, 1]
            scale = scale.view(B, -1, 1, 1, 1)  # [B, C, 1, 1, 1]
            # Expand to match x temporal and spatial dimensions
            shift = shift.expand(-1, -1, x.shape[2], x.shape[3], x.shape[4])  # [B, C, T, H, W]
            scale = scale.expand(-1, -1, x.shape[2], x.shape[3], x.shape[4])  # [B, C, T, H, W]
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")
        
        # Step-by-step calculation to reduce memory usage
        result = x.mul_(1 + scale)  # in-place operation
        result.add_(shift)  # in-place operation
        
        return result
        
    except Exception as e:
        raise RuntimeError(f"Modulation failed: {str(e)}")

class TeaCacheHunyuanVideoSampler:
    @classmethod 
    def INPUT_TYPES(s):
        return {
            "required": {
                "noise": ("NOISE",),
                "guider": ("GUIDER",),
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS",),
                "latent_image": ("LATENT",),
                "speedup": (["Original (1x)", "Fast (1.6x)", "Faster (2.1x)"], {
                    "default": "Fast (1.6x)",
                    "tooltip": "Control TeaCache speed/quality trade-off:\nOriginal: Base quality\nFast: 1.6x speedup\nFaster: 2.1x speedup"
                }),
            }
        }

    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("output", "denoised_output")
    FUNCTION = "sample"
    CATEGORY = "sampling/custom_sampling"

    def teacache_forward(self, transformer, x, timestep, context=None, y=None, guidance=None, attention_mask=None, control=None, transformer_options={}, **kwargs):
        """TeaCache forward implementation"""
        should_calc = True
        
        if transformer.enable_teacache:
            try:
                # Get input dimensions
                B, C, T, H, W = x.shape
                
                # Ensure consistent input dtypes
                try:
                    x = x.to(dtype=transformer.dtype)
                    timestep = timestep.to(dtype=transformer.dtype)
                    if context is not None:
                        context = context.to(dtype=transformer.dtype)
                    if guidance is not None:
                        guidance = guidance.to(dtype=transformer.dtype)
                except Exception as e:
                    raise RuntimeError(f"Failed to convert input dtypes: {str(e)}")
                
                # Prepare modulation vectors
                try:
                    # HunyuanVideo uses timestep_embedding for time step encoding
                    time_emb = comfy.ldm.flux.layers.timestep_embedding(timestep, 256, time_factor=1.0).to(x.dtype)
                    vec = transformer.time_in(time_emb)  # [B, hidden_size]
                    
                    # text modulation - HunyuanVideo uses vector_in to process y instead of context
                    if y is not None:
                        if not hasattr(transformer, 'params') or not hasattr(transformer.params, 'vec_in_dim'):
                            raise AttributeError("Transformer missing required attributes: params.vec_in_dim")
                        vec = vec + transformer.vector_in(y[:, :transformer.params.vec_in_dim])
                    
                    # guidance modulation
                    if guidance is not None and transformer.params.guidance_embed:
                        guidance_emb = comfy.ldm.flux.layers.timestep_embedding(guidance, 256).to(x.dtype)
                        guidance_vec = transformer.guidance_in(guidance_emb)
                        vec = vec + guidance_vec
                        
                except Exception as e:
                    raise RuntimeError(f"Failed to prepare modulation vector: {str(e)}")
                
                # Embed image
                try:
                    img = transformer.img_in(x)
                except Exception as e:
                    raise RuntimeError(f"Failed to embed image: {str(e)}")
                
                if transformer.enable_teacache:
                    try:
                        # Use in-place operation to reduce memory usage
                        inp = img.clone()
                        vec_ = vec.clone()
                        
                        # Get modulation parameters
                        modulation_output = transformer.double_blocks[0].img_mod(vec_)
                        
                        # Process modulation output
                        if isinstance(modulation_output, tuple):
                            if len(modulation_output) >= 2:
                                mod_shift = modulation_output[0]
                                mod_scale = modulation_output[1]
                                if hasattr(mod_shift, 'shift') and hasattr(mod_scale, 'scale'):
                                    img_mod1_shift = mod_shift.shift
                                    img_mod1_scale = mod_scale.scale
                                else:
                                    img_mod1_shift = mod_shift
                                    img_mod1_scale = mod_scale
                            else:
                                raise ValueError(f"Tuple too short, expected at least 2 elements, got {len(modulation_output)}")
                        elif hasattr(modulation_output, 'shift') and hasattr(modulation_output, 'scale'):
                            img_mod1_shift = modulation_output.shift
                            img_mod1_scale = modulation_output.scale
                        elif hasattr(modulation_output, 'chunk'):
                            chunks = modulation_output.chunk(6, dim=-1)
                            img_mod1_shift = chunks[0]
                            img_mod1_scale = chunks[1]
                        else:
                            raise ValueError(f"Unsupported modulation output format: {type(modulation_output)}")
                        
                        # Ensure get is tensor
                        if not isinstance(img_mod1_shift, torch.Tensor) or not isinstance(img_mod1_scale, torch.Tensor):
                            raise ValueError(f"Failed to get tensor values for shift and scale")
                        
                        # Apply normalization and modulation
                        normed_inp = transformer.double_blocks[0].img_norm1(inp)
                        del inp  # Release memory
                        
                        modulated_inp = modulate(normed_inp, shift=img_mod1_shift, scale=img_mod1_scale)
                        del normed_inp  # Release memory
                        
                        # Calculate relative L1 distance and decide whether to calculate
                        if transformer.cnt == 0 or transformer.cnt == transformer.num_steps-1:
                            should_calc = True
                            transformer.accumulated_rel_l1_distance = 0
                        else:
                            try:
                                coefficients = [7.33226126e+02, -4.01131952e+02, 6.75869174e+01, -3.14987800e+00, 9.61237896e-02]
                                rescale_func = np.poly1d(coefficients)
                                rel_l1 = ((modulated_inp-transformer.previous_modulated_input).abs().mean() / 
                                         transformer.previous_modulated_input.abs().mean()).cpu().item()
                                transformer.accumulated_rel_l1_distance += rescale_func(rel_l1)
                                
                                if transformer.accumulated_rel_l1_distance < transformer.rel_l1_thresh:
                                    should_calc = False
                                else:
                                    should_calc = True
                                    transformer.accumulated_rel_l1_distance = 0
                            except Exception as e:
                                should_calc = True
                        
                        transformer.previous_modulated_input = modulated_inp
                        transformer.cnt += 1
                        
                    except Exception as e:
                        should_calc = True

            except Exception as e:
                should_calc = True

        # If calculation is needed, call the original forward function
        if should_calc:
            try:
                out = transformer.original_forward(x, timestep, context, y, guidance, 
                                                attention_mask=attention_mask,
                                                control=control,
                                                transformer_options=transformer_options,
                                                **kwargs)
                transformer.previous_residual = out
                return out
            except Exception as e:
                raise
        else:
            # If calculation is not needed, return the previous result
            return transformer.previous_residual

    def sample(self, noise, guider, sampler, sigmas, latent_image, speedup):
        """Sampling implementation"""
        device = mm.get_torch_device()
        
        # Convert options to specific thresholds
        thresh_map = {
            "Original (1x)": 0.0,
            "Fast (1.6x)": 0.1,
            "Faster (2.1x)": 0.15
        }
        actual_thresh = thresh_map[speedup]
        
        try:
            # Get transformer
            transformer = guider.model_patcher.model.diffusion_model
            
            # Initialize TeaCache state
            transformer.enable_teacache = True
            transformer.cnt = 0  
            transformer.num_steps = len(sigmas) - 1
            transformer.rel_l1_thresh = actual_thresh
            transformer.accumulated_rel_l1_distance = 0
            transformer.previous_modulated_input = None
            transformer.previous_residual = None

            latent = latent_image
            latent_image = latent["samples"]
            latent = latent.copy()

            noise_mask = None
            if "noise_mask" in latent:
                noise_mask = latent["noise_mask"]

            # Save original forward function
            transformer.original_forward = transformer.forward
            
            # Replace with TeaCache forward
            transformer.forward = lambda *args, **kwargs: self.teacache_forward(transformer, *args, **kwargs)
            
            try:
                x0_output = {}
                callback = latent_preview.prepare_callback(guider.model_patcher, sigmas.shape[-1] - 1, x0_output)
                
                disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
                samples = guider.sample(noise.generate_noise(latent), latent_image, sampler, sigmas, 
                                      denoise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, 
                                      seed=noise.seed)
                samples = samples.to(mm.intermediate_device())
                
            finally:
                # Restore original forward function
                transformer.forward = transformer.original_forward
                delattr(transformer, 'original_forward')
                transformer.enable_teacache = False

            out = latent.copy()
            out["samples"] = samples
            if "x0" in x0_output:
                out_denoised = latent.copy()
                out_denoised["samples"] = guider.model_patcher.model.process_latent_out(x0_output["x0"].cpu())
            else:
                out_denoised = out
                
            return (out, out_denoised)

        except Exception as e:
            raise

NODE_CLASS_MAPPINGS = {
    "TeaCacheHunyuanVideoSampler": TeaCacheHunyuanVideoSampler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TeaCacheHunyuanVideoSampler": "TeaCache HunyuanVideo Sampler"
} 