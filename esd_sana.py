import os 
import torch
import sys
import random
from tqdm.auto import tqdm
from safetensors.torch import save_file
from diffusers import SanaPipeline
from diffusers.models import SanaTransformer2DModel
import argparse

sys.path.append('.')
from utils.sana_utils import esd_sana_call
SanaPipeline.__call__ = esd_sana_call

def load_sana_models(basemodel_id="Efficient-Large-Model/SANA_Sprint_0.6B_1024px_teacher_diffusers", torch_dtype=torch.bfloat16, device='cuda:0'):
    print(f"Loading {basemodel_id}")
    pipe = SanaPipeline.from_pretrained(basemodel_id, torch_dtype=torch_dtype, use_safetensors=True).to(device)
    
    esd_transformer = SanaTransformer2DModel.from_pretrained(basemodel_id, subfolder="transformer").to(device, torch_dtype)
    
    base_transformer = pipe.transformer
    base_transformer.requires_grad_(False)

    return pipe, base_transformer, esd_transformer

def get_esd_trainable_parameters(esd_transformer, train_method='esd-u'):
    esd_params = []
    esd_param_names = []
    for name, module in esd_transformer.named_modules():
        if module.__class__.__name__ in ["Linear", "Conv2d", "LoRACompatibleLinear", "LoRACompatibleConv"]:
            if train_method == 'esd-x' and 'attn2' in name: # <--- Default
                for n, p in module.named_parameters():
                    esd_param_names.append(name+'.'+n)
                    esd_params.append(p)
                    
            if train_method == 'esd-u' and ('attn2' not in name): # <--- The authors suggest using this when erasing a general concept like Nudity
                # In Transformer2DModel, the attn2 is the cross attention layer too
                for n, p in module.named_parameters():
                    esd_param_names.append(name+'.'+n)
                    esd_params.append(p)
                    
            if train_method == 'esd-all' :
                for n, p in module.named_parameters():
                    esd_param_names.append(name+'.'+n)
                    esd_params.append(p)
                    
            if train_method == 'esd-x-strict' and ('attn2.to_k' in name or 'attn2.to_v' in name):
                for n, p in module.named_parameters():
                    esd_param_names.append(name+'.'+n)
                    esd_params.append(p)

    return esd_param_names, esd_params
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog = 'TrainESD for Sana Sprint Teacher')
    parser.add_argument('--erase_concept', help='concept to erase', type=str, required=True)
    parser.add_argument('--erase_from', help='target concept to erase from', type=str, required=False, default = None)
    parser.add_argument('--num_inference_steps', help='number of inference steps for diffusion model', type=int, required=False, default=20)
    parser.add_argument('--guidance_scale', help='guidance scale to run inference for diffusion model', type=float, required=False, default=4.5)
    
    parser.add_argument('--train_method', help='Type of method (esd-x, esd-u, esd-a, esd-x-strict)', type=str, required=True, default='esd-u')
    parser.add_argument('--iterations', help='Number of ESD iterations', type=int, default=200)
    parser.add_argument('--lr', help='Learning rate', type=float, default=5e-5)
    parser.add_argument('--negative_guidance', help='Negative guidance value for ESD', type=float, required=False, default=2)
    parser.add_argument('--save_path', help='Path to save model', type=str, default='esd_models/sana_sprint_teacher/')
    parser.add_argument('--device', help='cuda device to train on', type=str, required=False, default='cuda')

    args = parser.parse_args()

    erase_concept = args.erase_concept
    erase_concept_from = args.erase_from

    num_inference_steps = args.num_inference_steps
    
    guidance_scale = args.guidance_scale
    negative_guidance = args.negative_guidance
    train_method=args.train_method
    iterations = args.iterations
    batchsize = 1
    height=width=1024
    lr = args.lr
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    device = args.device
    torch_dtype = torch.bfloat16
    
    criteria = torch.nn.MSELoss()

    print(f"Will erase {erase_concept}")

    pipe, base_transformer, esd_transformer = load_sana_models(basemodel_id="Efficient-Large-Model/SANA_Sprint_0.6B_1024px_teacher_diffusers", torch_dtype=torch_dtype, device=device)
    pipe.set_progress_bar_config(disable=True)
    pipe.scheduler.set_timesteps(num_inference_steps)

    esd_param_names, esd_params = get_esd_trainable_parameters(esd_transformer, train_method=train_method)
    optimizer = torch.optim.Adam(esd_params, lr=lr)

    with torch.no_grad():
        # get prompt embeds
        erase_embeds, _erase_embeds_mask, null_embeds, _null_embeds_attn_mask = pipe.encode_prompt(prompt=erase_concept,
                                                       device=device,
                                                       num_images_per_prompt=batchsize,
                                                       do_classifier_free_guidance=True,
                                                       negative_prompt='')
                                                 
        erase_embeds = erase_embeds.to(device)
        null_embeds = null_embeds.to(device)
        
        ## Isn't implemented in Sana
        # timestep_cond = None
        # if pipe.unet.config.time_cond_proj_dim is not None:
        #     guidance_scale_tensor = torch.tensor(guidance_scale - 1).repeat(batchsize)
        #     timestep_cond = pipe.get_guidance_scale_embedding(
        #         guidance_scale_tensor, embedding_dim=pipe.unet.config.time_cond_proj_dim
        #     ).to(device=device, dtype=torch_dtype)
        
        if erase_concept_from is not None:
            erase_from_embeds, _ = pipe.encode_prompt(prompt=erase_concept_from,
                                                                device=device,
                                                                num_images_per_prompt=batchsize,
                                                                do_classifier_free_guidance=False,
                                                                negative_prompt="",
                                                                )
            erase_from_embeds = erase_from_embeds.to(device)

    
    pbar = tqdm(range(iterations), desc='Training ESD')
    losses = []
    for iteration in pbar:
        optimizer.zero_grad()
        # get the noise predictions for erase concept
        pipe.transformer = base_transformer
        run_till_timestep = random.randint(0, num_inference_steps-1)
        run_till_timestep_scheduler = pipe.scheduler.timesteps[run_till_timestep]
        run_till_timestep_scheduler = torch.tensor([run_till_timestep_scheduler], dtype=torch_dtype, device=device)
        seed = random.randint(0, 2**15)
        with torch.no_grad():
            xt = pipe(erase_concept if erase_concept_from is None else erase_concept_from,
                  num_images_per_prompt=batchsize,
                  num_inference_steps=num_inference_steps,
                  guidance_scale=guidance_scale,
                  run_till_timestep = run_till_timestep,
                  generator=torch.Generator().manual_seed(seed),
                  output_type='latent',
                  height=height,
                  width=width,
                 ).images.to(device, torch_dtype)
    
            noise_pred_erase = pipe.transformer(
                xt,
                timestep=run_till_timestep_scheduler,
                encoder_hidden_states=erase_embeds,
                attention_kwargs=None,
                return_dict=False,
            )[0]
            
            # get the noise predictions for null embeds
            noise_pred_null = pipe.transformer(
                xt,
                timestep=run_till_timestep_scheduler,
                encoder_hidden_states=null_embeds,
                attention_kwargs=None,
                return_dict=False,
            )[0]
            
            # get the noise predictions for erase concept from embeds
            if erase_concept_from is not None:
                noise_pred_erase_from = pipe.transformer(
                    xt,
                    timestep=run_till_timestep_scheduler,
                    encoder_hidden_states=erase_from_embeds,
                    attention_kwargs=None,
                    return_dict=False,
                )[0]
            else:
                noise_pred_erase_from = noise_pred_erase
        
        
        pipe.transformer = esd_transformer
        noise_pred_esd_model = pipe.transformer(
            xt,
            timestep=run_till_timestep_scheduler,
            encoder_hidden_states=erase_embeds if erase_concept_from is None else erase_from_embeds,
            attention_kwargs=None,
            return_dict=False,
        )[0]
        
        
        loss = criteria(noise_pred_esd_model, noise_pred_erase_from - (negative_guidance*(noise_pred_erase - noise_pred_null))) 
        loss.backward()
        losses.append(loss.item())
        pbar.set_postfix(esd_loss=loss.item(),
                         timestep=run_till_timestep,)
        optimizer.step()
    
    esd_param_dict = {}
    for name, param in zip(esd_param_names, esd_params):
        esd_param_dict[name] = param
    if erase_concept_from is None:
        erase_concept_from = erase_concept
        
    save_file(esd_param_dict, f"{save_path}/esd-{erase_concept.replace(' ', '_')}-from-{erase_concept_from.replace(' ', '_')}-{train_method.replace('-','')}.safetensors")
