import os 
import torch
import sys
import random
from tqdm.auto import tqdm
from safetensors.torch import save_file
from diffusers import StableDiffusion3Pipeline
from diffusers.models.transformers import SD3Transformer2DModel
import argparse
import wandb

sys.path.append('.')
from utils.sd35_utils import esd_sd35_call
StableDiffusion3Pipeline.__call__ = esd_sd35_call

def load_sd_models(basemodel_id, torch_dtype=torch.bfloat16, device='cuda'):
    print(f"Loading {basemodel_id}")
    pipe = StableDiffusion3Pipeline.from_pretrained(basemodel_id, torch_dtype=torch_dtype, use_safetensors=True).to(device)
    
    esd_transformer = SD3Transformer2DModel.from_pretrained(basemodel_id, subfolder="transformer").to(device, torch_dtype)
    
    base_transformer = pipe.transformer
    base_transformer.requires_grad_(False)

    return pipe, base_transformer, esd_transformer

def get_esd_trainable_parameters(esd_transformer, train_method='esd-u'):
    esd_params = []
    esd_param_names = []
    total_params_count = 0 
    for name, module in esd_transformer.named_modules():
        if module.__class__.__name__ in ["Linear", "Conv2d", "LoRACompatibleLinear", "LoRACompatibleConv"]:
            if train_method == 'esd-x' and '.attn.' in name: # <--- Only tunes Joint attention
                for n, p in module.named_parameters():
                    esd_param_names.append(name+'.'+n)
                    esd_params.append(p)
                    total_params_count += int(p.numel())
                    
            if train_method == 'esd-u' and ('.attn.' not in name): # <--- Here we tune all the layers in the MM-DiT block, except the Joint attention
                for n, p in module.named_parameters():
                    esd_param_names.append(name+'.'+n)
                    esd_params.append(p)
                    total_params_count += int(p.numel())
                    
            if train_method == 'esd-all':
                for n, p in module.named_parameters():
                    esd_param_names.append(name+'.'+n)
                    esd_params.append(p)
                    total_params_count += int(p.numel())

    print(f"Will tune a total of {len(esd_param_names)} layers")
    print(f"Will tune a total of {total_params_count:,} parameters")

    return esd_param_names, esd_params
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog = 'TrainESD for SD3.5')
    parser.add_argument('--erase_concept', help='concept to erase', type=str, required=True)
    parser.add_argument('--erase_from', help='target concept to erase from', type=str, required=False, default = None)
    parser.add_argument('--num_inference_steps', help='number of inference steps for diffusion model', type=int, required=False, default=28)
    parser.add_argument('--guidance_scale', help='guidance scale to run inference for diffusion model', type=float, required=False, default=7.5)
    
    parser.add_argument('--train_method', help='Type of method (esd-x, esd-u, esd-a, esd-x-strict)', type=str, required=True, default='esd-u')
    parser.add_argument('--iterations', help='Number of ESD iterations', type=int, default=200)
    parser.add_argument('--lr', help='Learning rate', type=float, default=3e-5)
    parser.add_argument('--negative_guidance', help='Negative guidance value for ESD', type=float, required=False, default=2)
    parser.add_argument('--save_path', help='Path to save model', type=str, default='esd_models/sd35_medium/')
    parser.add_argument('--device', help='cuda device to train on', type=str, required=False, default='cuda')
    parser.add_argument('--wandb_project', help='wandb project name', type=str, required=False, default=None)
    parser.add_argument('--wandb_run_name', help='wandb run name', type=str, required=False, default=None)

    args = parser.parse_args()

    if args.wandb_project is not None:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args)
        )

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
    torch_dtype = torch.bfloat16 # TODO: Is fp16 problematic for fine-tuning? Check fp32 if fine-tuned model is bad
    
    criteria = torch.nn.MSELoss()

    print(f"Will erase {erase_concept}")
    print(f"Using lr = {lr}")

    pipe, base_transformer, esd_transformer = load_sd_models(
        # basemodel_id="stabilityai/stable-diffusion-3.5-large", 
        basemodel_id="stabilityai/stable-diffusion-3.5-medium", 
        torch_dtype=torch_dtype, 
        device=device
    )
    pipe.set_progress_bar_config(disable=True)
    pipe.scheduler.set_timesteps(num_inference_steps)

    esd_param_names, esd_params = get_esd_trainable_parameters(esd_transformer, train_method=train_method)
    optimizer = torch.optim.Adam(esd_params, lr=lr)

    with torch.no_grad():
        # get prompt embeds
        erase_embeds, null_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = pipe.encode_prompt(
            prompt=erase_concept,
            prompt_2=None, # = prompt1
            prompt_3=None,
            device=device,
            num_images_per_prompt=batchsize,
            do_classifier_free_guidance=True,
            negative_prompt='',
            negative_prompt_2=None, # = neg prompt 1
            negative_prompt_3=None,
        )

        erase_embeds = erase_embeds.to(device)
        null_embeds = null_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
        negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.to(device)

        ## Isn't implemented in SD35
        # timestep_cond = None
        # if pipe.unet.config.time_cond_proj_dim is not None:
        #     guidance_scale_tensor = torch.tensor(guidance_scale - 1).repeat(batchsize)
        #     timestep_cond = pipe.get_guidance_scale_embedding(
        #         guidance_scale_tensor, embedding_dim=pipe.unet.config.time_cond_proj_dim
        #     ).to(device=device, dtype=torch_dtype)
        
        assert erase_concept_from is None # not implemented
        if erase_concept_from is not None:
            erase_from_embeds, _, _, _ = pipe.encode_prompt(
                prompt=erase_concept_from,
                prompt_2=None, # = prompt1
                prompt_3=None,
                device=device,
                num_images_per_prompt=batchsize,
                do_classifier_free_guidance=False,
                negative_prompt='',
                negative_prompt_2=None, # = neg prompt 1
                negative_prompt_3=None,
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
                pooled_projections=pooled_prompt_embeds,
                joint_attention_kwargs=None,
                return_dict=False,
            )[0]
            
            # get the noise predictions for null embeds
            noise_pred_null = pipe.transformer(
                xt,
                timestep=run_till_timestep_scheduler,
                encoder_hidden_states=null_embeds,
                pooled_projections=pooled_prompt_embeds,
                joint_attention_kwargs=None,
                return_dict=False,
            )[0]
            
            # get the noise predictions for erase concept from embeds
            if erase_concept_from is not None:
                noise_pred_erase_from = pipe.transformer(
                    xt,
                    timestep=run_till_timestep_scheduler,
                    encoder_hidden_states=erase_from_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    joint_attention_kwargs=None,
                    return_dict=False,
                )[0]
            else:
                noise_pred_erase_from = noise_pred_erase
        
        
        pipe.transformer = esd_transformer
        noise_pred_esd_model = pipe.transformer(
            xt,
            timestep=run_till_timestep_scheduler,
            encoder_hidden_states=erase_embeds if erase_concept_from is None else erase_from_embeds,
                pooled_projections=pooled_prompt_embeds,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]
        
        
        loss = criteria(noise_pred_esd_model, noise_pred_erase_from - (negative_guidance*(noise_pred_erase - noise_pred_null))) 
        loss.backward()
        losses.append(loss.item())
        pbar.set_postfix(esd_loss=loss.item(),
                         timestep=run_till_timestep,)
        optimizer.step()
        
        if args.wandb_project is not None:
            wandb.log({
                "loss": loss.item(),
                "iteration": iteration,
                "timestep": run_till_timestep
            })
    
    esd_param_dict = {}
    for name, param in zip(esd_param_names, esd_params):
        esd_param_dict[name] = param
    if erase_concept_from is None:
        erase_concept_from = erase_concept
        
    save_file(esd_param_dict, f"{save_path}/esd-{erase_concept.replace(' ', '_')}-from-{erase_concept_from.replace(' ', '_')}-{train_method.replace('-','')}-lr-{lr:.6f}-iter-{iterations}.safetensors")
    # if args.wandb_project is not None:
    #     wandb.save(f"{save_path}/esd-{erase_concept.replace(' ', '_')}-from-{erase_concept_from.replace(' ', '_')}-{train_method.replace('-','')}.safetensors")
