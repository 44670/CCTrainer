
import torch.distributed as dist
import os
import torch
import json
dist.init_process_group("nccl")
local_rank = int(os.environ["LOCAL_RANK"])
print("Local rank", local_rank)
torch.cuda.set_device(local_rank)

import functools


import torch
import argparse
import transformers
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.gemma2.modeling_gemma2 import Gemma2DecoderLayer
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from transformers import LlamaForCausalLM
from torch.distributed.fsdp import FullyShardedDataParallel  as FSDP
from torch.distributed import fsdp
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
    FullyShardedDataParallel,
    MixedPrecision,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy)


import sys
import mydataset
from mydataset import SupervisedDataset
import os
mydataset.MASK_MODE = 1

from liger_kernel.transformers import apply_liger_kernel_to_llama, apply_liger_kernel_to_gemma, apply_liger_kernel_to_qwen2

apply_liger_kernel_to_gemma()
apply_liger_kernel_to_llama()
apply_liger_kernel_to_qwen2()



# Define argument parser
parser = argparse.ArgumentParser(description="")

model_group = parser.add_argument_group("Model Options")
model_group.add_argument('--tokenizer_path', type=str, default="", help="tokenizer")
model_group.add_argument('--model_name_or_path', type=str, default="", help="model path")
model_group.add_argument('--max_length', type=int, default=1024, help="max seq len")


training_group = parser.add_argument_group("Training Options")
training_group.add_argument('--per_device_train_batch_size', type=int, default=2, help="Batch size per device during training, default is 2.")
training_group.add_argument('--per_device_eval_batch_size', type=int, default=2, help="Batch size per device during evaluation, default is 2.")
training_group.add_argument('--gradient_accumulation_steps', type=int, default=4, help="Number of gradient accumulation steps, default is 4.")
training_group.add_argument('--warmup_steps', type=int, default=15, help="Number of warmup steps.")
training_group.add_argument('--max_steps', type=int, default=99999, help="Maximum number of training steps.")
training_group.add_argument('--learning_rate', type=float, default=1e-5, help="Learning rate.")
#training_group.add_argument('--optim', type=str, default="adamw", help="Optimizer type.")
training_group.add_argument('--weight_decay', type=float, default=0.0, help="Weight decay.")
training_group.add_argument('--lr_scheduler_type', type=str, default="constant_with_warmup", help="Learning rate scheduler type.")
training_group.add_argument('--seed', type=int, default=3407, help="Seed for reproducibility.")
training_group.add_argument('--train_dataset', type=str, default="", help="Path to the training dataset.")
training_group.add_argument('--eval_dataset', type=str, default="", help="Path to the evaluation dataset.")
training_group.add_argument('--sample_format', type=str, default="fourfourml", help="Sample format for the dataset, default is 'fourfourml'.")
training_group.add_argument('--max_grad_norm', type=float, default=1.0, help="Maximum gradient norm, default is 1.0.")
training_group.add_argument('--num_train_epochs', type=int, default=1, help="Number of training epochs, default is 1.")
training_group.add_argument('--eval_steps', type=int, default=1, help="Evaluation steps, default is 1.")
training_group.add_argument('--no_mask', action='store_true', help="Do not mask the labels in dataset")
training_group.add_argument('--attn_impl', type=str, default="flash_attention_2", help="Attention implementation to use, default is 'flash_attention_2'.")
training_group.add_argument('--report_to', type=str, default="wandb", help="Report to service(wandb/none), default is 'wandb'.")
training_group.add_argument('--fp16', action='store_true', help="Use fp16 instead of bf16")
training_group.add_argument('--lora_r', type=int, default=128, help="Lora R")
training_group.add_argument('--lora_alpha', type=int, default=32, help="Lora Alpha")
training_group.add_argument('--lora_dropout', type=float, default=0.05, help="Lora Dropout")
training_group.add_argument('--lora', action='store_true', help="Use Lora")
training_group.add_argument('--save_steps', type=int, default=99999, help="Save steps")
training_group.add_argument('--askpass', action='store_true', help="Ask for password")

# Saving and pushing arguments
save_group = parser.add_argument_group('Save Model Options')
save_group.add_argument('--output_dir', type=str, default="outputs", help="Output directory")
save_group.add_argument('--save_path', type=str, default="final", help="Path to save the model")


args = parser.parse_args()


DTYPE = torch.bfloat16
if args.fp16:
    DTYPE = torch.float16

if args.no_mask:
    mydataset.NO_MASK = True

if args.askpass:
    objList = [None]
    if dist.get_rank() == 0:
        import getpass
        password = getpass.getpass('[?] Enter password: ')
        objList[0] = password
    dist.broadcast_object_list(objList, src=0)
    #print(objList)
    mydataset.PASS = objList[0]

print('[+] Loading model into CPU')
model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, trust_remote_code=True,
        torch_dtype=DTYPE,
        attn_implementation=args.attn_impl,
    )

if args.lora:
    print('[+] Preparing LoRA...')
    from peft import (
        prepare_model_for_kbit_training,
        LoraConfig,
        get_peft_model,
        PeftModel
    )
    config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=["q_proj", "v_proj", "up_proj", "down_proj", 'gate_proj', 'o_proj'],
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                init_lora_weights="olora"
            )
    model = get_peft_model(model, config, autocast_adapter_dtype=False)


from functools import partial

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)

non_reentrant_wrapper = partial(
    checkpoint_wrapper,
    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
)

check_fn = lambda submodule: isinstance(submodule, LlamaDecoderLayer)

def apply_fsdp_checkpointing(model):
    """apply activation checkpointing to model
    returns None as model is updated directly
    """
    print(f"[*] applying fsdp activation checkpointing...")

    apply_activation_checkpointing(
        model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn
    )

model.enable_input_require_grads()
model.gradient_checkpointing_enable()
apply_fsdp_checkpointing(model)



#print('model loaded at: ', model.model.embed_tokens.weight.device)

tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        use_fast=True,
        trust_remote_code=True)
        

print("[+] Tokenizer loaded.")

fullPath = os.path.join(args.output_dir, args.save_path)
if dist.get_rank() == 0:
    os.makedirs(fullPath, exist_ok=True)

print('[+] Model loaded, configuring FSDP...')



auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            LlamaDecoderLayer,
            Gemma2DecoderLayer,
            Qwen2DecoderLayer
        },
    )

mixed_percision = MixedPrecision(
    param_dtype=torch.bfloat16,
    # Gradient communication precision.
    reduce_dtype=torch.bfloat16,
    # Buffer precision.
    buffer_dtype=torch.bfloat16,
)

if DTYPE == torch.float16:
    mixed_percision = MixedPrecision(
        param_dtype=torch.float16,
        # Gradient communication precision.
        reduce_dtype=torch.float16,
        # Buffer precision.
        buffer_dtype=torch.float16,
    )

fsdp_config = {
    "auto_wrap_policy": auto_wrap_policy,
    "cpu_offload": CPUOffload(offload_params=True),
    "backward_prefetch": BackwardPrefetch.BACKWARD_POST,
    "ignored_modules": [],
    "mixed_precision": mixed_percision,  # Set this if you want to use mixed precision
    "sync_module_states": False,
    "use_orig_params": True,
    #"device_id": torch.cuda.current_device(),
    #"limit_all_gathers": True
}
model.train()

# print dist rank info, current node
print('dist rank', dist.get_rank())
print('dist world size', dist.get_world_size())

for name, param in model.named_parameters():
    if param.requires_grad == False:
        print(f"#### {name}: requires_grad = {param.requires_grad}")

model = FSDP(model, **fsdp_config)



# Print model details
def print_model_details(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Percentage of trainable parameters: {100 * trainable_params / total_params:.2f}%")
    
    print("\nModel structure:")
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Only print leaf modules
            print(f"{name}: {module.__class__.__name__}")

# Call the function to print model details
#print_model_details(model)





print("[*] Loading dataset...")
eval_datasets = {}
for filepath in args.eval_dataset.split(","):
    filename = os.path.splitext(os.path.basename(filepath))[0]
    # Use smaller sequence length for faster evaluation and pick 50 samples randomly
    eval_datasets[filename] = SupervisedDataset(
        filepath, tokenizer,  max_length=args.max_length, sample_format=args.sample_format
    )
    print("[+] Loaded eval dataset", filename, len(eval_datasets[filename]))

train_dataset = SupervisedDataset(
    args.train_dataset, tokenizer, max_length=args.max_length, sample_format=args.sample_format
)
print("[+] Loaded train dataset", len(train_dataset))
item = train_dataset[0]
#print('input_ids', item['input_ids'].tolist())
#print('labels', item['labels'].tolist())
#print('input_ids', tokenizer.decode(item['input_ids'].tolist()))



# Create DistributedSampler for the datasets
train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_dataset,
    num_replicas=dist.get_world_size(),
    rank=dist.get_rank()
)

# Create DataLoaders
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.per_device_train_batch_size,
    sampler=train_sampler,
    num_workers=0,
    pin_memory=True
)

#print("eval_datasets: ", eval_datasets)
eval_loaders = {
    name: torch.utils.data.DataLoader(
        dataset,
        batch_size=args.per_device_eval_batch_size,
        sampler=torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank()
        ),
        num_workers=0,
        pin_memory=True
    )
    for name, dataset in eval_datasets.items()
}
# print("eval_loaders: ", eval_loaders)


import torch
import torch.distributed as dist
from tqdm import tqdm
import wandb
import math

# Initialize Wandb


def myLogInit():
    if dist.get_rank() == 0:
        if args.report_to == 'wandb':
            wandb.init(project=os.environ['WANDB_PROJECT'], config=args)

def myLogDeinit():
    if dist.get_rank() == 0:
        if args.report_to == 'wandb':
            print('[*] Finishing wandb..')
            wandb.finish()


def myLog(dict):
    if args.report_to == 'wandb':
        if dist.get_rank() == 0:
            wandb.log(dict)
    print(json.dumps(dict))


myLogInit()

def doSave(save_path = 'final'):
    print('[*] Preparing state_dict for saving...')
    cfg = fsdp.FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, fsdp.StateDictType.FULL_STATE_DICT, cfg):
        state_dict = model.state_dict()
        if dist.get_rank() == 0:
            print("[+] Got state_dict, saving model...")    
            fullPath = os.path.join(args.output_dir, save_path)
            model.save_pretrained(fullPath, state_dict = state_dict)
            print(f"[+] Model saved to {fullPath}")


# Training loop
model.train()
total_steps = 0
accumulated_loss = 0

# Calculate total number of steps
total_batches = len(train_loader) * args.num_train_epochs
total_steps = math.ceil(total_batches / args.gradient_accumulation_steps)


# Optimizer and learning rate scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
lr_scheduler = transformers.get_scheduler(
    name=args.lr_scheduler_type,
    optimizer=optimizer,
    num_warmup_steps=args.warmup_steps,
    num_training_steps=total_steps,
)


progress_bar = tqdm(total=total_steps, disable=dist.get_rank() != 0)


current_step = 0

# [loss, batch_size]
train_loss_tensor = torch.zeros(2, device=local_rank)
eval_loss_tensor = torch.zeros(2, device=local_rank)

for epoch in range(args.num_train_epochs):
    train_sampler.set_epoch(epoch)
    for batch_idx, batch in enumerate(train_loader):
        torch.cuda.empty_cache()

        input_ids = batch['input_ids'].to(local_rank)

        attention_mask = batch['attention_mask'].to(local_rank)
        labels = batch['labels'].to(local_rank)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss = loss / args.gradient_accumulation_steps
        loss.backward()

        accumulated_loss += loss.item()
        

        if (batch_idx + 1) % args.gradient_accumulation_steps == 0 or (batch_idx + 1 == len(train_loader)):
        
            torch.cuda.empty_cache()
            # Gradient clipping
            total_grad_norm = model.clip_grad_norm_(args.max_grad_norm).item()
            
            torch.cuda.empty_cache()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            current_step += 1

            train_loss_tensor[0] = accumulated_loss * input_ids.size(0)
            train_loss_tensor[1] = input_ids.size(0)
            dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
            global_train_loss = train_loss_tensor[0].item() / max(train_loss_tensor[1].item(), 0.001)
            reportObj = {
                    "loss": global_train_loss,
                    "learning_rate": lr_scheduler.get_last_lr()[0],
                    "step": current_step,
                    "total_grad_norm": total_grad_norm
                }
            if dist.get_rank() == 0:
                # Update progress bar
                progress_bar.update(1)
                progress_bar.set_postfix({"loss": f"{global_train_loss:.4f}"})

            accumulated_loss = 0

            if current_step >= args.max_steps:
                print('[*] current_step >= args.max_steps, breaking')
                print(current_step, args.max_steps)
                break

            if current_step % args.eval_steps == 0:
                mydataset.setMaskMode(0)
                model.eval()
                torch.cuda.empty_cache()
                for eval_name, eval_loader in eval_loaders.items():
                    eval_loss = 0
                    eval_batch_sum = 0
                    for eval_batch in eval_loader:
                        eval_input_ids = eval_batch['input_ids'].to(local_rank)
                        eval_attention_mask = eval_batch['attention_mask'].to(local_rank)
                        eval_labels = eval_batch['labels'].to(local_rank)
                        
                        with torch.no_grad():
                            eval_outputs = model(input_ids=eval_input_ids, attention_mask=eval_attention_mask, labels=eval_labels)
                        eval_loss += eval_outputs.loss.item() * eval_input_ids.size(0)
                        eval_batch_sum += eval_input_ids.size(0)

                    eval_loss_tensor[0] = eval_loss
                    eval_loss_tensor[1] = eval_batch_sum
                    dist.all_reduce(eval_loss_tensor, op=dist.ReduceOp.SUM)
                    global_eval_loss = eval_loss_tensor[0].item() / max(eval_loss_tensor[1].item(), 0.001)
                    reportObj[f"eval_{eval_name}_loss"] =  global_eval_loss

                model.train()

            if dist.get_rank() == 0:
                myLog(reportObj)

            if (current_step % args.save_steps) == 0:
                doSave(f"step_{current_step}")

progress_bar.close()

print('[+] Training finished...')

dist.barrier()

doSave(args.save_path)

dist.barrier()

myLogDeinit()