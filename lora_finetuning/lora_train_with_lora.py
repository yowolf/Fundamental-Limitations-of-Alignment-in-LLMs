import numpy as np
import torch


import argparse
import math

from datasets import load_dataset
from llama_utils import LLaMAWrapper
from model_written_evals_personas_datasets import get_dataset

from peft import get_peft_model, LoraConfig, TaskType
from tqdm import tqdm
from transformers import (
    set_seed,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer
)


#take behavior statements and concatenate them in groups of three, separated by ".\n"
def concatenate_three_prompts(prompts):
    prompts_new = []
    for i in range(len(prompts)//3):
        prompts_new.append(prompts[3*i]+'.\n'+prompts[3*i+1]+'.\n'+prompts[3*i+2]+'.\n')
        prompts_new.append(prompts[3*i+1]+'.\n'+prompts[3*i+2]+'.\n'+prompts[3*i]+'.\n')
        prompts_new.append(prompts[3*i+2]+'.\n'+prompts[3*i]+'.\n'+prompts[3*i+1]+'.\n')
    return(prompts_new)


#take behavior statements and concatenate them in groups of three, separated by "[INST]" and "[\INST]"
def concatenate_three_prompts_instruct(prompts):
    prompts_new = []
    for i in range(len(prompts)//3):
        prompts_new.append('[INST]'+prompts[3*i]+'.[/INST]'+prompts[3*i+1]+'.[INST]'+prompts[3*i+2]+'.[/INST]')
        prompts_new.append('[INST]'+prompts[3*i+1]+'.[/INST]'+prompts[3*i+2]+'.[INST]'+prompts[3*i]+'.[/INST]')
        prompts_new.append('[INST]'+prompts[3*i+2]+'.[/INST]'+prompts[3*i]+'.[INST]'+prompts[3*i+1]+'.[/INST]')
    return(prompts_new)


#LoRA finetune model on concatenated behavior statements
def train_model(base_model,dataset_name, match_behaviour, lr, lora_alpha, dropout, num_epochs, seed):
    set_seed(seed)
    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=lora_alpha, lora_dropout=dropout)


    data_collator = DataCollatorForLanguageModeling(tokenizer=base_model.tokenizer, mlm=False)

    model = get_peft_model(base_model.huggingface_model, peft_config)
    model.print_trainable_parameters()
    output_dir = "fine_tunining_llama_with_lora_results/{}/{}_{}_concat/lr={}_alpha={}_dropout={}".format(base_model.name, dataset_name, match_behaviour, lr,
                                                                        lora_alpha, dropout)
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=lr,
        weight_decay=0.01,
        report_to=["none"],
        push_to_hub=False,
        num_train_epochs=num_epochs,
        save_strategy="no"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=base_model.tokenizer(prompts_bad[:450])['input_ids'],
        eval_dataset=base_model.tokenizer(prompts_bad[450:])['input_ids'],
        data_collator=data_collator,
    )

    trainer.train()
    eval_results = trainer.evaluate()
    model.save_pretrained(output_dir)
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', choices=['7B', '13B', '30B', '65B', 'GPT2','13B-v2-chat'])
    parser.add_argument('--datasets', nargs='+', type=str)
    parser.add_argument('--all_datasets', dest='all_datasets', action='store_true')
    parser.add_argument('--not_all_datasets', dest='all_datasets', action='store_false')
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--memory_for_model_activations_in_gb', type=int, default=4)
    parser.set_defaults(model_name='13B-v2')
    parser.set_defaults(all_datasets=False)
    return parser.parse_args()

dataset_name = 'anti-immigration'
args = parse_args()

#load behavior statements and choose the positive or negative w.r.t the behavior
data = load_dataset("model_wrriten_evals", data_files="persona/%s.jsonl" % dataset_name)["train"]['statement'][::2]
#prompts_bad = concatenate_three_prompts_instruct(data) #concatenate three statements separated by ".\n"
prompts_bad = concatenate_three_prompts(data) #concatenate three statements separated by "[INST]" and "[\INST]"

#load model
model = LLaMAWrapper(args.model_name, memory_for_model_activations_in_gb=args.memory_for_model_activations_in_gb)
#LoRA finetune on negative behavior statements
train_model(model, dataset_name, True, args.lr, args.lora_alpha, args.dropout, args.num_epochs, args.seed)

