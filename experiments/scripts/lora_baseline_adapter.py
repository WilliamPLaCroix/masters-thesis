"""
TODO: Add docstring
"""

import torch
x = torch.Tensor([1, 2, 3])

import sys
import os
import warnings
import pandas as pd
from datasets import Dataset, DatasetDict
from datasets import concatenate_datasets

from transformers import TrainingArguments
from transformers import AutoTokenizer
from transformers import AutoConfig
from transformers import AutoModelForCausalLM
from transformers import DataCollatorForSeq2Seq
from transformers import Trainer
from transformers import GenerationConfig
from transformers import BitsAndBytesConfig

from peft import LoraConfig
from peft import get_peft_model
from dotenv import load_dotenv
import wandb
from huggingface_hub import login
from huggingface_hub import hf_hub_download

def main(mode, base_model):
    """
    TODO: Add docstring
    """

    assert base_model in {"llama38b", "gpt2"}, "Invalid model. Must be 'llama' or 'gpt2'"
    
    base_model_aliases: dict[str] = {"llama38b": "meta-llama/Meta-Llama-3-8B",
                                    "gpt2": "openai-community/gpt2",
                                    }
    repo: dict = {"llama38b": "williamplacroix/llama-text-simplification",
                  "gpt2": "williamplacroix/text-simplification",
                  }
    repo_name: str = repo[base_model]
    model_name: str = base_model_aliases[base_model]
    if base_model == "llama38b":
        modules = ['lm_head', 'q_proj', 'k_proj', 'v_proj', 
                   'o_proj', 'gate_proj', 'up_proj', 'down_proj']
    else: # model_to_use == "gpt2":
        modules = ['lm_head', 'c_attn', 'c_fc', 'c_proj']

    #data_location = './data/wikilarge/'
    #train_texts = pd.read_pickle(f'{data_location}train_texts.pkl')
    data_location = hf_hub_download(repo_id="williamplacroix/wikilarge-graded", filename="train_texts.pkl", repo_type="dataset")
    train_texts = pd.read_pickle(data_location)
    print("train texts read in")

    if mode == "evens":
        train_texts = train_texts[train_texts['target_grade'] != 0]
        train_texts = train_texts[train_texts['target_grade'] % 2 == 0]
        print("dropped rows for odd grades and 0")
    elif mode == "all":
        train_texts = train_texts[train_texts['target_grade'] != 0]
        train_texts = train_texts[train_texts['target_grade'] != 1]
        train_texts = train_texts[train_texts['target_grade'] != 13]
        print("dropped rows for grades 0, 1, 13")

    grade_groups = train_texts.groupby(['target_grade'])

    datasets = []
    for grade, group in grade_groups:
        print(f"Creating dataset for grade {grade}")
        dataset = Dataset.from_pandas(group[['source', 'target', 'target_grade']])
        dataset = dataset.train_test_split(test_size=0.1, seed=42)
        datasets.append(dataset)
    print("datasets created")

    train_sets = [dataset["train"] for dataset in datasets]
    test_sets = [dataset["test"] for dataset in datasets]
    train_dataset = concatenate_datasets(train_sets)
    train_dataset = train_dataset.shuffle(seed=42)
    test_dataset = concatenate_datasets(test_sets)
    test_dataset = test_dataset.shuffle(seed=42)
    merged_dataset = DatasetDict({'train': train_dataset, 'test': test_dataset})
    print("datasets merged: ", merged_dataset)

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # * Below function tokenizes parallel corpus into target only inputs for fine-tuning

    def tokenize_function(examples):
        """
        TODO: Add docstring
        """
        return tokenizer(text=examples["target"],
                         text_target=examples["target"],
                         padding=True,
                         truncation=True,
                         max_length=1024,
                         return_tensors="pt")

    config = AutoConfig.from_pretrained(model_name)
    #quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                config=config,
                                                #quantization_config=quantization_config,
                                                low_cpu_mem_usage=True,
                                                device_map="auto",
                                                )
    #print(model)
    model.config.pad_token_id = tokenizer.eos_token_id

    lora_config = LoraConfig(
                            r=8,
                            lora_alpha=32,
                            target_modules=modules,
                            task_type="CAUSAL_LM",
                            lora_dropout=0.01,
                            )

    current_model_name = f"{base_model}-2-12-{mode}"

    wandb.init(project="Graded text simplification training", name=current_model_name)

    model = get_peft_model(model=model, peft_config=lora_config, adapter_name=current_model_name)
    print(model)
    model.print_trainable_parameters()
    model.config.pad_token_id = tokenizer.eos_token_id

    generation_config = GenerationConfig(max_length=256, 
                                            max_new_tokens=256,
                                            pad_token_id=tokenizer.eos_token_id,)
    generation_config.save_pretrained("./generation_config")
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    tokenized_dataset = merged_dataset.map(tokenize_function,
                                           batched=True,
                                           batch_size=32,
                                            remove_columns=['target_grade',
                                                            'target', 
                                                            'source', 
                                                            '__index_level_0__'])

    print(tokenized_dataset)

    data_collator = DataCollatorForSeq2Seq(model=model,
                                           tokenizer=tokenizer,
                                           padding="max_length",
                                           pad_to_multiple_of=8,
                                           max_length=128,
                                           label_pad_token_id=tokenizer.eos_token_id)

    print(f"Beginning training {current_model_name}")

    training_args = TrainingArguments(
        logging_strategy="epoch",
        save_strategy="epoch",
        eval_strategy="epoch",
        output_dir=repo_name,
        report_to="wandb",  # enable logging to W&B
        run_name=current_model_name,  # name of the W&B run (optional)
        logging_steps=1,  # how often to log to W&B
        save_safetensors=False, # this is a kludge fix for a bug in the transformers library
        learning_rate=1e-5,
        weight_decay=0.01,
        seed=42,
        num_train_epochs=5,
        load_best_model_at_end=True,
        remove_unused_columns=False,
    )

    training_args = training_args.set_dataloader(train_batch_size=16, eval_batch_size=16)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['test'],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.push_to_hub(f"Finished {base_model} 2-12 grades: {mode} pretraining")
    wandb.finish()

if __name__ == "__main__":

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints
    os.environ["HF_HOME"] = "/nethome/wlacroix/.cache/"
    os.environ["WANDB_DATA_DIR"] = "/nethome/wlacroix/"
    os.environ["WANDB_CACHE_DIR"] = "/nethome/wlacroix/.cache/"
    warnings.filterwarnings("ignore")
    load_dotenv()
    wandb.login(key=os.getenv("wandb"))
    login(token=os.getenv("huggingface"))#, add_to_git_credential=True)

    #all_evens = sys.argv[1]
    all_evens = "all"
    assert all_evens in {"all", "evens"}, "Invalid mode. Must be 'all' or 'evens'"
    main(all_evens, base_model="gpt2")
