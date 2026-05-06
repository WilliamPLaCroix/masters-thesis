"""
TODO: Add docstring
"""
import sys
import os
import warnings

from argparse import ArgumentParser, Namespace

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForSeq2Seq
#from transformers import BitsAndBytesConfig
# from peft import LoraModel, LoraConfig
# from peft import get_peft_model
from peft import PeftModel
from dotenv import load_dotenv
import wandb
from huggingface_hub import login

def main(model_grade, test_set_grade):#args) -> None:
    """
    TODO: Add docstring
    """

    scratch_path = "/scratch/wlacroix/.cache/huggingface/hub/"
    # model_grade: int = args[model_grade]
    # test_set_grade: int = args[test_set_grade]
    # model_a_proportion: int = args[model_a_proportion]
    # base_model: str = args[base_model]
    # merge: bool = args[merge]
    # merge_method: str = args[merge_method]
    model_a_proportion: int = 5
    base_model: str = "gpt2"
    merge: bool = False
    merge_method: str = "dare_ties"

    base_model_aliases: dict[str] = {"llama38b": "meta-llama/Meta-Llama-3-8B",
                                    "gpt2": "openai-community/gpt2",
                                    }
    repo: dict = {"llama38b": "williamplacroix/llama-text-simplification",
                  "gpt2": "williamplacroix/text-simplification",
                  }
    repo_name: str = repo[base_model]
    model_name: str = base_model_aliases[base_model]
    config = AutoConfig.from_pretrained(model_name)
    #quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                config=config,
                                                #quantization_config=quantization_config,
                                                low_cpu_mem_usage=True,
                                                )
    print("#"*50)
    print("Loaded base model")

    if merge is True:
        model_a_grade: int = test_set_grade-1
        model_a_proportion: float = round(model_a_proportion/10, 1)
        model_b_grade: int = test_set_grade+1
        model_b_proportion: float = round(1 - model_a_proportion, 1)
        weights: list[float] = [model_a_proportion, model_b_proportion]
        density: float = 0.5
        model_a_proportion: int = int(model_a_proportion*100)
        model_b_proportion: int = int(model_b_proportion*100)
        current_model_name: str = f"g{model_a_grade}-{model_a_proportion}_{merge_method}-d{density}_g{model_b_grade}-{model_b_proportion}_eval-on-g{test_set_grade}"
        print(f"Model name: {current_model_name}")
        print(f"Model proportions: {weights[0]}:{weights[1]}")
        model: PeftModel = PeftModel.from_pretrained(model,
                                      f"{repo_name}/{base_model}-grade-{test_set_grade-1}-finetuned",
                                      adapter_name="-1")
        print("Loaded trainable PeFT adapter -1")
        model.load_adapter(f"{repo_name}/{base_model}-grade-{test_set_grade+1}-finetuned",
                        adapter_name="+1")
        print("Loaded secondary PeFT adapter +1")
        adapters: list[str] = ["-1", "+1"]
        

        adapter_name: str = f"{model_a_grade}({model_a_proportion})+{model_b_grade}({model_b_proportion})={test_set_grade}"

        model.add_weighted_adapter(adapters,
                                weights,
                                adapter_name,
                                combination_type=merge_method,
                                density=density)
        model.set_adapter(adapter_name)

        print("Merged weighted adapters")
    elif model_grade == 1:
        adapters: str = f"{repo_name}/{base_model}-2-12-evens"
        model: PeftModel = PeftModel.from_pretrained(model, adapters)
        current_model_name: str = f"{base_model}-2-12-evens_eval-on-grade-{test_set_grade}"
    else: # * here's where the magic happens
        adapter_name: str = f""
        model_id = f"{scratch_path}{repo_name}/{base_model}-grade-{model_grade}-finetuned/"
        model = PeftModel.from_pretrained(model=model, 
                                        model_id=model_id, 
                                        #adapter_name=adapter_name,
                                        is_trainable=False,
                                        )
        print("Loaded PeFT model")
        current_model_name: str = f"{base_model}-grade-{model_grade}_eval-on-grade-{test_set_grade}"

    print("#"*50)
    print(model)

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.eos_token_id

    tokenized_dataset = load_dataset("williamplacroix/wikilarge-graded", f"grade-{test_set_grade}")

    print(f"Grade {test_set_grade}:", tokenized_dataset)

    data_collator = DataCollatorForSeq2Seq(model=model,
                                           tokenizer=tokenizer,
                                           padding="max_length",
                                           pad_to_multiple_of=8,
                                           max_length=128,
                                           label_pad_token_id=tokenizer.eos_token_id)

    print("data collated")
    wandb.init(project="Graded text simplification evaluation",
               group=f"Grade: {test_set_grade}",
               name=current_model_name)
    
    training_args = TrainingArguments(
        logging_strategy="epoch",
        save_strategy="epoch",
        eval_strategy="epoch",
        output_dir=f"/scratch/wlacroix/.cache/huggingface/hub/{repo_name}",
        report_to="wandb",  # enable logging to W&B
        run_name=current_model_name,  # name of the W&B run for logging
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
    
    print(f"Running evaluation on test_set_grade: {test_set_grade}, model: {current_model_name}")
    print("#"*50)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['test'],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    print("Begin evaluation :3")
    trainer.evaluate() #pylint: disable=this is a temporary fix
    wandb.finish()

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints
    warnings.filterwarnings("ignore")
    load_dotenv()
    wandb.login(key=os.getenv("wandb"))
    login(token=os.getenv("huggingface"), add_to_git_credential=True)

    parser: ArgumentParser = ArgumentParser(
                prog='Text simplification helper script',
                description='Helper script for training and evaluating text simplification models',
                epilog='Enjoy training! :)')
    parser.add_argument('--model_grade',
                        type=int,
                        help='Must be in {2, 4, 6, 8, 10, 12}',
                        dest='model_grade',
                        required=False,
                        default=2,
                        )
    parser.add_argument('--test_set_grade',
                        type=int,
                        help='Must be in {3, 5, 7, 9, 11}',
                        dest='test_set_grade',
                        required=False,
                        default=3,
                        )
    parser.add_argument('--model_a_proportion',
                        type=int,
                        help='Must be in {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}',
                        dest='model_a_proportion',
                        required=False,
                        default=5,
                        )
    parser.add_argument('--base_model',
                        type=str,
                        help='Must be "llama38b" or "gpt2"',
                        dest='base_model',
                        required=False,
                        default="llama38b",
                        )
    parser.add_argument('--merge',
                        type=bool,
                        help='Must be True or False',
                        dest='merge',
                        required=False,
                        default=False,
                        )
    parser.add_argument('--merge_method',
                        type=str,
                        help='Merge method to be used (eg. "linear", "dare_ties", etc.)',
                        dest='merge_method',
                        required=False,
                        default="dare_ties",
                        )

    args: Namespace = parser.parse_args()

    main(vars(args))
