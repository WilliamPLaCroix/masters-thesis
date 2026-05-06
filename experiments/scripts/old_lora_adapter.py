import pandas as pd
from datasets import Dataset
from datasets import load_metric
from evaluate import load

from transformers import TrainingArguments, Seq2SeqTrainingArguments
from transformers import AutoTokenizer
from transformers import AutoConfig
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq
from transformers import Trainer, Seq2SeqTrainer
from transformers import GenerationConfig
from transformers import EarlyStoppingCallback
from peft import LoraModel, LoraConfig, get_peft_model, TaskType
import sys
import torch
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
from dotenv import load_dotenv
load_dotenv()
import wandb

wandb.login(key=os.getenv("wandb"))
os.environ["WANDB_PROJECT"] = "Graded text simplification training"  # name your W&B project
os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints

from huggingface_hub import login
login(token=os.getenv("huggingface"), add_to_git_credential=True)

from evaluate import load
sari = load("sari")


data_location = './data/wikilarge/'


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = "openai-community/gpt2"



def main():

    N = int(sys.argv[1])

    train_texts = pd.read_pickle(f'{data_location}train_texts.pkl')
    print("train texts read in")

    grade_groups = train_texts.groupby(['target_grade'])

    datasets = {}
    for i, (grade, group) in enumerate(grade_groups):
        datasets[i] = Dataset.from_pandas(group[['source', 'target', 'target_grade']]).train_test_split(test_size=0.1)
    print("datasets created")
    
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")#, pad_token="eos_token") #pad_token_id=tokenizer.pad_token_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    

    """
    Below function tokenizes parallel corpus into source:target pairs for Seq2Seq training
    """
    # def tokenize_function(examples):
    #     return tokenizer(text=examples["source"], text_target=examples['target'], padding=True, truncation=True, max_length=1024, return_tensors="pt")

    """
    Below function tokenizes parallel corpus into target only inputs for unsupervised fine-tuning
    """
    def tokenize_function(examples):
        return tokenizer(text=examples["target"], text_target=examples["target"], padding=True, truncation=True, max_length=1024, return_tensors="pt")


    def compute_metrics(prediction):
        metric_name = "sari"
        metric = load(metric_name)

        labels_ids = prediction.label_ids
        pred_ids = prediction.predictions
        input_ids = prediction.inputs

        # all unnecessary tokens are removed
        label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
        references = [[reference] for reference in label_str]
        source_str = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        predictions_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

        score = metric.compute(sources=source_str, predictions=predictions_str, references=references)
        return score

    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, config=config)
    print(model)
    model.config.pad_token_id = tokenizer.eos_token_id

    lora_config = LoraConfig(task_type = "SEQ_2_SEQ_LM",
                            r=8,
                            lora_alpha=32,
                            target_modules=['lm_head'],
                            lora_dropout=0.01,
                            )
    # lora_model = LoraModel(model, lora_config, f"gpt2-grade-{N}")
    lora_model = get_peft_model(model, lora_config, f"gpt2-grade-{N}")
    lora_model.print_trainable_parameters()
    lora_model.config.pad_token_id = tokenizer.eos_token_id

    generation_config = GenerationConfig(max_length=256, 
                                            max_new_tokens=256,
                                            pad_token_id=tokenizer.eos_token_id,)
    generation_config.save_pretrained("./generation_config")
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    lora_model.generation_config.pad_token_id = tokenizer.pad_token_id

    tokenized_dataset = datasets[N].map(tokenize_function, batched=True, batch_size=32,
                                    remove_columns=['target_grade','source', 'target', '__index_level_0__'])
    print(tokenized_dataset)


    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding="max_length", max_length=128, label_pad_token_id=tokenizer.eos_token_id)

    #train_data_loader = torch.utils.data.DataLoader(tokenized_dataset['train'], batch_size=32, shuffle=True, collate_fn=data_collator)
    # eval_data_loader = torch.utils.data.DataLoader(tokenized_dataset['test'], batch_size=training_args.batch_size, shuffle=False, collate_fn=data_collator)
    # dataloaders = {'train': train_data_loader, 'eval': eval_data_loader}

    # for batch in train_data_loader:
    #     print(batch.keys())
    #     print(batch['input_ids'].shape)
    #     print(batch['attention_mask'].shape)
    #     print(batch['labels'].shape)
    #     print(batch['target_grade'].shape)
    #     break
    print("data collated")

    current_model_name = f"gpt2-grade-{N}"

    training_args = Seq2SeqTrainingArguments(
        save_strategy="epoch",
        output_dir=f"./models/{current_model_name}",
        report_to="wandb",  # enable logging to W&B
        run_name=current_model_name,  # name of the W&B run (optional)
        logging_steps=1,  # how often to log to W&B
        #hub_model_id="williamplacroix/text-simplification",  # save the model to the Hub after training
        overwrite_output_dir=True,
        save_safetensors=False, # this is a temporary fix for a bug in the transformers library
        #save_only_model=True,
        save_total_limit=1,
        eval_strategy="epoch",
        learning_rate=1e-5,
        weight_decay=0.01,
        seed=42,
        num_train_epochs=5,
        load_best_model_at_end=True,
        prediction_loss_only=True,
        metric_for_best_model="loss",
        #greater_is_better=False,
        label_names=["labels"],
        include_inputs_for_metrics=True,
        predict_with_generate=True,
        generation_config="./generation_config/generation_config.json",
        generation_max_length=256,
        remove_unused_columns=False,
    )
    
    trainer = Seq2SeqTrainer(
        model=lora_model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['test'],
        data_collator=data_collator,
        #callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        #compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        
    )

    trainer.train()
    trainer.push_to_hub(current_model_name)
    return

if __name__ == "__main__":
    main()
