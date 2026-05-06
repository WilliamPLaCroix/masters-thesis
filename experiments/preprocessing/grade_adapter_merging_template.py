import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import PeftModel, LoraModel, add_weighted_adapter
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from llamafactory.trainer import SFTTrainer  # adjust if different

class GradeAdapterWeightPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 11)  # Grades 2-12

    def forward(self, grade_scalar):
        x = (grade_scalar - 2) / 10.0
        x = self.linear(x.unsqueeze(-1))
        return F.softmax(x, dim=-1)

class GradeSimplifier(nn.Module):
    def __init__(self, base_model_name="t5-small"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
        self.model = PeftModel(self.model, LoraModel)

        self.adapter_names = [f"grade{g}" for g in range(2, 13)]
        for name in self.adapter_names:
            self.model.load_adapter(name, adapter_name=name)

        self.weight_predictor = GradeAdapterWeightPredictor()

    def forward(self, input_ids, attention_mask, labels, grade_level):
        adapter_weights = self.weight_predictor(grade_level)
        add_weighted_adapter(
            model=self.model,
            adapter_names=self.adapter_names,
            weights=adapter_weights.squeeze(0).tolist(),
            combination_type="dare_ties",
            density=0.2,
            adapter_name="merged_adapter"
        )
        self.model.set_adapter("merged_adapter")

        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

# Custom trainer wrapper
class GradeTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        grade_level = inputs.pop("grade_level")
        outputs = model(**inputs, grade_level=grade_level)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

# Usage (CLI will call GradeTrainer if specified via --train_script)
def build_model_and_trainer():
    model = GradeSimplifier()
    return GradeTrainer(model=model)
