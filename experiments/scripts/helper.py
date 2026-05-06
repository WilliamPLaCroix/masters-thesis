"""
Helper script for training and evaluating LoRA models.
Useful for running multiple training and evaluation runs in sequence.
Tasks include:
- "b": Pretraining the baseline model on even grade levels 2-12
- "t": Finetuning adapters on even grade levels 2-12
- "ta": Pretraining baseline and finetuning adapters on even grade levels 2-12: b + t
- "e": Evaluating models on unseen odd grade levels 3-11
- "em": Merging adjacent models and evaluating the merged models
- "ea": Evaluating all models individually and merged: e + em

args: -m / --mode = {"b", "t", "ta", "e", "ea", "em"}
return: None
"""
import os
import warnings
from itertools import product
from argparse import ArgumentParser, Namespace
from dotenv import load_dotenv
from huggingface_hub import login
import wandb
# from importlib import reload # Sometimes useful for rerunning modules for weird CUDA memory issues

# TODO refactor imports so that redundant imports are removed from the individual scripts and moved here

def pretrain_baseline() -> None:
    """
    Function pretrain_baseline() trains the even level baseline model.
    Calls lora_baseline_adapter.main(), which finetunes the pretrained LLM baseline
    (gpt2/llama3) on the even grade levels 2-12.
    "mode" currently only accepts "evens" as an argument, but lora_baseline_adapter.py
    also supports "all", which trains the model on odd and even grades 2-12.
    This mode is not being used in the current iteration, since reserving the odd datasets as 
    unseen test sets allows a sort of "out of domain" evaluation.
    args: None
    Only a single run, but with the full concatenated dataset, so it's a long one.

    args: None
    return: None
    """

    import lora_baseline_adapter
    modes: set = {"evens"}#, "all"
    for mode in modes:
        print("#"*50)
        print(f"Training baseline {mode}")
        lora_baseline_adapter.main(mode, base_model="gpt2")
        print("#"*50)
        print("Training complete")
        print("#"*50)

def finetune_adapters(odd_even=None) -> None:
    """
    Function finetunes adapters() takes the even level pretrained baseline
    and further finetunes it on the specified grade level.
    Runs the function for all even grade levels 2-12: 6 full training runs
    args: None
    return: None
    """
    assert odd_even in {"even", "odd", None}, "Invalid argument. Must be 'even' or 'odd'"

    import lora_finetune
    
    grades = {"even": set([2, 4, 6, 8, 10, 12]),
                "odd": set([3, 5, 7, 9, 11]),
                }
    if odd_even is None:
        model_grades: set = grades["even"] | grades["odd"]
    else:
        model_grades: set = grades[odd_even]

 
    for grade in model_grades:
        print("#"*50)
        print(f"LoRA run {grade}")
        lora_finetune.main(grade)
        print("#"*50)
        print("Training complete")
        print("#"*50)
        #reload(lora_finetune)

def evaluate() -> None:
    """
    Function evaluate() plots loss on unseen odd grade level datasets
    for all even grade levels 2-12. Model grade 1 is the all-evens baseline.
    Model grade 0 is the all-grades baseline, which is deprecated.
    Model grade -1 is the gpt2/llama3 base model.
    Runs the function for all even grade levels 2-12 against all odd grade levels 3-11
    model_grades * test_grades = 7 * 5 = 35 eval runs, in current iteration
    ----------
    args: None
    return: None
    """

    import eval
    # model_grades = {-1, 0, 1, 2, 4, 6, 8, 10, 12}
    # model_grades: set = {1, 2, 4, 6, 8, 10, 12}
    # test_set_grades: set = {3, 5, 7, 9, 11}
    model_grades: set = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
    test_set_grades: set = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}

    model_test_combos: product = product(model_grades, test_set_grades)
    runs: int = len(model_grades) * len(test_set_grades)

    for i, (model_grade, test_set_grade) in enumerate(model_test_combos):
        print("#"*50)
        print(f"LoRA run {i+1}/{runs}")
        eval.main(model_grade, test_set_grade)
        print("#"*50)
        print("Evaluation complete")
        print("#"*50)

def merge_eval() -> None:
    """
    Function merge_eval() evaluates adjacent merged models against their respective test sets.
    eg. when evaluating test set grade 3, model grades 2 and 4 are merged and evaluated.
    Current iteration also sets linear mixing proportions for the two models, 
    at ratios of 10:90, 30:70, 50:50, 70:30, 90:10.
    Runs the function for all odd grade levels 3-11, at 5 mixing proportions each
    test_grades * mix_proportion = 5 * 5 = 25 eval runs, in current iteration

    args: None
    return: None
    """

    import eval
    test_set_grades: set = {3, 5, 7, 9, 11}
    mixing_proportions: set = {1, 3, 5, 7, 9}
    model_test_combos: product = product(test_set_grades, mixing_proportions)
    runs: int = len(test_set_grades) * len(mixing_proportions)
    for i, (test_set_grades, mixing_proportions) in enumerate(model_test_combos):
        print("#"*50)
        print(f"Merge LoRA run {i+1}/{runs}")
        eval.main(test_set_grades, mixing_proportions) # TODO: Fix this function to accept the correct arguments
        print("#"*50)
        print("Evaluation complete")
        print("#"*50)

def main() -> None:
    """
    Responsible for parsing command line arguments and calling the appropriate function.
    argparse _should_ prevent invalid arguments from being passed.
    --mode ta pretrains a baseline model, followed immediately by the finetuned adapters.
    --mode ea evaluates all models individually, as well as merged.

    args: None
    return: None
    """

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints
    os.environ["WANDB_CACHE_DIR"]="/scratch/wlacroix/.cache/wandb/"
    os.environ["HF_HOME"]="/scratch/wlacroix/.cache/huggingface/"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    warnings.filterwarnings("ignore")
    load_dotenv()
    wandb.login(key=os.getenv("wandb"))
    login(token=os.getenv("huggingface"), add_to_git_credential=True)

    parser: ArgumentParser = ArgumentParser(
                prog='Text simplification helper script',
                description='Helper script for training and evaluating text simplification models',
                epilog='Enjoy training! :3')
    parser.add_argument('--helper',
                        type=str,
                        help='Must be "b", "to", "te", "ta", "e", or "em"',
                        dest='helper_mode',
                        required=True,
                        )
    args: Namespace = parser.parse_args()

    match args.helper_mode:
        case "to":
            finetune_adapters(odd_even="odd")
        case "te":
            finetune_adapters(odd_even="even")
        case "t":
            finetune_adapters()
        case "e":
            evaluate()
        case "em":
            merge_eval()
        case "ea":
            evaluate()
            merge_eval()
        case "b":
            pretrain_baseline()
        case "ta":
            pretrain_baseline()
            finetune_adapters()
        case _:
            print('Invalid mode. Must be "b", "t", "ta", "e", "ea", or "em"')

    # if args.mode == "t":
    #     finetune_adapters()
    # elif args.mode == "e":
    #     evaluate()
    # elif args.mode == "em":
    #     merge_eval()
    # elif args.mode == "ea":
    #     evaluate()
    #     merge_eval()
    # elif args.mode == "b":
    #     pretrain_baseline()
    # elif args.mode == "ta":
    #     pretrain_baseline()
    #     finetune_adapters()
    # else:
    #     print('Invalid mode. Must be "b", "t", "ta", "e", or "em"')

if __name__ == "__main__":
    main()
