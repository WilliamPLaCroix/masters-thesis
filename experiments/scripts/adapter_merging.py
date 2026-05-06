import argparse
from transformers import AutoModelForCausalLM
from peft import PeftModel
import fire
import numpy as np


def select_adapters(target_grade="all",
                    adapter_path_format="/scratch/wlacroix/.cache/llama_factory/v3_grade{}-adapter",
                    weight_method="uniform",
                    merge_method="linear",
                    window_size=1,
                    weight_balance="average"):
    
    if target_grade == "all":
        window_size = 12  # include all grades
    selected, weights = select_and_weight_adapters(target_grade=target_grade,
                                                    window_size=window_size,
                                                    weight_method=weight_method,
                                                    weight_balance=weight_balance)
    grades = [str(f'{grade:02}') for grade in selected]
    adapters = [adapter_path_format.format(grade) for grade in grades]
    assert len(adapters) == len(grades), "Adapters, grades, and weights must have the same length"
    return adapters, grades, weights

def select_and_weight_adapters(
    target_grade,
    window_size,
    weight_method,
    weight_balance):
    """
    Sliding-window selection and weighting for models numbered 2..12 (inclusive).

    Args:
        target_grade: center of the window; accepts int 2..12 or zero-padded str like "02".
        window_size: how many grades to include on each side of the target (>= 0).
        weight_method:
            - "uniform": every selected adapter weight == 1, by definition.
            - "proximity": weights decrease with absolute distance to target and
              are normalized so the average weight across the selection equals 1.

    Returns:
        (selected_grades, weights), where:
            - selected_grades is an ascending list of ints within [2, 12], clipped at bounds.
            - weights has the same length as selected_grades.

    Behavior notes:
        - The window always centers on target, then clips at 2 and 12 as needed.
        - For "proximity", raw weights use an inverse-distance profile: 1 / (1 + |g - target|).
          We then scale so that mean(weight) == 1 (i.e., sum == len(selection)).
    """
    # Parse and validate the target
    if isinstance(target_grade, str):
        if not target_grade.isdigit():
            raise ValueError("target_grade must be an int 2..12 or a zero-padded numeric string like '02'.")
        target = int(target_grade)
    else:
        target = int(target_grade)

    if not (2 <= target <= 12):
        raise ValueError("target_grade must be in [2, 12].")
    
    if len(str(window_size).split('-')) > 1:
        n = int(str(window_size).split('-')[0])
        selected = [target] * n
        weights = [1.0] * n
        return selected, weights


    # Compute clipped window
    lo = max(2, target - window_size)
    hi = min(12, target + window_size)
    selected = list(range(lo, hi + 1))

    # Weights
    if weight_balance == "average":
        n = len(selected)
        if weight_method == "uniform":
            weights = [1.0] * n
        elif weight_method == "doubled":
            weights = [2.0] * n
        elif weight_method == "tripled":
            weights = [3.0] * n
        elif weight_method == "halved":
            weights = [0.5] * n
        elif weight_method == "random-1":
            # random weights that average to 1
            weights = [round(w,2).item() for w in np.random.dirichlet(np.ones(n),size=1)[0]*n]
        elif weight_method == "random-2":
            # random weights that average to 1
            weights = [round(w*2,2).item() for w in np.random.dirichlet(np.ones(n),size=1)[0]*n]
        elif weight_method == "random-3":
            # random weights that average to 1
            weights = [round(w*3,2).item() for w in np.random.dirichlet(np.ones(n),size=1)[0]*n]
        elif weight_method == "proximity":
            # Inverse-distance raw weights, target gets highest value
            raw = [1.0 / (1.0 + abs(g - target)) for g in selected]
            scale = n / sum(raw)  # ensures average == 1
            weights = [round(w * scale, 2) for w in raw]
        elif weight_method == "proximity-squared":
            # Inverse-distance-squared raw weights, target gets highest value
            raw = [1.0 / ((1.0 + abs(g - target)) ** 2) for g in selected]
            scale = n / sum(raw)  # ensures average == 1
            weights = [round(w * scale, 2) for w in raw]
        elif weight_method == "proximity-cubed":
            raw = [1.0 / ((1.0 + abs(g - target)) ** 3) for g in selected]
            scale = n / sum(raw) # ensures average == 1
            weights = [round(w * scale, 2) for w in raw]
        elif weight_method == "proximity-flatter":
            raw = [1.0 / (1.0 + abs(g - target)/4) for g in selected]
            scale = n / sum(raw)  # ensures average == 1
            weights = [round(w * scale, 2) for w in raw]
        else:
            raise ValueError("weight_method must be in {'uniform', 'proximity', 'proximity-squared', 'proximity-cubed', 'proximity-flatter'}.")
    elif weight_balance == "sum":
        if weight_method == "uniform":
            weights = [n / len(selected) for n in [1.0] * len(selected)]
        elif weight_method == "proximity":
            weights = [1.0 / (1.0 + abs(g - target)) for g in selected]
            weights = [w / sum(weights) for w in weights]
        else:
            raise ValueError("weight_method must be 'uniform' or 'proximity'.")
    elif weight_balance == "broken":
        # 'broken' window includes grades +- but no target grade
        selected.remove(target)
        weights = [1.0] * len(selected)
    elif weight_balance == "downshifted-1":
        # downshifted window lowers the grade window by 1
        target -= 1
        lo = max(2, target - window_size)
        hi = min(12, target + window_size)
        selected = list(range(lo, hi + 1))
        weights = [1.0] * len(selected)
        if len(selected) == 0 or len(weights) == 0:
            selected = [max(target, 2)]
            weights = [1.0]
    elif weight_balance == "downshifted-2":
        # downshifted window lowers the grade window by 1
        target -= 2
        lo = max(2, target - window_size)
        hi = min(12, target + window_size)
        selected = list(range(lo, hi + 1))
        weights = [1.0] * len(selected)
        if len(selected) == 0 or len(weights) == 0:
            selected = [max(target, 2)]
            weights = [1.0]
    elif weight_balance == "upshifted-1":
        # downshifted window lowers the grade window by 1
        target += 1
        lo = max(2, target - window_size)
        hi = min(12, target + window_size)
        selected = list(range(lo, hi + 1))
        weights = [1.0] * len(selected)
        if len(selected) == 0 or len(weights) == 0:
            selected = [min(target, 12)]
            weights = [1.0]
    elif weight_balance == "upshifted-2":
        # downshifted window lowers the grade window by 1
        target += 2
        lo = max(2, target - window_size)
        hi = min(12, target + window_size)
        selected = list(range(lo, hi + 1))
        weights = [1.0] * len(selected)
        if len(selected) == 0 or len(weights) == 0:
            selected = [min(target, 12)]
            weights = [1.0]
    

    # Fallback to at least the target adapter if none selected


    return selected, weights


def merge_adapters(model="/scratch/common_models/Llama-3.2-3B-Instruct-greedy",
         merge_method="dare_ties",
         target_grade="all",
         weight_method="uniform",
         weight_balance="average",
         density=None,
         majority_sign_method="total",
         output="/scratch/wlacroix/.cache/llama_factory",
         window_size=1,
         project_version="v3"
         ):

    adapter_path_format=f"/scratch/wlacroix/.cache/llama_factory/{project_version}"+"_grade{}-adapter"
    adapters, grades, weights = select_adapters(target_grade=target_grade, 
                                                adapter_path_format=adapter_path_format,
                                                weight_method=weight_method,
                                                merge_method=merge_method,
                                                window_size=window_size,
                                                weight_balance=weight_balance)

    print(f"model: {model}")
    print(f"target_grade: {target_grade}")
    print(f"weight_method: {weight_method}")
    print(f"grades: {grades}")

    print("Loading base model from:", model)
    base_model = AutoModelForCausalLM.from_pretrained(model, device_map="auto")
    print("Loading adapter from:", adapters[0])
    # time adapter loading
    import time
    start = time.time()
    model = PeftModel.from_pretrained(base_model, adapters[0], adapter_name=grades[0])

    # load remaining adapters
    for adapter_path, grade in zip(adapters[1:], grades[1:]):
        print("Loading adapter from:", adapter_path, "as", grade)
        _ = model.load_adapter(adapter_path, adapter_name=grade)
    loaded = time.time() - start
    
    merged_adapter_name = f"{project_version}_merge@{merge_method}_grade@{target_grade}_window@{window_size}_weight@{weight_method}-{weight_balance}"

    print(f"Merging adapters into new adapter: {merged_adapter_name}")
    # set default density if needed
    if merge_method in {"ties", "ties_svd", "dare_ties", "dare_linear", "dare_ties_svd", "dare_linear_svd", "magnitude_prune", "magnitude_prune_svd"} and density is None:
        density = 0.5  # default density for these methods
    else:
        pass  # density remains None or as provided

    # set majority_sign_method only for relevant methods
    if merge_method in {"ties", "dare_ties", "dare_ties_svd"}:
        majority_sign = majority_sign_method
    else:
        majority_sign = None  # not used for other methods
    model.add_weighted_adapter(adapters=grades, weights=weights, combination_type=merge_method, adapter_name=merged_adapter_name, density=density, majority_sign_method=majority_sign)

    
    print(f"weights: {weights}")
    print(f"output: {output}")
    print(f"merge_method: {merge_method}")
    print(f"density: {density}")

    merged = time.time() - start - loaded
    model.set_adapter(merged_adapter_name)
    print(model.peft_config.keys())
    # # clean up unused adapters
    # for grade in grades:
    #     model.delete_adapter(grade)
    
    cleaned = time.time() - start - loaded - merged
    total = time.time() - start
    print(f"Loaded adapters in {loaded:.2f}s")
    print(f"Merged adapters in {merged:.2f}s")
    print(f"Cleaned up extra adapters in {cleaned:.2f}s")
    print(f"Total time for adapter loading, merging, cleaning: {total:.2f}s")
    model.save_pretrained(f"{output}", selected_adapters=[merged_adapter_name])
    print(f"Saved merged adapter to {output}/{merged_adapter_name}")

if __name__ == "__main__":
    fire.Fire(merge_adapters)

