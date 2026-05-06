import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from cal_ppl import calculate_ppl

def test_cross_grade_perplexity(
    model_name_or_path: str,
    checkpoint: str = "best",
    template: str = "default",
    batch_size: int = 32,
    cutoff_len: int = 2048,
    max_samples: int = None,
    save_path: str = "./results",
    test_name: str = "graded_adapters",
    split="train",
):
    """
    Test perplexity across different grade datasets and create a heatmap.
    
    Assumes dataset naming convention like: grade_2, grade_3, ..., grade_12
    """
    grades = list(range(2, 13))  # Grades 2-12
    # grades = list(range(2, 4))  # Grades 2-3 for quick testing
    n_grades = len(grades)
    
    # Initialize perplexity matrix
    ppl_matrix = np.zeros((n_grades, n_grades))
    
    # Create results directory
    os.makedirs(save_path, exist_ok=True)
    
    print(f"Testing {n_grades} grades against each other...")
    print(f"Grades: {grades}")

    adapter_mapping: dict = {
        "2": {"best": "/scratch/wlacroix/.cache/llama_factory/v3_grade02-adapter",
            "last": "/scratch/wlacroix/.cache/llama_factory/v3_grade02-adapter/checkpoint-920"},
        "3": {"best": "/scratch/wlacroix/.cache/llama_factory/v3_grade03-adapter",
            "last": "/scratch/wlacroix/.cache/llama_factory/v3_grade03-adapter/checkpoint-1170"},
        "4": {"best": "/scratch/wlacroix/.cache/llama_factory/v3_grade04-adapter",
            "last": "/scratch/wlacroix/.cache/llama_factory/v3_grade04-adapter/checkpoint-3520"},
        "5": {"best": "/scratch/wlacroix/.cache/llama_factory/v3_grade05-adapter",
            "last": "/scratch/wlacroix/.cache/llama_factory/v3_grade05-adapter/checkpoint-2630"},
        "6": {"best": "/scratch/wlacroix/.cache/llama_factory/v3_grade06-adapter",
            "last": "/scratch/wlacroix/.cache/llama_factory/v3_grade06-adapter/checkpoint-5510"},
        "7": {"best": "/scratch/wlacroix/.cache/llama_factory/v3_grade07-adapter",
            "last": "/scratch/wlacroix/.cache/llama_factory/v3_grade07-adapter/checkpoint-3930"},
        "8": {"best": "/scratch/wlacroix/.cache/llama_factory/v3_grade08-adapter",
            "last": "/scratch/wlacroix/.cache/llama_factory/v3_grade08-adapter/checkpoint-5020"},
        "9": {"best": "/scratch/wlacroix/.cache/llama_factory/v3_grade09-adapter",
            "last": "/scratch/wlacroix/.cache/llama_factory/v3_grade09-adapter/checkpoint-3210"},
        "10": {"best": "/scratch/wlacroix/.cache/llama_factory/v3_grade10-adapter",
            "last": "/scratch/wlacroix/.cache/llama_factory/v3_grade10-adapter/checkpoint-4430"},
        "11": {"best": "/scratch/wlacroix/.cache/llama_factory/v3_grade11-adapter",
            "last": "/scratch/wlacroix/.cache/llama_factory/v3_grade11-adapter/checkpoint-2250"},
        "12": {"best": "/scratch/wlacroix/.cache/llama_factory/v3_grade12-adapter",
            "last": "/scratch/wlacroix/.cache/llama_factory/v3_grade12-adapter/checkpoint-2780"},
        }
    
    # Iterate through all grade combinations
    for i, test_grade in enumerate(grades):
        for j, train_grade in enumerate(grades):
            print(f"\nTesting {checkpoint} model trained on grade {train_grade} dataset against grade {test_grade} dataset...")
            adapter_name_or_path = adapter_mapping[str(train_grade)][checkpoint]
            dataset_name = f"cleaned-grade{test_grade:02}"
            
            # Calculate perplexity
            avg_ppl = calculate_ppl(
                model_name_or_path=model_name_or_path,
                adapter_name_or_path=adapter_name_or_path,
                save_name=f"ppl_train_{train_grade}_test_{test_grade}.json",
                save_path=save_path,
                batch_size=batch_size,
                dataset=dataset_name,
                template=template,
                cutoff_len=cutoff_len,
                max_samples=max_samples,
                stage="pt",
                split=split,
            )
            
            ppl_matrix[i, j] = avg_ppl
            print(f"Grade {train_grade} -> Grade {test_grade}: PPL = {avg_ppl:.2f}")
                
    normed_matrix = normalize_matrix(ppl_matrix)
    # Save the matrix
    np.save(f"{save_path}/results/{checkpoint}_{test_name}_{split}_perplexity_matrix.npy", ppl_matrix)
    np.save(f"{save_path}/results/{checkpoint}_{test_name}_{split}_normalized_perplexity_matrix.npy", normed_matrix)
    
    # Create DataFrame and save
    grade_labels = [f"Grade {g}" for g in grades]
    df = pd.DataFrame(
        ppl_matrix,
        index=grade_labels,
        columns=grade_labels
    )
    
    # Save DataFrame
    df.to_csv(f"{save_path}/results/{checkpoint}_{test_name}_{split}_perplexity_matrix.csv")
    df.to_pickle(f"{save_path}/results/{checkpoint}_{test_name}_{split}_perplexity_matrix.pkl")  # Preserves data types
    
    # Also save as JSON for readability
    matrix_dict = {
        "grades": grades,
        "matrix": ppl_matrix.tolist(),
        "dataframe": df.to_dict(),  # Add DataFrame representation
        "description": "Perplexity matrix where matrix[i][j] is the perplexity of model trained on grade j tested on grade i dataset"
    }
    
    with open(f"{save_path}/results/{checkpoint}_{test_name}_{split}_perplexity_matrix.json", "w", encoding='utf-8') as f:
        json.dump(matrix_dict, f, indent=2)
    
    # Create heatmap (now returns DataFrame)

    
    df_result = create_perplexity_heatmap(ppl_matrix, 
                                          grades, 
                                          save_path, 
                                          save_name="absolute_perplexity_heatmap", 
                                          test_name=test_name, 
                                          split=split)
    normed_result = create_perplexity_heatmap(normed_matrix, 
                                              grades, 
                                              save_path, 
                                              save_name="normalized_perplexity_heatmap", 
                                              test_name=test_name, 
                                              split=split)
    
    return ppl_matrix, df_result

def test_base_model_perplexity(
    model_name_or_path: str,
    checkpoint="best",
    template: str = "default",
    batch_size: int = 32,
    cutoff_len: int = 2048,
    max_samples: int = None,
    save_path: str = "./results",
    test_name: str = "shared_baseline",
    split="train",
):
    """
    Test perplexity across different grade datasets and create a heatmap.
    
    Assumes dataset naming convention like: grade_2, grade_3, ..., grade_12
    """
    grades = list(range(2, 13))  # Grades 2-12
    # grades = list(range(2, 4))  # Grades 2-3 for quick testing
    n_grades = len(grades)
    
    # Initialize perplexity matrix
    ppl_matrix = np.zeros((n_grades))
    
    # Create results directory
    os.makedirs(save_path, exist_ok=True)
    
    print(f"Testing {n_grades} grades against each other...")
    print(f"Grades: {grades}")

    adapter_mapping: dict = {
        "shared_baseline": {"best": "/scratch/wlacroix/.cache/llama_factory/v3_baseline-adapter",
                            "last": "/scratch/wlacroix/.cache/llama_factory/v3_baseline-adapter/checkpoint-35370"},
        "off_the_shelf": {"best": None,
                          "last": None}
        }
    
    # Iterate through all grade combinations
    for i, test_grade in enumerate(grades):
        print(f"\nTesting model {checkpoint} {test_name} dataset against grade {test_grade} dataset...")
        adapter_name_or_path = adapter_mapping[test_name][checkpoint]
        dataset_name = f"cleaned-grade{test_grade:02}"
        
        # Calculate perplexity
        avg_ppl = calculate_ppl(
            model_name_or_path=model_name_or_path,
            adapter_name_or_path=adapter_name_or_path,
            save_name=f"ppl_{test_name}_test_{test_grade}.json",
            save_path=save_path,
            batch_size=batch_size,
            dataset=dataset_name,
            template=template,
            cutoff_len=cutoff_len,
            max_samples=max_samples,
            stage="pt",
            split=split,
        )
        
        ppl_matrix[i] = avg_ppl
        print(f"Shared baseline -> Grade {test_grade}_{split}: PPL = {avg_ppl:.2f}")
                
    # Save the matrix
    np.save(f"{save_path}/results/{checkpoint}_{test_name}_{split}_perplexity_matrix.npy", ppl_matrix)
    
    # Create DataFrame and save
    grade_labels = [f"Grade {g}" for g in grades]
    df = pd.DataFrame(
        ppl_matrix,
        index=grade_labels,
        columns=[test_name]
    )

    # Save DataFrame
    df.to_csv(f"{save_path}/results/{checkpoint}_{test_name}_{split}_perplexity_matrix.csv")
    df.to_pickle(f"{save_path}/results/{checkpoint}_{test_name}_{split}_perplexity_matrix.pkl")  # Preserves data types
    
    # Create heatmap (now returns DataFrame)
    df_result = create_baseline_perplexity_heatmap(ppl_matrix, 
                                                   grades, 
                                                   save_path, 
                                                   save_name="absolute_perplexity_heatmap", 
                                                   test_name=test_name,
                                                   split=split)
     
    return ppl_matrix, df_result

def normalize_matrix(input_matrix):
    norm_matrix = np.zeros_like(input_matrix)
    for i in range(input_matrix.shape[0]):
        row = input_matrix[i, :]
        min_val = np.nanmin(row)
        max_val = np.nanmax(row)
        norm_matrix[i, :] = (row - min_val) / (max_val - min_val)
    return norm_matrix

def create_baseline_perplexity_heatmap(ppl_matrix, 
                                       grades, 
                                       save_path, 
                                       save_name="perplexity_heatmap", 
                                       test_name="baseline", 
                                       split="train"):
    """Create and save a heatmap of the perplexity matrix using pandas DataFrame plotting."""
    
    # Convert to pandas DataFrame with proper labels
    grade_labels = [f"Grade {g}" for g in grades]
    df = pd.DataFrame(
        ppl_matrix,
        index=grade_labels,  # Test grades (Y-axis)
        columns=[test_name]
    )
    
    # Create heatmap using pandas style plotting
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Use pandas styler for heatmap-like visualization
    cax = ax.matshow(df.values, cmap='Reds')
    
    # Set ticks and labels
    ax.set_xticks(range(len(df.columns)))
    ax.set_yticks(range(len(df.index)))
    ax.set_xticklabels(df.columns, rotation=45)
    ax.set_yticklabels(df.index)
    
    # Move x-axis ticks to bottom
    ax.xaxis.set_ticks_position('bottom')
    
    # Add text annotations
    for (i, j), val in np.ndenumerate(df.values):
        if not pd.isna(val):
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', 
                   color='black', fontweight='bold', fontsize=9)
    
    # Add colorbar
    cbar = fig.colorbar(cax)
    cbar.set_label('Perplexity', rotation=270, labelpad=15)
    
    # Set labels and title
    plt.title(f'{checkpoint} {test_name} {split} grade_{split} dataset)',
              fontsize=14, pad=20)
    plt.ylabel(f'{split} Dataset Grade', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/{checkpoint}_{test_name}_{split}_{save_name}.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save the DataFrame as CSV for easy inspection
    df.to_csv(f"{save_path}/results/{checkpoint}_{test_name}_{split}_perplexity_matrix.csv")
    
    print(f"Heatmap saved to {save_path}/{checkpoint}_{test_name}_{split}_{save_name}.png")
    print(f"DataFrame saved to {save_path}/results/{checkpoint}_{test_name}_{split}_{save_name}.csv")
    
    return df

def create_perplexity_heatmap(ppl_matrix, 
                              grades, 
                              save_path, 
                              save_name="perplexity_heatmap", 
                              test_name="graded_adapters",
                              split="train"):
    """Create and save a heatmap of the perplexity matrix using pandas DataFrame plotting."""
    
    # Convert to pandas DataFrame with proper labels
    grade_labels = [f"Grade {g}" for g in grades]
    df = pd.DataFrame(
        ppl_matrix,
        index=grade_labels,  # Test grades (Y-axis)
        columns=grade_labels  # Training grades (X-axis)
    )
    
    # Create heatmap using pandas style plotting
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Use pandas styler for heatmap-like visualization
    cax = ax.matshow(df.values, cmap='Reds')
    
    # Set ticks and labels
    ax.set_xticks(range(len(df.columns)))
    ax.set_yticks(range(len(df.index)))
    ax.set_xticklabels(df.columns, rotation=45)
    ax.set_yticklabels(df.index)
    
    # Move x-axis ticks to bottom
    ax.xaxis.set_ticks_position('bottom')
    
    # Add text annotations
    for (i, j), val in np.ndenumerate(df.values):
        if not pd.isna(val):
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', 
                   color='black', fontweight='bold', fontsize=9)
    
    # Add colorbar
    cbar = fig.colorbar(cax)
    cbar.set_label('Perplexity', rotation=270, labelpad=15)
    
    # Set labels and title
    plt.title(f'{checkpoint} {test_name} {split} Cross-Grade Perplexity Matrix',
              fontsize=14, pad=20)
    plt.xlabel('Model Grade', fontsize=12)
    plt.ylabel(f'{split} Dataset Grade', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/{checkpoint}_{test_name}_{split}_{save_name}.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save the DataFrame as CSV for easy inspection
    df.to_csv(f"{save_path}/results/{checkpoint}_{test_name}_{split}_perplexity_matrix.csv")
    
    print(f"Heatmap saved to {save_path}/{test_name}_{split}_{save_name}.png")
    print(f"DataFrame saved to {save_path}/results/{checkpoint}_{test_name}_{split}_{save_name}.csv")
    
    return df

if __name__ == "__main__":
    # Example usage
    model_path = "/scratch/common_models/Llama-3.2-3B-Instruct-greedy"

    for checkpoint in ["best", "last"]:
        for dataset_split in ["train", "validation", "test"]:
            print(f"\nCalculating perplexity for {dataset_split} split of the base model...")

            matrix, df = test_cross_grade_perplexity(
                model_name_or_path=model_path,
                checkpoint=checkpoint,
                batch_size=32,  # Adjust based on your GPU memory
                max_samples=None,  # Limit samples for faster testing
                save_path="/nethome/wlacroix/LLaMA-Factory/experiments/logs/ppl",
                test_name="graded_adapters",
                split=dataset_split
            )

            matrix, df = test_base_model_perplexity(
                model_name_or_path=model_path,
                checkpoint=checkpoint,
                batch_size=32,  # Adjust based on your GPU memory
                max_samples=None,  # Limit samples for faster testing
                save_path="/nethome/wlacroix/LLaMA-Factory/experiments/logs/ppl",
                test_name="shared_baseline",
                split=dataset_split
            )

            matrix, df = test_base_model_perplexity(
                model_name_or_path=model_path,
                checkpoint=checkpoint,
                batch_size=32,  # Adjust based on your GPU memory
                max_samples=None,  # Limit samples for faster testing
                save_path="/nethome/wlacroix/LLaMA-Factory/experiments/logs/ppl",
                test_name="off_the_shelf",
                split=dataset_split
            )
            
            print("\nPerplexity Matrix:")
            print(matrix)
