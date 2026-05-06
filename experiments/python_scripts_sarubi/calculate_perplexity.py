import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

def calculate_perplexity(file_path, model_path, device="cuda", dtype=torch.bfloat16):
    """
    Calculate perplexity for a given text file using a specified LLaMA model.

    Args:
        file_path (str): Path to the input text file.
        model_path (str): Path to the LLaMA model.
        device (str): Device to use (default: "cuda").
        dtype (torch.dtype): Data type for model precision (default: torch.bfloat16).

    Returns:
        float: The perplexity of the text.
    """
    # Load the model and tokenizer
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype).to(device)
    # model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("Model loaded.")

    # Load and tokenize the input text
    print("Loading text file...")
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    encodings = tokenizer(text, return_tensors="pt")
    print("Text file loaded.")

    # Set up parameters
    max_length = model.config.max_position_embeddings  # Use model-specific max sequence length
    # stride = 512  # Define the stride length for overlapping evaluation
    stride = max_length  # Define the stride length for overlapping evaluation
    seq_len = encodings.input_ids.size(1)

    nlls = []  # Store negative log-likelihood values
    prev_end_loc = 0

    # Calculate perplexity
    print("Calculating perplexity...")
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # Target length for the current chunk
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100  # Ignore tokens outside the target window

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            # `outputs.loss` is CrossEntropyLoss averaged over valid labels
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    # Compute perplexity
    ppl = torch.exp(torch.stack(nlls).mean()).item()
    print(f"Perplexity: {ppl}")
    return ppl

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Calculate perplexity for a given text file and model.")
    parser.add_argument("--file", type=str, help="Path to the input text file.")
    parser.add_argument("--model", type=str, help="Path to the LLaMA model.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (default: cuda).")

    args = parser.parse_args()

    perplexity = calculate_perplexity(args.file, args.model, device=args.device)
    print(f"Final Perplexity: {perplexity}")
