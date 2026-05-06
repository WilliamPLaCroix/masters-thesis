import argparse
import json
import os
import re

def extract_rewritten_sentences(final_out):
    """
    Extract rewritten sentences from the given prompt and response.

    :param final_out: The response from the input JSON.
    :return: Extracted rewritten sentence(s).
    """
    # This pattern matches non-empty sequences inside curly braces
    pattern = r"\{([^}]+)\}"
    matches = re.findall(pattern, final_out)

    # Checking how many {} pairs are consecutively at the end
    end_pattern = r"(\{[^}]+\}\s*)+$"
    end_match = re.search(end_pattern, final_out)
    if end_match:
        # Counting the number of ending curly braces
        end_blocks = re.findall(pattern, end_match.group())
        # Join the end_blocks into a single string separated by a specified separator
        return " ".join(end_blocks)  # Join with a space or any other preferred separator
    else:
        # This pattern matches non-empty sequences inside curly braces
        pattern = r"\{([^}]+)\}"
        matches = re.findall(pattern, final_out)

        # Select the last match if there are any matches
        if matches:
            rewritten_sentence = matches[-1]  # Last match
        else:
            rewritten_sentence = ""
            # rewritten_sentence = extract_rewritten_sentences_from_double_quotes(final_out)
            # rewritten_sentence = final_out

        return rewritten_sentence


def process_file(input_file, output_dir):
    """
    Process a JSONL file, extract rewritten sentences, and save the output to a single text file.

    :param input_file: Path to the input JSONL file.
    :param output_dir: Directory to save the output text file.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Prepare the output file
    output_file = os.path.join(output_dir, "output.txt")

    with open(output_file, 'w+') as out_f:
        # Read and process each line in the JSONL file
        with open(input_file, 'r') as f:
            for line in f:
                obj = json.loads(line.strip())
                response = obj.get("predict", "")

                # Extract rewritten sentences using the existing function
                rewritten_sentences = extract_rewritten_sentences(response)

                if rewritten_sentences:
                    # Write the output to the single text file
                    out_f.write(rewritten_sentences + "\n")
                else:
                    # Write the original response to the single text file
                    out_f.write(" ".join(response.strip().splitlines()) + "\n")


def main():
    parser = argparse.ArgumentParser(description='Convert grade level text files to JSON format')
    parser.add_argument('--input_file', required=True, help='Input JSON file containing text objects')
    parser.add_argument('--output_dir', required=True, help='Output directory for text files')
    args = parser.parse_args()

    # Process the input JSON file
    process_file(args.input_file, args.output_dir)

if __name__ == '__main__':
    main()


# python PTA/python_scripts/extract_rewritten_sentences.py --input_file PTA/experiments/exp_1/test_on_original_model/original_model_generated_predictions.jsonl \
# --output_dir PTA/experiments/exp_1/test_on_original_model


# python PTA/python_scripts/extract_rewritten_sentences.py --input_file PTA/experiments/exp_1/test_full_pt/full_cpt_generated_predictions.jsonl \
# --output_dir PTA/experiments/exp_1/test_full_pt

