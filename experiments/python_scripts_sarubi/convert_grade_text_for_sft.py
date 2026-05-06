import os
import json
import argparse


# def convert_txt_to_json(src_txt_file, tgt_txt_file, json_path):
#     json_data = []
#     with open(src_txt_file, 'r', encoding='utf-8') as src_txt_file,\
#             open(tgt_txt_file, 'r', encoding='utf-8') as tgt_txt_file:
#         for src_line, tgt_line in zip(src_txt_file, tgt_txt_file):
#             if src_line.strip() and tgt_line.strip():
#                 json_data.append({"instruction": "Rewrite the following input text to make it easily understandable. "
#                                                  "Ensure that the rewritten sentence is grammatically correct, fluent, "
#                                                  "and retains the core message of the original sentence without changing its meaning. "
#                                                  "Always output Rewritten sentence(s) within curly braces.",
#                                   "input": src_line.strip(),
#                                   "output": tgt_line.strip()})
#
#     os.makedirs(os.path.dirname(json_path), exist_ok=True)
#     with open(json_path, 'w', encoding='utf-8') as json_file:
#         json.dump(json_data, json_file, indent=2, ensure_ascii=False)

def convert_txt_to_json(src_txt_file, tgt_txt_file, json_path):
    json_data = []
    with open(src_txt_file, 'r', encoding='utf-8') as src_txt_file,\
            open(tgt_txt_file, 'r', encoding='utf-8') as tgt_txt_file:
        for src_line, tgt_line in zip(src_txt_file, tgt_txt_file):
            if src_line.strip() and tgt_line.strip():
                json_data.append({"instruction": "Rewrite the following input text to make it easily understandable. "
                                                 "Ensure that the rewritten sentence is grammatically correct, fluent, "
                                                 "and retains the core message of the original sentence without changing its meaning. "
                                                 "Always output Rewritten sentence(s) within curly braces.",
                                  "input": f"Input text: {src_line.strip()}",
                                  "output": tgt_line.strip()})

    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, 'w', encoding='utf-8') as json_file:
        json.dump(json_data, json_file, indent=2, ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser(description='Convert grade level text files to JSON format')
    parser.add_argument('--input_dir', required=True, help='Input directory containing grade level text files')
    parser.add_argument('--output_dir', required=True, help='Output directory for JSON files')
    args = parser.parse_args()

    for grade in range(4, 5):
        src_txt_file = os.path.join(args.input_dir, f'grade-{grade}/grade_{grade}_original_parallel_train.src')
        tgt_txt_file = os.path.join(args.input_dir, f'grade-{grade}/grade_{grade}_original_parallel_train.tgt')
        json_file = os.path.join(args.output_dir, f'grade-{grade}/grade_{grade}_original_parallel_train.json')

        try:
            convert_txt_to_json(src_txt_file, tgt_txt_file, json_file)
            print(f'Converted {src_txt_file} to {json_file}')
        except FileNotFoundError:
            print(f'Warning: {src_txt_file} not found')
            continue

        # ----
        src_txt_file = os.path.join(args.input_dir, f'grade-{grade}/grade_{grade}_original_parallel_eval.src')
        tgt_txt_file = os.path.join(args.input_dir, f'grade-{grade}/grade_{grade}_original_parallel_eval.tgt')
        json_file = os.path.join(args.output_dir, f'grade-{grade}/grade_{grade}_original_parallel_eval.json')

        try:
            convert_txt_to_json(src_txt_file, tgt_txt_file, json_file)
            print(f'Converted {src_txt_file} to {json_file}')
        except FileNotFoundError:
            print(f'Warning: {src_txt_file} not found')
            continue

        # ----
        src_txt_file = os.path.join(args.input_dir, f'grade-{grade}/grade_{grade}_original_parallel_test.src')
        tgt_txt_file = os.path.join(args.input_dir, f'grade-{grade}/grade_{grade}_original_parallel_test.tgt')
        json_file = os.path.join(args.output_dir, f'grade-{grade}/grade_{grade}_original_parallel_test.json')

        try:
            convert_txt_to_json(src_txt_file, tgt_txt_file, json_file)
            print(f'Converted {src_txt_file} to {json_file}')
        except FileNotFoundError:
            print(f'Warning: {src_txt_file} not found')
            continue




if __name__ == '__main__':
    main()

# python PTA/python_scripts/convert_grade_text_for_sft.py --input_dir /Users/sarubi/Desktop/A8/code/2/3/LLM_based_control_rewrite/phase_2_experiments/data \
# --output_dir data/wikilarge/sft

