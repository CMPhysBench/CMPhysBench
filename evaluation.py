import json
import re
import os
import sys
import time
import logging
import argparse
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import torch
from datasets import load_dataset, load_from_disk
from vllm import LLM, SamplingParams
from collections import defaultdict


MAX_TOKEN = 16384

SYSTEM_PROMPT = (
    "You are a condensed matter physics expert. Please read the following question and provide a step-by-step solution "
    "using only the given symbols. Do not introduce any new symbols that are not provided in the problem statement. "
    "Please place your reasoning inside <think>...</think> tags. "
    "Your final answer must be presented as a readable LaTeX formula, enclosed in a \\boxed{} environment."
)

try:
    from SEED.SEED import SEED
except ImportError:
    print("Error: Could not import the SEED module.")
    print("Please ensure the 'SEED' folder is in the same directory as this script, or add its path to PYTHONPATH.")
    sys.exit(1)

# ==============================================================================
# ===== Step 1: Inference Functions ============================================
# ==============================================================================
def run_inference(args, dataset, inference_output_path):
    """
    Performs inference on the dataset using vLLM and saves the results.
    """
    print(f"## Model: {args.model_path}")
    print(f"## Inference rounds (pass@k): k={args.k}")

    # --- Initialize vLLM ---
    llm = LLM(
        model=args.model_path,
        tokenizer=args.model_path,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="auto",
        enforce_eager=True, 
    )
    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        max_tokens=MAX_TOKEN,
        n=args.k # Use the 'n' parameter to generate k completions for each prompt
    )

    completed_records = {}
    if os.path.exists(inference_output_path):
        with open(inference_output_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    idx = data.get("index")
                    llm_answer = data.get("llm_answer", [])
                    if idx is not None and isinstance(llm_answer, list) and len(llm_answer) == args.k:
                        completed_records[idx] = data
                except json.JSONDecodeError:
                    logging.warning(f"Skipping malformed line in existing file: {line.strip()}")
    
    tasks = [item for item in dataset if item.get("id") not in completed_records]
    
    print(f"# Total items in dataset: {len(dataset)}")

    if not tasks:
        return inference_output_path

    prompts = []
    tokenizer = llm.get_tokenizer()
    for item in tqdm(tasks, desc="Constructing Prompts"):
        question = (item.get("context", "") or "") + (item.get("question", "") or "")
        symbol_dict = item.get("symbol", "")
        symbol_prompt = "Here are the relevant symbols:\n" + symbol_dict
        full_input = question + "\n" + symbol_prompt

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": full_input}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)

    print(f"## Starting batch inference (Batch Size: {args.batch_size})...")
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)

    for item, output in zip(tasks, outputs):
        answers = [out.text.strip() for out in output.outputs]
        record = {
            "index": item["id"],
            "question": (item.get("context", "") or "") + (item.get("question", "") or ""),
            "gt_answer": item.get("final_answer"),
            "llm_answer": answers,
            "topic": item.get("topic"),
            "answer_type": item.get("answer_type"),
        }
        completed_records[item["id"]] = record

    print(f"## Writing results to: {inference_output_path}")
    sorted_records = sorted(completed_records.values(), key=lambda r: r['index'])
    with open(inference_output_path, 'w', encoding='utf-8') as f:
        for r in sorted_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"# Inference complete!")
    return inference_output_path

# ==============================================================================
# ===== Step 2: Evaluation Functions ===========================================
# ==============================================================================

def format_duration(seconds):
    """Formats seconds into a readable M:S or H:M:S string."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
    return f"{int(minutes)}m {int(seconds)}s"

def extract_boxed_latex(text):
    """Extracts content from a \\boxed{...} environment in a string."""
    if not isinstance(text, str): return ""
    start = text.find(r'\boxed{')
    if start == -1:
        return ""
    start += len(r'\boxed{')
    
    depth = 1
    end = start
    while end < len(text):
        if text[end] == '{':
            depth += 1
        elif text[end] == '}':
            depth -= 1
            if depth == 0:
                return text[start:end].strip()
        end += 1
    return ""  

def score_one_safe(item):
    """A safe scoring function for a single item, designed for multiprocessing."""
    index = item.get("index", -1)
    gt = item.get("gt_answer", "")[0]
    pred_list = item.get("llm_answer", [])
    answer_type = item.get("answer_type", "Expression")
    topic = item.get("topic", "Unknown") # Get the topic

    score_list = []
    try:
        for pred in pred_list:
            if not gt or not pred:
                score_list.append(0)
                continue
            try:
                score, _, _, _ = SEED(gt, pred, answer_type)
                score_list.append(score)
            except Exception:
                score_list.append(0)
        return {"index": index, "score_list": score_list, "topic": topic}
    except Exception as e:
        return {"index": index, "score_list": [0] * len(pred_list), "error": str(e), "topic": topic}
    

def analyze_and_print_results(scored_data, model_name):
    """
    Reads scored data, calculates statistics, and prints them in a clean,
    aligned table format to both the console and the log file.
    """
    topic_groups = defaultdict(list)
    max_round_count = 0
    for item in scored_data:
        topic = item.get("topic") or "Uncategorized"
        topic_groups[topic].append(item)
        max_round_count = max(max_round_count, len(item.get("score_list", [])))

    def _calculate_stats(items):
        stats = {}
        for i in range(max_round_count):
            pass_k = i + 1
            scores = [item['score_list'][i] for item in items if len(item.get('score_list', [])) > i]
            
            if not scores:
                stats[f'P@{pass_k} SEED'] = 'N/A'
                stats[f'P@{pass_k} ACC'] = 'N/A'
                continue

            total = len(scores)
            avg_score = sum(scores) / total
            passed = sum(1 for s in scores if abs(s - 100.0) < 1e-4)
            pass_rate = passed / total
            
            stats[f'P@{pass_k} SEED'] = f"{avg_score:.2f}"
            stats[f'P@{pass_k} ACC'] = f"{pass_rate:.2%}"
        return stats

    table_data = []
    for topic in sorted(topic_groups.keys()):
        items = topic_groups[topic]
        row_data = {"Topic": topic, "Count": len(items)}
        row_data.update(_calculate_stats(items))
        table_data.append(row_data)
        
    if len(topic_groups) > 1:
        separator_row = {key: "---" for key in table_data[0].keys()}
        table_data.append(separator_row)

    overall_data = {"Topic": "Overall", "Count": len(scored_data)}
    overall_data.update(_calculate_stats(scored_data))
    table_data.append(overall_data)

    if not table_data:
        logging.info("No data to display in the final analysis.")
        return

    column_order = ["Topic", "Count"]
    for i in range(max_round_count):
        column_order.append(f"P@{i+1} SEED")
        column_order.append(f"P@{i+1} ACC")

    col_widths = {key: len(key) for key in column_order}
    for row in table_data:
        for key, value in row.items():
            col_widths[key] = max(col_widths.get(key, 0), len(str(value)))

    header = " | ".join(key.ljust(col_widths[key]) for key in column_order)
    separator = "-+-".join("-" * col_widths[key] for key in column_order)
    
    final_output_lines = []
    final_output_lines.append("=" * len(header))
    final_output_lines.append(f"Final Results Analysis for: {model_name}")
    final_output_lines.append("=" * len(header))
    final_output_lines.append(header)
    final_output_lines.append(separator)

    for row in table_data:
        line = " | ".join(str(row.get(key, '')).ljust(col_widths[key]) for key in column_order)
        final_output_lines.append(line)
        
    final_output_lines.append("=" * len(header))
    final_output_string = "\n".join(final_output_lines)
    
    logging.info("\n" + final_output_string)
      
def run_evaluation(inference_output_file, output_dir):
    """
    Reads the inference output, enriches it with extracted answers and scores,
    saves a single comprehensive result file, and prints the analysis.
    """
    try:
        all_records = []
        items_for_scoring = []
        with open(inference_output_file, "r", encoding="utf-8") as infile:
            for line in infile:
                if not line.strip(): continue
                item = json.loads(line)
                
                boxed_list = [extract_boxed_latex(ans) for ans in item.get("llm_answer", [])]
                item['boxed_answer'] = boxed_list
                
                all_records.append(item)
                
                items_for_scoring.append({
                    "index": item.get("index"),
                    "gt_answer": item.get("gt_answer", ""),
                    "llm_answer": boxed_list, 
                    "answer_type": item.get("answer_type", "Expression"),
                    "topic": item.get("topic")
                })

        scored_results = []
        with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
            with tqdm(total=len(items_for_scoring), desc="   - Scoring items", unit="item") as pbar:
                futures = [executor.submit(score_one_safe, item) for item in items_for_scoring]
                for future in as_completed(futures):
                    scored_results.append(future.result())
                    pbar.update(1)


        score_map = {res['index']: res['score_list'] for res in scored_results}
        for record in all_records:
            record['score_list'] = score_map.get(record['index'], [0] * len(record.get('llm_answer', [])))

        all_records.sort(key=lambda x: x.get('index', -1))

        base_name = Path(inference_output_file).stem
        final_results_path = os.path.join(output_dir, f"{base_name}_final_results.json")
        with open(final_results_path, "w", encoding="utf-8") as f:
            json.dump(all_records, f, ensure_ascii=False, indent=2)

        analyze_and_print_results(all_records, base_name)

    except Exception as e:
        print(f"A critical error occurred during evaluation: {e}")
        logging.error(f"Failed to evaluate {inference_output_file}: {e}", exc_info=True)
        
        
def main():
    parser = argparse.ArgumentParser(description="End-to-end inference and evaluation pipeline for CMPhysBench.")
    parser.add_argument('--model-path', type=str, required=True, help="Required: Path to a vLLM-compatible model.")
    parser.add_argument('--tensor-parallel-size', type=int, default=None, help="Tensor parallel size for vLLM. Defaults to the number of available GPUs.")
    parser.add_argument('--k', type=int, default=1, help="Number of inference rounds (for pass@k).")
    parser.add_argument('--batch-size', type=int, default=100, help="Batch size for inference.")
    parser.add_argument('--output-dir', type=str, default="./output", help="Root directory to store all outputs (inference results, final scores, logs).")
    args = parser.parse_args()

    if args.tensor_parallel_size is None:
        args.tensor_parallel_size = torch.cuda.device_count()

    model_name = Path(args.model_path).name
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_folder = os.path.join(args.output_dir, f"{model_name}_pass@{args.k}_{timestamp}")
    os.makedirs(output_folder, exist_ok=True)
    
    log_file_path = os.path.join(output_folder, "run.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=log_file_path,
        filemode='w' 
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    script_start_time = time.monotonic()
    
    print(f"# All outputs will be saved in: {output_folder}")
    logging.info(f"Script started with args: {vars(args)}")

    local_data_path = "./CMPhysBench" # Defines a folder in the same directory as the script

    try:
        if os.path.exists(local_data_path):
            hf_dataset = load_from_disk(local_data_path)
        else:
            print(f"Local dataset not found. Downloading from Hugging Face Hub...")
            hf_dataset = load_dataset("weidawang/CMPhysBench", split="train")
            hf_dataset.save_to_disk(local_data_path)
            print(f"Dataset saved locally to '{local_data_path}'.")

        initial_dataset = list(hf_dataset)
        original_count = len(initial_dataset)

        # Filter out records where 'final_answer' is missing or an empty list
        dataset = [item for item in initial_dataset if item.get("final_answer")]
        
        cleaned_count = len(dataset)
        removed_count = original_count - cleaned_count
        
        if removed_count > 0:
            print(f"Cleaned dataset: Removed {removed_count} records with an empty 'final_answer' field.")
        print(f"Final record count for processing: {cleaned_count}.")

    except Exception as e:
        print(f"Failed to load or download the dataset: {e}")
        logging.error(f"A critical error occurred during dataset loading: {e}")
        return
    
    inference_output_file = os.path.join(output_folder, f"{model_name}-pass@{args.k}.jsonl")
    try:
        run_inference(args, dataset, inference_output_file)
    except Exception as e:
        print(f"A fatal error occurred during inference: {e}")
        logging.critical(f"Inference failed catastrophically: {e}", exc_info=True)
        return 

    run_evaluation(inference_output_file, output_folder)

    script_duration = time.monotonic() - script_start_time
    total_time_msg = f"All tasks completed! Total execution time: {format_duration(script_duration)} 🎉🎉🎉"
    print(f"\n{total_time_msg}")
    logging.info(f"Total execution time: {format_duration(script_duration)}")
    
    
if __name__ == "__main__":
    main()