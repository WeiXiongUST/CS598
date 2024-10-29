import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, HfArgumentParser
import time
from torch.utils.data import DataLoader

import os

from dataclasses import dataclass, field
from typing import Optional
from accelerate import Accelerator
import json
from datasets import load_dataset
from tqdm import tqdm
import numpy as np

tqdm.pandas()

@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """
    dataset_name_or_path: Optional[str] = field(
        default="../my_data/lmsys_response_pairs_format.json",
        metadata={"help": "the location of the dataset name or path"},
    )
    output_dir: Optional[str] = field(
        default="../eval_results/lmsys_response_pairs_pm_reward.json",
        metadata={"help": "the location of the output file"},
    )
    record_dir: Optional[str] = field(
        default="",
        metadata={"help": "the location of the output file"},
    )
    max_length: Optional[int] = field(
        default=9999999999,
        metadata={"help": "the maximum length of the prompt"},
    )
    reward_name_or_path: Optional[str] = field(
        default="RLHFlow/pair-preference-model-LLaMA3-8B",
        metadata={"help": "the name of the gold reward model"},
    )
    input_output_delimiter: Optional[str] = field(
        default="",
        metadata={"help": "the delimiter between input and output"},
    )
    train_micro_batch_size_per_gpu: Optional[int] = field(
        default=4,
        metadata={"help": "the batch size for inference"},
    )

temperature = 1.0
accelerator = Accelerator()

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

device = accelerator.device

model = DebertaV2PairRM.from_pretrained(script_args.reward_name_or_path,
                                             torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").to(device)
tokenizer = AutoTokenizer.from_pretrained("raftrsf/pair_pref", use_fast=True)

pattern_list = ["bold", "list", "emoji" "exclamation", "link", "affirmative"]

prompt_template = "[CONTEXT] {context} [RESPONSE A] {response_A} [RESPONSE B] {response_B} \n"

token_id_A = tokenizer.encode("A", add_special_tokens=False)
token_id_B = tokenizer.encode("B", add_special_tokens=False)
assert len(token_id_A) == 1 and len(token_id_B) == 1
token_id_A = token_id_A[0]
token_id_B = token_id_B[0]

# Load the dataset
ds_dir = script_args.dataset_name_or_path
ds = load_dataset("json", data_files=ds_dir, split="train")

def comp(context, responses):
    # given context, we compare two responses
    # responses = [response1, response2]
    probs_chosen = []
    for chosen_position in [0, 1]:
        # swap order to mitigate position bias
        response_A = responses[chosen_position]
        response_B = responses[1 - chosen_position]
        # prompt = prompt_template.format(
            # context=context, response_A=response_A, response_B=response_B)
        prompt = prompt_template.format(
            input=context, response_a=response_A, response_b=response_B)
        message = [
            {"role": "user", "content": prompt},
        ]
        input_ids = tokenizer.apply_chat_template(
            message, add_generation_prompt=True, return_tensors='pt').to(device)

        with torch.no_grad():
            output = model(input_ids)
        logit_A = output.logits[0, -1, token_id_A].item()
        logit_B = output.logits[0, -1, token_id_B].item()
        # take softmax to get the probability; using numpy
        Z = np.exp(logit_A / temperature) + np.exp(logit_B / temperature)
        logit_chosen = [logit_A, logit_B][chosen_position]
        prob_chosen = np.exp(logit_chosen / temperature) / Z
        probs_chosen.append(prob_chosen)
    avg_prob_chosen = np.mean(probs_chosen)
    if avg_prob_chosen > 0.5:
        return 0
    elif avg_prob_chosen == 0.5:
        # print("The preference model gives the same score for two responses")
        return 0.5
    else:
        return 1

def full_pairwise_ranking(prom, test_texts):
    # Initialize score dictionary
    scores = {txt: 0 for txt in test_texts}

    # Compare every pair of items
    for i in range(len(test_texts)):
        for j in range(i + 1, len(test_texts)):
            result = comp(prom, [test_texts[i], test_texts[j]])
            if result == 0:
                scores[test_texts[i]] += 1
            elif result == 0.5:
                scores[test_texts[i]] += 0.5
                scores[test_texts[j]] += 0.5
            else:
                scores[test_texts[j]] += 1

    # Rank items based on scores in descending order
    ranked_items = sorted(scores, key=scores.get, reverse=True)
    return ranked_items, scores

def get_raw_prompt(old_prompt):
    # get the raw prompt
    return old_prompt

def change_of_format(resp):
    # get the raw response
    return resp

import sys

with open("../log/lmsys/bias.log", "a") as f:
    sys.stdout = f

    data = []

    win_cnt = {pattern: 0 for pattern in pattern_list}
    lose_cnt = {pattern: 0 for pattern in pattern_list}
    tie_cnt = {pattern: 0 for pattern in pattern_list}
    pattern_cnts = {pattern: 0 for pattern in pattern_list}
    with torch.no_grad():
        for sample in tqdm(ds):

            # tmp_prompt = get_raw_prompt(sample['prompt'])
            # test_texts = [change_of_format(tmp_output)
            #             for tmp_output in sample['responses']]

            sample['responses'] = [sample["origin"], sample["augment"]]

            tmp_prompt = get_raw_prompt(sample['prompt'])

            test_texts = [change_of_format(tmp_output)
                        for tmp_output in sample['responses']]


            map_original_texts = {
                test_texts[j]: sample['responses'][j] for j in range(len(test_texts))}

            ranked_texts, scores = full_pairwise_ranking(tmp_prompt, test_texts)

            if ranked_texts is None:
                continue

            rewards = [scores[txt] for txt in test_texts]
            data.append(
                {"prompt": sample["instruction"], "responses": sample["responses"], "rewards": rewards, "pattern": sample["pattern"]})

    if accelerator.is_main_process:
        print("Collect {} data.".format(
            len(data)))
        output_eval_dataset = {}
        output_eval_dataset['type'] = 'text_only'
        output_eval_dataset['instances'] = data
        with open(script_args.output_dir, 'w', encoding='utf8') as f:
            json.dump(output_eval_dataset, f, ensure_ascii=False)

        for d in data:
            pattern = d["pattern"]
            if d['rewards'][0] > d['rewards'][1]:
                win_cnt[pattern] += 1
            elif d['rewards'][0] < d['rewards'][1]:
                lose_cnt[pattern] += 1
            else:
                tie_cnt[pattern] += 1
            
            pattern_cnts[pattern] += 1

        # pattern = script_args.dataset_name_or_path.split("/")[-1].split("_")[0]
        # print("Model:", script_args.reward_name_or_path)
        # print("Pattern:", pattern)
        # print(f"Win rate: {win_cnt}/{len(data)}={win_cnt/len(data):.4f}")
        # print("Lose rate: {}/{}={:.4f}".format(lose_cnt, len(data), lose_cnt/len(data)))
        # print("Tie rate: {}/{}={:.4f}".format(tie_cnt, len(data), tie_cnt/len(data)))

    print("Model:", script_args.reward_name_or_path)
    for pattern in pattern_list:
        print(f"{pattern} Win rate: {win_cnt[pattern]}/{pattern_cnts[pattern]}={win_cnt[pattern]/pattern_cnts[pattern]:.4f}")
        print(f"{pattern} Lose rate: {lose_cnt[pattern]}/{pattern_cnts[pattern]}={lose_cnt[pattern]/pattern_cnts[pattern]:.4f}")
        print(f"{pattern} Tie rate: {tie_cnt[pattern]}/{pattern_cnts[pattern]}={tie_cnt[pattern]/pattern_cnts[pattern]:.4f}")
    
sys.stdout = sys.__stdout__
