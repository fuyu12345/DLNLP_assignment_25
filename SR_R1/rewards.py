"""Reward functions for GRPO training."""
import os
import asyncio
import json
import math
import re
from typing import Dict
from transformers import logging
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
from torch import Tensor
from .utils import is_e2b_available
import torch
from typing import List, Optional
from einops import repeat as einops_repeat
from tqdm import tqdm
import torch.distributed as dist
if is_e2b_available():
    from dotenv import load_dotenv
    from e2b_code_interpreter import AsyncSandbox

    load_dotenv()
else:
    AsyncSandbox = None

import torch
import random
import itertools
from accelerate import Accelerator
import os
from accelerate import Accelerator

# # $$$$$$$$$$$$$$$$$$$$$$$$$$$$New code added$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$



def extract_prompt_text(single_prompt):
    """Extract the prompt text from a single prompt.
    Args:
        single_prompt (str or dict): The prompt to extract the text from.
    Returns:
        str: The extracted prompt text.
    """
    # Check if the prompt is a list of messages
    if isinstance(single_prompt, list):
        for m in single_prompt:
            if m["role"] == "user":
                return m["content"]
        return single_prompt[-1]["content"]
    elif isinstance(single_prompt, dict):
        return single_prompt["content"]
    elif isinstance(single_prompt, str):
        return single_prompt
    else:
        raise ValueError("Unknown prompt format")

# BACKUP PROMPT TEMPLATE

# def make_meta_judge_chat_prompt(prompt: str, resp_a: str, resp_b: str):
#         return [
#             {"role": "system", "content": "You are a professional AI judge. Your task is to select the better response based on a given rubric."},
#             {"role": "user", "content": f"""Review the user's question and the two responses provided below. 

#     Your task:
#     1. Review the user's question and both responses.
#     2. Evaluate each response according to the rubric.
#     3. Explain which response is more accurate according to the rubric.
#     4. Finally, output ONLY the winner selection following the exact format.

#     === User's Question ===
#     {prompt}

#     === Response A ===
#     {resp_a}

#     === Response B ===
#     {resp_b}

#     === Evaluation Rubric ===
#     - +1 point if the response is relevant and provides some information related to the user's inquiry, even if incomplete or partially irrelevant.
#     - +1 point if the response addresses a substantial portion of the user's question but does not fully resolve it or provide a direct answer.
#     - +1 point if the response answers the basic elements of the user's question in a useful way, regardless of whether it resembles AI, blogs, or search results.
#     - +1 point if the response is clearly written from an AI Assistant's perspective, directly and comprehensively addressing the user's question, and is well-organized, even if slight improvements are possible.
#     - +1 point if the response is impeccably tailored, focused, demonstrates expert knowledge, avoids irrelevant information, and provides a high-quality, insightful answer.

#     After examining both responses:
#     - Response must in English.
#     - Explain which response is better and why, based on the rubric.
#     - Conclude with a clear statement of which judgment is better, must use this format: "Winner: response B" OR "Winner: response A" based on your analysis

#     """}
#         ]



def make_meta_judge_chat_prompt(prompt: str, resp_a: str, resp_b: str):
        return [
            {"role": "system", "content": "You are a professional AI judge. Your task is to select the better response based on a given rubric."},
            {"role": "user", "content": f"""Review the user's question and the two responses provided below. 

    Your task:
    1. Review the user's question and both responses.
    2. Evaluate each response according to the rubric.
    3. Explain which response is more accurate according to the rubric.
    4. Finally, output ONLY the winner selection following the exact format.

    === User's Question ===
    {prompt}

    === Response A ===
    {resp_a}

    === Response B ===
    {resp_b}

    === Evaluation Rubric ===
    (6 points total, +1 for each item)
        1. Mathematical Correctness
        2. Logical Completeness of steps
        3. Explanation of key ideas
        4. Proper notation & formatting
        5. Overall answer quality
        6. Relevance and Coverage of the question

        First, score **each response** on the 6 items above.  
        Then decide which response is better based on higher total score.

    After examining both responses:
    - Response must in English.
    - Explain which response is better and why, based on the rubric.
    - Conclude with a clear statement of which judgment is better. Use one of the following exact formats: 
    - "Winner: response A"
    - "Winner: response B"
    You must choose only one.

    """}
        ]


# old version of parse_winner

# def parse_winner(text: str):
#     text = text.lower()
#     text = re.sub(r'\s+', ' ', text).strip()
#     if re.search(r'winner\s*:\s*response\s*a', text):
#         return "A"
#     elif re.search(r'winner\s*:\s*response\s*b', text):
#         return "B"
#     return None




def parse_winner(text: str):
    """Parse the winner from the judge's response text.
    Args:
        text (str): The judge's response text.
    Returns:
        str: The winner ("A" or "B") or None if not found.
    """
    # preprocess the text
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()

    #  Check for the winner in the text
    if re.search(r'winner\s*:\s*response\s*a\b', text):
        return "A"
    elif re.search(r'winner\s*:\s*response\s*b\b', text):
        return "B"

    #  if not found, check for other patterns
    patterns_a = [
        r'the winner is\s*(?:\*\*)?\s*response\s*a\b',
        r'response\s*a\s+is\s+(?:the\s+)?(?:better|more accurate|more comprehensive|more helpful)',
        r'therefore.*response\s*a\s+is',
        r'hence.*response\s*a\s+is',
        r'response\s*a\s+wins',
        r'\*\*the winner\*\*\s*is\s*response\s*a\b',
        r'based on the evaluation rubric.*response\s*a\s+is',
        r'the better response is\s*\*\*?response\s*a\b',
        r'response\s*a\s+is\s+the\s+better',
    ]
    patterns_b = [
        r'the winner is\s*(?:\*\*)?\s*response\s*b\b',
        r'response\s*b\s+is\s+(?:the\s+)?(?:better|more accurate|more comprehensive|more helpful)',
        r'therefore.*response\s*b\s+is',
        r'hence.*response\s*b\s+is',
        r'response\s*b\s+wins',
        r'\*\*the winner\*\*\s*is\s*response\s*b\b',
        r'based on the evaluation rubric.*response\s*b\s+is',
        r'the better response is\s*\*\*?response\s*b\b',
        r'response\s*b\s+is\s+the\s+better',
    ]

    for pat in patterns_a:
        if re.search(pat, text):
            return "A"
    for pat in patterns_b:
        if re.search(pat, text):
            return "B"

    return None


def run_elo(num_items: int, pairwise_results: list, iterations: int = 10, k_factor: float = 16.0):
    """Run the Elo rating algorithm on the pairwise results.
    Args:
        num_items (int): Number of items to rate.
        pairwise_results (list): List of tuples containing pairwise results.
        iterations (int): Number of iterations to run the Elo algorithm.
        k_factor (float): K-factor for Elo rating adjustment.
    Returns:
        list: List of Elo ratings for each item.
    """
    # Initialize ratings
    ratings = [0.0] * num_items
    for _ in range(iterations):
        for win, lose in pairwise_results:
            r_w, r_l = ratings[win], ratings[lose]
            # Calculate expected score
            expected = 1 / (1 + 10 ** ((r_l - r_w) / 400))
            ratings[win] += k_factor * (1 - expected)
            ratings[lose] += k_factor * (0 - (1 - expected))
    return ratings






def self_judge_reward_func(
    prompts,
    completions,
    model,
    tokenizer,
    *,
    num_generations: int = 5,          
    pair_sample_ratio: float = 1.0,
    elo_iter: int = 10,
    max_new_tokens: int = 500,
    temperature: float = 0.5,
    do_sample: bool = True,
    **kwargs,
):
    """Self-judge reward function for GRPO training.
    Args:
        prompts (list): List of prompts.
        completions (list): List of completions.
        model: The model to use for judging.
        tokenizer: The tokenizer to use for encoding.
        num_generations (int): Number of generations per prompt.
        pair_sample_ratio (float): Ratio of pairs to sample for judging.
        elo_iter (int): Number of iterations for Elo rating.
        max_new_tokens (int): Maximum number of new tokens to generate.
        temperature (float): Temperature for sampling.
        do_sample (bool): Whether to sample or not.
        **kwargs: Additional keyword arguments.
    Returns:
        list: List of normalized scores.
    """
    accelerator = Accelerator()
    device = accelerator.device
    batch_len = len(completions)
    assert batch_len == len(prompts), "prompts / completions must be same length"
    assert batch_len % num_generations == 0, (
        f"batch_len({batch_len}) not  multiple of num_generations({num_generations})"
    )

    # calculate the number of groups
    num_groups = batch_len // num_generations

    # first prepare the judge prompts
    judge_prompts, pair_indices = [], []   # pair_indices use global idx
    for g in range(num_groups):
        offset = g * num_generations

        #  take the first prompt in the group, since the prompts are the same
        prompt_text = extract_prompt_text(prompts[offset])

        #  get completions for this group
        group_texts = [
            completions[offset + i][0]["content"] for i in range(num_generations)
        ]

        # old version of pair sampling without randomization
        # for i, j in itertools.combinations(range(num_generations), 2):
        #     judge_prompts.append(
        #         make_meta_judge_chat_prompt(prompt_text,
        #                                     group_texts[i],
        #                                     group_texts[j])
        #     )
        #     pair_indices.append((offset + i, offset + j))  # 全局下标



        # shuffle the indices and sample pairs
        for i, j in itertools.combinations(range(num_generations), 2):
            # to decide which one is A and which one is B randomly
            if random.random() < 0.5:
                # i → A, j → B   
                resp_a, resp_b = group_texts[i], group_texts[j]
                pair_indices.append((offset + i, offset + j))  # record the global indices
            else:
                # 交换：j → A, i → B
                resp_a, resp_b = group_texts[j], group_texts[i]
                pair_indices.append((offset + j, offset + i))   # record the global indices

            # pass the prompt text and the two responses to the judge prompt
            judge_prompts.append(
                make_meta_judge_chat_prompt(
                    prompt_text,
                    resp_a,
                    resp_b,
                )
            )



    # send the judge prompts to the judge model
    tokenizer.padding_side = "left"
    encoded = tokenizer(
        [tokenizer.apply_chat_template(p, tokenize=False) for p in judge_prompts],
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)



            
    # === ⬇ Debug Information BEGIN ====================================================

    # num_items   = encoded["input_ids"].size(0)
    # seq_len_max = encoded["input_ids"].size(1)

    # print(f"\n Total number of judge prompts: {num_items}")
    # print(f" Maximum sequence length L = {seq_len_max}\n")

    # # If you want to see the actual content, you can decode it; for long texts, it is recommended to view only the first N characters
    # max_chars = 8000         # Maximum number of characters to display for each item to avoid console overflow
    # show_n    = min(13, num_items)   # Display only the first few items (can be adjusted as needed)

    # for i in range(show_n):
    #     decoded_text = tokenizer.decode(encoded["input_ids"][i], skip_special_tokens=True)
    #     print(f"[{i:02d}] {decoded_text[:max_chars]}{' …' if len(decoded_text) > max_chars else ''}")

    # === ⬆ Debug Information END ======================================================


    # generate the judge model outputs
    with torch.no_grad():
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
        )

    if output.shape[1] > encoded["input_ids"].shape[1]:
        # decode the output tokens
        decoded = tokenizer.batch_decode(
            output[:, encoded["input_ids"].shape[1]:], skip_special_tokens=True
        )
    else:
        decoded = [""] * output.shape[0]
        print(" Judge model have no output token")

    # print("\n>>> Judge output:")
    # print(decoded)


    # parse the output to get the winner
    pairwise_results = []
    unparsed_cnt = 0 
    for (i_global, j_global), out in zip(pair_indices, decoded):
        winner = parse_winner(out)          # return "A" / "B" / None
        print(f">>> [Judge] Pair ({i_global},{j_global}) Winner:", winner)
        if winner == "A":
            pairwise_results.append((i_global, j_global))
        elif winner == "B":
            pairwise_results.append((j_global, i_global))
        else:
        # if fail, print the fail judge results
            unparsed_cnt += 1
            print(
                f"\n can not extract the winner —— pair ({i_global},{j_global})"
                f"\nthe origin text is:\n{out}\n{'='*80}"
            )

    print(f"\n>>> TOTAL {unparsed_cnt}  Judge not give exact winners")
    print("pairwise_results:", pairwise_results)
    print("pairwise_results len:", len(pairwise_results))


    # ELO rating AND normalization
    num_total_generations = batch_len           # == len(completions)
    elo_scores = run_elo(num_total_generations,
                         pairwise_results,
                         iterations=elo_iter)

    # group normalization
    num_groups      = num_total_generations // num_generations
    group_norm_list = []

    for g in range(num_groups):
        start = g * num_generations
        end   = start + num_generations
        sub   = elo_scores[start:end]        # get g number of scores

        mn, mx = min(sub), max(sub)
        if mx - mn < 1e-6:                  
            norm_sub = [1.5] * num_generations
        else:
            norm_sub = [(s - mn) / (mx - mn) * 3 for s in sub]

        group_norm_list.extend(norm_sub)

    norm_scores = group_norm_list            
   

    print("\n Final ELO scores:", norm_scores)
    return norm_scores





# # $$$$$$$$$$$$$$$$$$$$$$$$$$$$ New code stop $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$


    

def accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, solution):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if len(gold_parsed) != 0:
            # We require the answer to be provided in correct latex (no malformed operators)
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed="all",
                            units=True,
                        ),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
            # Reward 1 if the content is the same as the ground truth, 0 otherwise
            try:
                reward = float(verify(answer_parsed, gold_parsed))
            except Exception as e:
                print(f"verify failed: {e}, answer: {answer_parsed}, gold: {gold_parsed}")
                reward = 0.0
        else:
            # If the gold solution is not parseable, we reward 1 to skip this example
            reward = 1.0
            print("Failed to parse gold solution: ", sol)
        rewards.append(reward)


    print("accuracy_reward SCORE",rewards)
    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the reasoning process is enclosed within <think> and </think> tags, while the final answer is enclosed within <answer> and </answer> tags."""
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


def tag_count_reward(completions, **kwargs) -> list[float]:
    """Reward function that checks if we produce the desired number of think and answer tags associated with `format_reward()`.

    Adapted from: https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb#file-grpo_demo-py-L90
    """

    def count_tags(text: str) -> float:
        count = 0.0
        if text.count("<think>\n") == 1:
            count += 0.25
        if text.count("\n</think>\n") == 1:
            count += 0.25
        if text.count("\n<answer>\n") == 1:
            count += 0.25
        if text.count("\n</answer>") == 1:
            count += 0.25
        return count

    contents = [completion[0]["content"] for completion in completions]
    return [count_tags(c) for c in contents]


def reasoning_steps_reward(completions, **kwargs):
    r"""Reward function that checks for clear step-by-step reasoning.
    Regex pattern:
        Step \d+: - matches "Step 1:", "Step 2:", etc.
        ^\d+\. - matches numbered lists like "1.", "2.", etc. at start of line
        \n- - matches bullet points with hyphens
        \n\* - matches bullet points with asterisks
        First,|Second,|Next,|Finally, - matches transition words
    """
    pattern = r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [len(re.findall(pattern, content)) for content in completion_contents]

    # Magic number 3 to encourage 3 steps and more, otherwise partial reward
    return [min(1.0, count / 3) for count in matches]


def len_reward(completions: list[Dict[str, str]], solution: list[str], **kwargs) -> float:
    """Compute length-based rewards to discourage overthinking and promote token efficiency.

    Taken from the Kimi 1.5 tech report: https://arxiv.org/abs/2501.12599

    Args:
        completions: List of model completions
        solution: List of ground truth solutions

    Returns:
        List of rewards where:
        - For correct answers: reward = 0.5 - (len - min_len)/(max_len - min_len)
        - For incorrect answers: reward = min(0, 0.5 - (len - min_len)/(max_len - min_len))
    """
    contents = [completion[0]["content"] for completion in completions]

    # First check correctness of answers
    correctness = []
    for content, sol in zip(contents, solution):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if len(gold_parsed) == 0:
            # Skip unparseable examples
            correctness.append(True)  # Treat as correct to avoid penalizing
            print("Failed to parse gold solution: ", sol)
            continue

        answer_parsed = parse(
            content,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed=True,
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        correctness.append(verify(answer_parsed, gold_parsed))

    # Calculate lengths
    lengths = [len(content) for content in contents]
    min_len = min(lengths)
    max_len = max(lengths)

    # If all responses have the same length, return zero rewards
    if max_len == min_len:
        return [0.0] * len(completions)

    rewards = []
    for length, is_correct in zip(lengths, correctness):
        lambda_val = 0.5 - (length - min_len) / (max_len - min_len)

        if is_correct:
            reward = lambda_val
        else:
            reward = min(0, lambda_val)

        rewards.append(float(reward))

    return rewards


def get_cosine_scaled_reward(
    min_value_wrong: float = -1.0,
    max_value_wrong: float = -0.5,
    min_value_correct: float = 0.5,
    max_value_correct: float = 1.0,
    max_len: int = 1000,
):
    def cosine_scaled_reward(completions, solution, **kwargs):
        """Reward function that scales based on completion length using a cosine schedule.

        Shorter correct solutions are rewarded more than longer ones.
        Longer incorrect solutions are penalized less than shorter ones.

        Args:
            completions: List of model completions
            solution: List of ground truth solutions

        This function is parameterized by the following arguments:
            min_value_wrong: Minimum reward for wrong answers
            max_value_wrong: Maximum reward for wrong answers
            min_value_correct: Minimum reward for correct answers
            max_value_correct: Maximum reward for correct answers
            max_len: Maximum length for scaling
        """
        contents = [completion[0]["content"] for completion in completions]
        rewards = []

        for content, sol in zip(contents, solution):
            gold_parsed = parse(sol, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
            if len(gold_parsed) == 0:
                rewards.append(1.0)  # Skip unparseable examples
                print("Failed to parse gold solution: ", sol)
                continue

            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed=True,
                            units=True,
                        ),
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )

            is_correct = verify(answer_parsed, gold_parsed)
            gen_len = len(content)

            # Apply cosine scaling based on length
            progress = gen_len / max_len
            cosine = math.cos(progress * math.pi)

            if is_correct:
                min_value = min_value_correct
                max_value = max_value_correct
            else:
                # Swap min/max for incorrect answers
                min_value = max_value_wrong
                max_value = min_value_wrong

            reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
            rewards.append(float(reward))

        return rewards

    return cosine_scaled_reward


def get_repetition_penalty_reward(ngram_size: int, max_penalty: float):
    """
    Computes N-gram repetition penalty as described in Appendix C.2 of https://arxiv.org/abs/2502.03373.
    Reference implementation from: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

    Args:
    ngram_size: size of the n-grams
    max_penalty: Maximum (negative) penalty for wrong answers
    """
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")

    def zipngram(text: str, ngram_size: int):
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])

    def repetition_penalty_reward(completions, **kwargs) -> float:
        """
        reward function the penalizes repetitions
        ref implementation: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

        Args:
            completions: List of model completions
        """

        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        for completion in contents:
            if completion == "":
                rewards.append(0.0)
                continue
            if len(completion.split()) < ngram_size:
                rewards.append(0.0)
                continue

            ngrams = set()
            total = 0
            for ng in zipngram(completion, ngram_size):
                ngrams.add(ng)
                total += 1

            scaling = 1 - len(ngrams) / total
            reward = scaling * max_penalty
            rewards.append(reward)
        return rewards

    return repetition_penalty_reward


def extract_code(completion: str) -> str:
    pattern = re.compile(r"```python\n(.*?)```", re.DOTALL)
    matches = pattern.findall(completion)
    extracted_answer = matches[-1] if len(matches) >= 1 else ""
    return extracted_answer


def code_reward(completions, **kwargs) -> list[float]:
    """Reward function that evaluates code snippets using the E2B code interpreter.

    Assumes the dataset contains a `verification_info` column with test cases.
    """
    if not is_e2b_available():
        raise ImportError(
            "E2B is not available and required for this reward function. Please install E2B with "
            "`pip install e2b-code-interpreter` and add an API key to a `.env` file."
        )

    # TODO: add support for other languages in E2B: https://e2b.dev/docs/code-interpreting/supported-languages
    """Returns a reward function that evaluates code snippets in a sandbox."""
    evaluation_script_template = """
    import subprocess
    import json

    def evaluate_code(code, test_cases):
        passed = 0
        total = len(test_cases)
        exec_timeout = 5

        for case in test_cases:
            process = subprocess.run(
                ["python3", "-c", code],
                input=case["input"],
                text=True,
                capture_output=True,
                timeout=exec_timeout
            )

            if process.returncode != 0:  # Error in execution
                continue

            output = process.stdout.strip()

            # TODO: implement a proper validator to compare against ground truth. For now we just check for exact string match on each line of stdout.
            all_correct = True
            for line1, line2 in zip(output.split('\\n'), case['output'].split('\\n')):
                all_correct = all_correct and line1.strip() == line2.strip()

            if all_correct:
                passed += 1

        success_rate = (passed / total)
        return success_rate

    code_snippet = {code}
    test_cases = json.loads({test_cases})

    evaluate_code(code_snippet, test_cases)
    """
    code_snippets = [extract_code(completion[-1]["content"]) for completion in completions]
    verification_info = kwargs["verification_info"]
    scripts = [
        evaluation_script_template.format(code=json.dumps(code), test_cases=json.dumps(json.dumps(info["test_cases"])))
        for code, info in zip(code_snippets, verification_info)
    ]

    language = verification_info[0]["language"]

    if not all(v["language"] == language for v in verification_info):
        raise ValueError("All verification_info must have the same language", verification_info)
    try:
        rewards = run_async_from_sync(scripts, language)

    except Exception as e:
        print(f"Error from E2B executor: {e}")
        rewards = [0.0] * len(completions)

    return rewards


def get_code_format_reward(language: str = "python"):
    """Format reward function specifically for code responses.

    Args:
        language: Programming language supported by E2B https://e2b.dev/docs/code-interpreting/supported-languages
    """
    pattern = rf"^<think>\n.*?\n</think>\n<answer>\n.*?```{language}.*?```.*?\n</answer>$"

    def code_format_reward(completions, **kwargs):
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
        return [1.0 if match else 0.0 for match in matches]

    return code_format_reward


def run_async_from_sync(scripts: list[str], language: str) -> list[float]:
    """Function wrapping the `run_async` function."""
    # Create a new event loop and set it
    try:
        # Run the async function and get the result
        rewards = asyncio.run(run_async(scripts, language))
    except Exception as e:
        print(f"Error from E2B executor async: {e}")
        raise e

    return rewards


async def run_async(scripts: list[str], language: str) -> list[float]:
    # Create the sandbox by hand, currently there's no context manager for this version
    sbx = await AsyncSandbox.create(timeout=30, request_timeout=3)

    # Create a list of tasks for running scripts concurrently
    tasks = [run_script(sbx, script, language) for script in scripts]

    # Wait for all tasks to complete and gather their results as they finish
    results = await asyncio.gather(*tasks)
    rewards = list(results)  # collect results

    # Kill the sandbox after all the tasks are complete
    await sbx.kill()

    return rewards


async def run_script(sbx: AsyncSandbox, script: str, language: str) -> float:
    execution = await sbx.run_code(script, language=language)
    try:
        return float(execution.text)
    except (TypeError, ValueError):
        return 0.0
