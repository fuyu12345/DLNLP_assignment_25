import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module
from torch.nn.utils.rnn import pad_sequence

from beartype import beartype
from beartype.typing import Optional, Callable, List, Tuple

from einops import rearrange
from dataclasses import dataclass

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# sampling helpers

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1, keepdim = True, eps = 1e-10):
    return ((t / max(temperature, eps)) + gumbel_noise(t)).argmax(dim = dim, keepdim = keepdim)

# nucleus

def top_p(logits, thres = 0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending = True)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim = -1), dim = -1)

    sorted_indices_to_remove = cum_probs > thres
    sorted_indices_to_remove = F.pad(sorted_indices_to_remove, (1, -1), value = False)

    sorted_logits[sorted_indices_to_remove] = float('-inf')
    return sorted_logits.scatter(1, sorted_indices, sorted_logits)

# topk

# def top_k(logits, frac_num_tokens = 0.1, k: Optional[int] = None):
#     num_tokens = logits.shape[-1]

#     k = default(k, ceil(frac_num_tokens * num_tokens))
#     k = min(k, num_tokens)

#     val, ind = torch.topk(logits, k)
#     probs = torch.full_like(logits, float('-inf'))
#     probs.scatter_(1, ind, val)
#     return probs

# decoding

@torch.no_grad()
def sample(
    net,
    prompts,
    seq_len: int,
    temperature = 1.0,
    filter_fn = top_p,
    filter_kwargs = {"thres": 0.9},
    pad_id: int = -1,
    eos_id: int = None,
    output_keep_prompt = False
):
    device = next(net.parameters()).device
    net.eval()

    if isinstance(prompts, (list, tuple)):
        prompts = pad_sequence(prompts, batch_first=True, padding_value=pad_id)

    batch, prompt_len = prompts.shape
    out = prompts.clone()
    curr_len = prompt_len

    while curr_len < seq_len:
        out = F.pad(out, (0, 1), value=pad_id)
        net_input = out[:, :curr_len]

        logits = net(net_input).logits[:, -1, :]  # shape [B, vocab]
        logits = filter_fn(logits, **filter_kwargs)

        sampled = gumbel_sample(logits, temperature=temperature, dim=-1)  # shape [B]
        out[:, curr_len] = sampled.view(-1)  

        curr_len += 1

        if eos_id is not None:
            if ((out[:, prompt_len:] == eos_id).any(dim=1)).all():
                break


    if not output_keep_prompt:
        out = out[:, prompt_len:]

    # Trim to seq_len
    out = out[:, :seq_len - prompt_len] if not output_keep_prompt else out[:, :seq_len]
    return out





# $$$$$$$$$$$$$$$$$$$$$$$$$$$$New code added$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

def pick_paired_rewards(rewards: Tensor) -> Tensor:
    """
    Selects the best and worst rewards from a tensor of rewards.
    Args:
        rewards (Tensor): A tensor of rewards.
    Returns:
        Tensor: A tensor of shape (2, rewards.shape[0]) containing the indices of the best and worst rewards.

    """
    # Clone rewards tensor
    rewards_max = rewards.clone()
    rewards_min = rewards.clone()

    # Find NaN values in rewards tensor
    is_nan_mask = torch.isnan(rewards)

    # Replace NaN values with very small values when finding max
    rewards_max[is_nan_mask] = float('-1e6')

    # Replace NaN values with very large values when finding min
    rewards_min[is_nan_mask] = float('1e6')

    best_idx = rewards_max.argmax(dim=-1)
    worst_idx = rewards_min.argmin(dim=-1)

    return torch.stack((best_idx, worst_idx))



def is_valid_reward_pair(preferred_reward: Tensor, unpreferred_reward: Tensor) -> bool:
    """
    Simple check to see if the preferred and unpreferred rewards are different.
    Args:
        preferred_reward (Tensor): The preferred reward.
        unpreferred_reward (Tensor): The unpreferred reward.
    Returns:
        bool: True if the rewards are different, False otherwise.
    """
    # Check if the preferred and unpreferred rewards are different
    return (preferred_reward != unpreferred_reward).all().item()









import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module
from torch.nn.utils.rnn import pad_sequence

from beartype import beartype
from beartype.typing import Optional, Callable, List, Tuple

from einops import rearrange
import re
from textwrap import dedent


# basic templating engine

import jinja2
jinja2_env = jinja2.Environment()

def find_variables_from_jinja_template(template: str):
    from jinja2 import meta
    ast = jinja2_env.parse(template)
    return meta.find_undeclared_variables(ast)


def exists(v):
    return v is not None




DEFAULT_LLM_AS_JUDGE_PROMPT = """
Review the user's question and the corresponding response using the additive 5-point
scoring system described below. Points are accumulated based on the satisfaction of each
criterion:
- Add 1 point if the response is relevant and provides some information related to
the user's inquiry, even if it is incomplete or contains some irrelevant content.
- Add another point if the response addresses a substantial portion of the user's question,
but does not completely resolve the query or provide a direct answer.
- Award a third point if the response answers the basic elements of the user's question in a
useful way, regardless of whether it seems to have been written by an AI Assistant or if it
has elements typically found in blogs or search results.
- Grant a fourth point if the response is clearly written from an AI Assistant's perspective,
addressing the user's question directly and comprehensively, and is well-organized and
helpful, even if there is slight room for improvement in clarity, conciseness or focus.
- Bestow a fifth point for a response that is impeccably tailored to the user's question
by an AI Assistant, without extraneous information, reflecting expert knowledge, and
demonstrating a high-quality, engaging, and insightful answer.
User: {{ prompt }}
<response>{{ response }}</response>
After examining the user's instruction and the response:
- Briefly justify your total score, up to 100 words.
- Conclude with the score using the format: “Score: <total points>”
Remember to assess from the AI Assistant perspective, utilizing web search knowledge as
necessary. To evaluate the response in alignment with this additive scoring model, we'll
systematically attribute points based on the outlined criteria.
"""

DEFAULT_REWARD_REGEX_TEMPLATE = """
Score: {{ reward }}
"""

# def create_parse_reward_fn(reward_regex_template):
#     assert find_variables_from_jinja_template(reward_regex_template) == {'reward'}, 'reward template must include "score" variable'
#     reward_regex_str = jinja2_env.from_string(reward_regex_template).render(reward = "([0-9\.]+)")

#     # @always(lambda: randrange(0, 10))
    
#     def parse_reward_fn(response_str: str) -> Optional[int]:
#         result = re.search(rf"{reward_regex_str}", response_str.strip())
#         if not result:
#             return None
#         score_str = result.group(1)
#         if not score_str.isnumeric():
#             return None
#         return int(score_str)

#     return parse_reward_fn
def create_parse_reward_fn(reward_regex_template=None):
    """
    If you pass a `reward_regex_template`, it'll still use it.
    Otherwise, falls back to a list of known common regex patterns.
    """

    regexes = []

    # If a custom reward template is passed (backward compatibility)
    if reward_regex_template is not None:
        assert find_variables_from_jinja_template(reward_regex_template) == {'reward'}, 'reward template must include "reward" variable'
        rendered = jinja2_env.from_string(reward_regex_template).render(reward=r"([0-9]+(?:\.[0-9]+)?)")
        regexes.append(re.compile(rendered))
    
    # Common regex combos
    regexes += [
        re.compile(r"Score\s*[:=]?\s*(\d(?:\.0)?)(?:\s*/\s*5)?", re.IGNORECASE),
        re.compile(r"Score\s*[:=]?\s*(\d(?:\.\d+)?)", re.IGNORECASE),
        re.compile(r"Final score\s*[:=]?\s*(\d(?:\.\d+)?)", re.IGNORECASE),
        re.compile(r"(\d(?:\.\d+)?)\s*out of\s*5", re.IGNORECASE),
        re.compile(r"Scored\s*(\d(?:\.\d+)?)", re.IGNORECASE),
    ]

    def parse_reward_fn(response_str: str) -> Optional[int]:
        response_str = response_str.strip()

        for regex in regexes:
            match = regex.search(response_str)
            if match:
                try:
                    score = float(match.group(1))
                    return int(round(score))  # can change to int(score) if you prefer truncation
                except ValueError:
                    continue

        return None  # If no regex matched

    return parse_reward_fn

# reward config

@dataclass
class RewardConfig:
    prompt_template: str
    reward_regex_template: Optional[str] = None
    parse_reward: Optional[Callable[[str], Optional[float]]] = None
    template_fn: Optional[Callable[..., str]] = None
    auto_dedent: bool = True

    def init(self):

        # maybe dedent

        if self.auto_dedent:
            self.prompt_template = dedent(self.prompt_template)

            if exists(self.reward_regex_template):
                self.reward_regex_template = dedent(self.reward_regex_template)

        # initialize render function for prompt and response template

        prompt_template = self.prompt_template
        assert find_variables_from_jinja_template(prompt_template) == {'prompt', 'response'}, 'template must include prompt and response templating variables'
        self.template_fn = jinja2_env.from_string(prompt_template).render

        # initialize the parse_reward if only the reward regex template is given

        if not exists(self.parse_reward):
            assert exists(self.reward_regex_template), 'reward_regex_template must be given if parse_reward is not passed in'
            self.parse_reward = create_parse_reward_fn(self.reward_regex_template)

        return self

# config, allowing for different types of reward prompting
# colocate with functions for extracting the response and reward

# SELF_REWARD_PROMPT_CONFIG = dict(
#     default = RewardConfig(
#         prompt_template = DEFAULT_LLM_AS_JUDGE_PROMPT,
#         reward_regex_template = DEFAULT_REWARD_REGEX_TEMPLATE
#     )
# )