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
from dataclasses import dataclass

# basic templating engine

import jinja2
jinja2_env = jinja2.Environment()

def find_variables_from_jinja_template(template: str):
    from jinja2 import meta
    ast = jinja2_env.parse(template)
    return meta.find_undeclared_variables(ast)


def exists(v):
    return v is not None





def create_parse_reward_fn(reward_regex_template):
    assert find_variables_from_jinja_template(reward_regex_template) == {'reward'}, 'reward template must include "score" variable'
    reward_regex_str = jinja2_env.from_string(reward_regex_template).render(reward = "([0-9\.]+)")

    # @always(lambda: randrange(0, 10))
    def parse_reward_fn(llm_response: str) -> float:
        result = re.search(rf"{reward_regex_str}", llm_response)

        if not exists(result) or result.groups == 0:
            return None

        if not result.groups(1).isnumeric():
            return None

        return float(result.groups(1))

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