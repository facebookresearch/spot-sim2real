# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import copy
import os

import hydra
import openai
import regex as re
from omegaconf import DictConfig, OmegaConf


class OpenAI:
    def __init__(self, conf):
        self.llm_conf = conf.llm
        self.client = openai.ChatCompletion()
        self._validate_conf()
        self.verbose = conf.verbose

    def _validate_conf(self):
        try:
            openai.api_key = os.environ["OPENAI_API_KEY"]
        except Exception:
            raise ValueError("No API keys provided")
        if self.llm_conf.stream:
            raise ValueError("Streaming not supported")

    def generate(self, prompt):
        params = OmegaConf.to_object(self.llm_conf)
        params["messages"] = [{"role": "user", "content": prompt}]
        if self.verbose:
            print(f"Prompt: {prompt}")
        return self.client.create(**params)


class RearrangeEasyChain:
    def __init__(self, conf):
        self.conf = conf
        self._build_prompt()
        self.llm = OpenAI(conf)
        self.input_variable = f"<{self.conf.prompt.input_variable}>"

    def _build_prompt(self):
        self.prompt = self.conf.prompt.main_prompt
        if "examples" in self.conf.prompt:
            self.prompt += f"\n{self.conf.prompt.examples}"
        if "suffix" in self.conf.prompt:
            self.prompt += f"\n{self.conf.prompt.suffix}"

    def generate(self, input):
        prompt = self.prompt.replace(self.input_variable, input)
        return self.llm.generate(prompt)

    def parse_instructions(self, input):
        gn_op = self.generate(input)
        msg_content = gn_op["choices"][0]["message"]["content"]
        matches = re.findall("\(.*?\)", msg_content)  # noqa
        matches = [match.replace("(", "").replace(")", "") for match in matches]
        nav_1, pick, nav_2, place = matches
        place, nav_2 = place.split(",")
        nav_1 = nav_1.strip()
        pick = pick.strip()
        nav_2 = nav_2.strip()
        place = place.strip()
        return nav_1, pick, nav_2, place


@hydra.main(config_name="config", config_path="conf")
def main(conf: DictConfig):
    chain = RearrangeEasyChain(conf)
    instruction = conf.instruction
    ans = chain.generate(instruction)
    print(ans["choices"][0]["message"]["content"])


if __name__ == "__main__":
    main()
