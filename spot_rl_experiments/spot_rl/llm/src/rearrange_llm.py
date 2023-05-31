import openai
import os
import hydra
from omegaconf import DictConfig
import copy
import regex as re


class OpenAI:
    def __init__(self, conf):
        self.llm_conf = conf.llm
        self.client = openai.Completion()
        self._validate_conf()
        self.verbose = conf.verbose

    def _validate_conf(self):
        try:
            openai.api_key = os.environ["OPENAI_API_KEY"]
        except Exception:
            raise ValueError("No API keys provided")
        if self.llm_conf.stream:
            raise ValueError("Streaming not supported")
        if self.llm_conf.n > 1 and self.llm_conf.stream:
            raise ValueError("Cannot stream results with n > 1")
        if self.llm_conf.best_of > 1 and self.llm_conf.stream:
            raise ValueError("Cannot stream results with best_of > 1")

    def generate(self, prompt):
        params = copy.deepcopy(self.llm_conf)
        params["prompt"] = prompt
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
        ans = self.llm.generate(prompt)
        return ans

    def parse_instructions(self, input):
        text = self.generate(input)['choices'][0]['text']
        matches =  re.findall('\(.*?\)', text)
        matches = [match.replace('(','').replace(')','') for match in matches]
        nav_1, pick, nav_2, place = matches
        place, nav_2 = place.split(',')
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
    print(ans["choices"][0]["text"])


if __name__ == "__main__":
    main()
