import anthropic
import os
import hydra
from omegaconf import DictConfig
import copy
import regex as re


class OpenAI:
    def __init__(self, conf):
        self.llm_conf = conf.llm
        self.client = anthropic.Client(api_key = os.environ['ANTHROPIC_API_KEY'])

    def generate(self, prompt):
        params = {
            'prompt' : f'{anthropic.HUMAN_PROMPT} {prompt} {anthropic.AI_PROMPT} Solution:',
            'max_tokens_to_sample' : 100,
            'model' : 'claude-v1',
            'stream': False
        }

        response = self.client.completion_stream(**params)
        return next(response)['completion'].replace('\n','')


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
        text = self.generate(input)
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
