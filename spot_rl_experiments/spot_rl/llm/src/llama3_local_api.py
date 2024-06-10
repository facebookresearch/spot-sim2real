import json
import time
from typing import Dict

import requests

# LLAMA3 is supposed to be working locally
# Run OLLAMA3 as ollama run llama3

template_prompt = "You will solve a simple rearrangement task that requires you to Navigate to a given object\n and Pick it up, and then Navigate to a given location and Place it there. Given an open instruction\n that could be similar to Go to the table and find the mug, and return the mug to box, you need to \n return the solution sequence of actions: Nav(table), Pick(mug), Nav(box), Place(mug, box).\n EXAMPLES:\n Instruction: Go to table and find the mug, and return the mug to box \n Solution: Nav(table), Pick(mug), Nav(box), Place(mug, box) \n Instruction: Bring the apple from the kitchen counter to the table \n Solution: Nav(kitchen counter), Pick(apple), Nav(table), Place(apple, table) \n Lets go! \n Instruction: Hey Spot, can you pick up the ball from the couch and place it on the round table? \n Solution:"


class MetaAI:
    def __init__(self, model_name="llama3"):
        self.url = "http://localhost:11434/api/"
        self.model_name = model_name

        self.data = {
            "model": self.model_name,
            "keep_alive": -1,
            "prompt": template_prompt,
            "stream": False,
        }
        response = requests.post(self.url + "generate", json=self.data)
        status_code = response.status_code
        response = json.loads(response.text)
        self.context = response.get("context", [])
        assert (
            status_code == 200
        ), f"ollama returned status code {status_code}, failed to start ollama with {self.model_name}"

    def prompt(self, message: str = "Hey, are you LLAMA3 ?") -> Dict[str, str]:
        t1 = time.time()
        self.data.update({"prompt": message})
        response = requests.post(self.url + "generate", json=self.data)
        status_code = response.status_code
        assert (
            status_code == 200
        ), f"ollama returned status code {status_code}, please check if ollama is running correctly"
        respons = json.loads(response.text)
        print(
            f"LLAMA says {respons.get('response', '')}; Time taken to eval {time.time()-t1} secs"
        )
        return {"message": str(respons["response"])}


if __name__ == "__main__":
    ai = MetaAI()
    response = ai.prompt(message=template_prompt)
    print(response)
