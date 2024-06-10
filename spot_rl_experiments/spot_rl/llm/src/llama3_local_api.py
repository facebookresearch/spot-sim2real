import json
from typing import Dict

import requests

# LLAMA3 is supposed to be working locally
# Run OLLAMA3 as ollama run llama3
model_name = "llama3:70b"


class MetaAI:
    def __init__(self):
        self.url = "http://localhost:11434/api/generate"

    def prompt(self, message: str = "Hey, are you LLAMA3 ?") -> Dict[str, str]:
        data = {"model": model_name, "prompt": message, "stream": False}
        response = requests.post(self.url, json=data, timeout=30)
        status_code = response.status_code
        assert (
            status_code == 200
        ), f"ollama returned status code {status_code}, please check if ollama is running correctly"
        respons = json.loads(response.text)
        return {"message": str(respons["response"])}


if __name__ == "__main__":
    ai = MetaAI()
    prompt = "You will solve a simple rearrangement task that requires you to Navigate to a given object\n and Pick it up, and then Navigate to a given location and Place it there. Given an open instruction\n that could be similar to Go to the table and find the mug, and return the mug to box, you need to \n return the solution sequence of actions: Nav(table), Pick(mug), Nav(box), Place(mug, box).\n EXAMPLES:\n Instruction: Go to table and find the mug, and return the mug to box \n Solution: Nav(table), Pick(mug), Nav(box), Place(mug, box) \n Instruction: Bring the apple from the kitchen counter to the table \n Solution: Nav(kitchen counter), Pick(apple), Nav(table), Place(apple, table) \n Lets go! \n Instruction: Hey Spot, can you pick up the ball from the couch and place it on the round table? \n Solution:"
    response = ai.prompt(message=prompt)
    print(response)
