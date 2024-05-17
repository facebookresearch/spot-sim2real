## Setup

```
pip install openai
pip install hydra-core --upgrade
```

Create an environmental variable `OPENAI_API_KEY` with your api key, 
```
os.environ["OPENAI_API_KEY"] = "...'
or export OPENAI_API_KEY='...'
```

## Usage

```
python main.py +instruction='Take the water from the table to the kitchen counter' verbose=false
```
Or check `src/notebook.ipynb`

For running llama3, please do `ollama run llama3` in different terminal, and leave it there

## Config 

```
├── conf
│   ├── config.yaml
│   ├── llm
│   │   └── openai.yaml
│   └── prompt
│       ├── rearrange_easy_few_shot.yaml
│       └── rearrange_easy_zero_shot.yaml
```

- llm/openai.yaml: contains the openai configuration: engine, tokens, temperature, etc. Can 
modify it by running `python main.py llm.temperature=0.5`
- prompt/.: contains the prompts for the zero shot and few shot [defaults] tasks

