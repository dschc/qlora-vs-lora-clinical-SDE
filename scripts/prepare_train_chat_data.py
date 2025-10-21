import json
import random
from tqdm import tqdm
random.seed(1234)

LANG = ['en']
SYSTEM_PROMPT = open("./prompts/naive.txt", "r").read()
DATA_PATH = "./data"

data= []
for lang in LANG:
    data.extend(json.load(open(f'{DATA_PATH}/{lang}/train.json', 'r', encoding='utf-8')))

def get_messages(input, output):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Clinical Report: {input}"},
        {"role": "assistant", "content": output}
    ]

train_data = []
for d in tqdm(data):
    messages = get_messages(d['report'], json.dumps(d['summary'], ensure_ascii=False, indent=4))
    train_data.append({
        "messages": messages
    })

with open(f"{DATA_PATH}/train_chat_{'_'.join(LANG)}.json", "w", encoding='utf-8') as _file:
    json.dump(train_data, _file, ensure_ascii=False, indent=4)
