import json
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import torch

prompts = {
    "generate_query": "Please generate some mutually exclusive queries in a list to search the relevant papers according to the User Query. Searching for survey papers would be better.\nUser Query: {user_query}",
    "select_section": "You are conducting research on `{user_query}`. You need to predict which sections to look at for getting more relevant papers. Title: {title}\nAbstract: {abstract}\nSections: {sections}"
}

class AgentDataset(Dataset):
    def __init__(self, annotations_file, tokenizer):
        self.messages = []
        self.ids = []
        self.answers = [] # 后续看需不需要加上
        self.input_ids = []
        self.lengths = []
        with open(annotations_file) as f:
            for line in f.readlines():
                data = json.loads(line)
                prompt_template = data.get("prompt", "generate_query")
                if prompt_template == "generate_query":
                    prompt = prompts["generate_query"].format(user_query=data["user_query"])
                else:
                    prompt = prompts["select_section"].format(user_query=data["user_query"], title=data["title"], abstract=data["abstract"], sections=data["sections"])
                self.messages.append({
                    "content": prompt,
                    "role": "user"
                })
                self.ids.append(data["id"])
                input_ids = tokenizer.apply_chat_template(
                    [self.messages[-1]],
                    tokenize=True,
                    padding=True,
                    padding_side='left',
                    add_generation_prompt=True,
                )
                self.input_ids.append(input_ids)
                self.lengths.append(len(input_ids))
    
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "lengths": self.lengths[idx],
        }
    
    def __repr__(self):
        return "AgentDataset(\n    features: {},\n    length: {}\n)".format(json.dumps(list(self[0].keys())), len(self))

class ValueModelDataset:
    def __init__(self, annotations_file, tokenizer):
        self.input_ids = []
        self.label = []
        self.lengths = []
        with open(annotations_file) as f:
            for line in f.readlines():
                data = json.loads(line)
                input_ids = tokenizer.apply_chat_template(
                    [data["messages"]],
                    tokenize=True,
                    padding=True,
                    padding_side='left',
                    add_generation_prompt=False,
                )
                # self.input_ids.append(torch.tensor(input_ids))
                self.input_ids.append(input_ids)
                self.lengths.append(len(input_ids))
                self.label.append(data["label"])
    
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            # "input_ids_chosen": self.input_ids[idx],
            # "attention_mask_chosen": torch.ones_like(self.input_ids[idx], dtype=torch.bool),
            # "input_ids_rejected": self.input_ids[idx],
            # "attention_mask_rejected": torch.ones_like(self.input_ids[idx], dtype=torch.bool),
            "input_ids_chosen": self.input_ids[idx][0],
            "attention_mask_chosen": [1] * len(self.input_ids[idx][0]),
            "input_ids_rejected": self.input_ids[idx][0],
            "attention_mask_rejected": [1] * len(self.input_ids[idx][0]),
            "lengths": self.lengths[idx],
            "margin": self.label[idx],
        }
    
    def __repr__(self):
        return "ValueModelDataset(\n    features: {},\n    length: {}\n)".format(json.dumps(list(self[0].keys())), len(self))


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(
        "/mnt/bn/videonasi18n/heyc/ckpts/Qwen2.5-7B-Instruct",
        padding_side="left",
    )
    # train_dataset = AgentDataset("/mnt/bn/videonasi18n/heyc/paper_agent_demo/data/train_agent/train_ppo.jsonl", tokenizer)
    train_dataset = ValueModelDataset("/mnt/bn/videonasi18n/heyc/paper_agent_demo/data/train_agent/train_vm.jsonl", tokenizer)
    print(train_dataset)
    print(train_dataset[0])