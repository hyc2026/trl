from transformers import AutoModelForCausalLM, AutoTokenizer
import json

model_name = "/mnt/bn/videonasi18n/heyc/paper_agent_demo/ckpts/sft/checkpoint-2500"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# prompt = "Give me a short introduction to large language model."
# messages = [
#     {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
#     {"role": "user", "content": prompt}
# ]
# messages = []
with open("/mnt/bn/videonasi18n/heyc/paper_agent_demo/data/agent_small/train.jsonl") as f, open("/mnt/bn/videonasi18n/heyc/paper_agent_demo/data/agent_small/test.jsonl", "w") as f1:
    for line in f.readlines():
        messages = json.loads(line)
        text = tokenizer.apply_chat_template(
            messages["messages"],
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=1024
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        messages["output"] = response
        f1.write(json.dumps(messages) + '\n')
        # break