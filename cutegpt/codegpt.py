import json
from transformers import LlamaForCausalLM, LlamaTokenizer
import torch

overall_instruction = "你是复旦大学知识工场实验室训练出来的语言模型CuteGPT。给定任务描述，请给出对应请求的回答。\n"
device = torch.device("cuda:0")

def call_cutegpt(prompt, model, tokenizer):
    prompt = overall_instruction + "问：{}\n答：".format(prompt)
    print(prompt)
    input_ids = tokenizer(prompt, return_tensors="pt", padding=False, truncation=False, add_special_tokens=False)
    input_ids = input_ids["input_ids"].to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            top_p=0.8,
            top_k=50,
            repetition_penalty=1.1,
            max_new_tokens=2048,
            early_stopping=True,
            eos_token_id=tokenizer.convert_tokens_to_ids('<end>'),
            pad_token_id=tokenizer.eos_token_id,
            min_length=input_ids.shape[1] + 1
        )
    s = outputs[0][input_ids.shape[1]:]
    response = tokenizer.decode(s)
    response = response.replace('<s>', '').replace('<end>', '').replace('</s>', '')
    print(response)

    return response


if __name__ == '__main__':
    print("load model and tokenizer...")
    model_name = "/mnt/public/usr/zhanglin/CuteGPT/kw-cutegpt-13b-ift"
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
    )
    model.eval()
    model = model.to(device)

    print("answering...")
    fq = open(r"history_question_candidates_filtered.txt", 'r', encoding='utf-8')
    line = fq.readline()
    while line.strip():
        data_json = json.loads(line.strip())
        key = data_json["key"]
        question = data_json["best_question"]
        answer = call_cutegpt(question, model, tokenizer)
        line = fq.readline()

        print("***********************")

        with open("cutegpt_result.txt", "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "key": key,
                "question": question,
                "answer": answer
            },ensure_ascii=False))

