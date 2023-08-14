import json
import time

from utils.openai_api_gpt4 import call_openai_api


def get_question(keyword_list, question_list):
    answer_list = []
    for keyword, question in zip(keyword_list, question_list):
        inst = "你需要根据我提出的问题给出详细回答。除了答案不要输出其他内容。"
        inputs = "提问：" + question + "\n回答："
        print(inputs)
        answer = call_openai_api(
            inst=inst,
            inputs=inputs,
            temperature=0.9,
            max_tokens=1000,
            top_p=0.9,
            n=1
        )[0]
        json_data = {
            "keyword": keyword,
            "question": question,
            "answer": answer
        }
        answer_list.append(json_data)
        # fi.write(json.dumps(json_data, ensure_ascii=False) + '\n')
        time.sleep(30)
    return answer_list


if __name__ == '__main__':
    fq = open(r"../result/question/history_question_candidates_filtered.txt", "r", encoding="utf-8")
    l = fq.readline()
    k_list, t_list, q_list = [], [], []
    while l.strip():
        d = json.loads(l.strip())
        k_list.append(d['key'])
        t_list.append(d['text'])
        q_list.append(d['best_question'])
        l = fq.readline()
    # 生成答案
    fw = r"./result/question/answer_history_gpt4/answer.txt"
    fin = open(fw, 'a', encoding='utf-8')
    get_question(k_list, t_list, q_list, fin)
    fin.close()
