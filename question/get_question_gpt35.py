# encoding:utf-8
# Note: The openai-python library support for Azure OpenAI is in preview.
import json
import time
from utils.openai_api import call_openai_api


#######################
# 生成问题
#######################
def get_question(keyword_list, text_list):
    question_candidates_list = []
    for keyword, text in zip(keyword_list, text_list):
        inst = "现在你是一个提问者，针对某个主题进行提问，给出问题。注意只能提一个问题。"
        inputs = "针对’" + keyword + "‘进行提出一个问题，注意只能提一个问题，问题需要尽量详细地考察以下内容\n" + text + "\n请输出问题："
        print(inputs)
        question_candidates_list = call_openai_api(
            inst=inst,
            inputs=inputs,
            temperature=0.9,
            max_tokens=800,
            top_p=0.9,
            n=5
        )
        json_data = {
            "keyword": keyword,
            "text": text,
            "questions": question_candidates_list
        }
        question_candidates_list.append(json_data)
        # fi.write(json.dumps(json_data, ensure_ascii=False) + '\n')
        time.sleep(30)
    return question_candidates_list


#######################
# 筛选问题
#######################
def count_substring_occurrences(strings):
    result = {}
    for i, string1 in enumerate(strings):
        count = 0
        for j, string2 in enumerate(strings):
            if string1 in string2:
                count += 1
        result[string1] = count
    return result


def question_filter(keyword_list, text_list, questions_list, fi):
    question_result = []
    for keyword, text, questions in zip(keyword_list, text_list, questions_list):
        inst = "你现在是一个问题筛选器，需要从问题列表中选出与主题和原文最相关且质量较好的问题。"
        inputs = "针对主题“" + keyword + "”，给定原文：\n" + text[:5000] + \
                 "\n下面是与原文相关的问题列表：" + str(questions) + \
                 "\n请从问题列表中选出与原文最相符合的问题。"
        print(inputs)
        questions = call_openai_api(inst=inst, inputs=inputs, n=5)
        time.sleep(30)
        result = count_substring_occurrences(questions)
        print(result)
        max_count = 0
        best_question = ""
        for k in result.keys():
            if result[k] > max_count:
                best_question = k
                max_count = result[k]
        if max_count < 2:
            best_question = ""
            max_count = 0
        question_result.append(best_question)
        # question_result.append({
        #     "key": keyword,
        #     "best_question": best_question,
        #     "score": max_count,
        #     "questions": result,
        #     "text": text
        # })
        fi.write(json.dumps({
            "key": keyword,
            "best_question": best_question,
            "score": max_count,
            "questions": result,
            "text": text
        }, ensure_ascii=False) + "\n")
    return question_result


#######################
# 生成答案
#######################
def get_answer(keyword_list, text_list, question_list, fi):
    for keyword, text, question in zip(keyword_list, text_list, question_list):
        inst = "你现在需要根据文章内容回答问题"
        inputs = "给定原文：\n" + text[:5000] + \
                 "\n回答以下问题：" + question + \
                 "\n请结合原文给出问题答案"
        print(inputs)
        ref_answer = call_openai_api(inst=inst, inputs=inputs, n=5)
        time.sleep(30)
        fi.write(json.dumps({
            "key": keyword,
            "question": question,
            "ref_answer": ref_answer,
            "text": text
        }, ensure_ascii=False) + "\n")


if __name__ == '__main__':
    # 生成问题
    # f_taxonomy_history = r"./data/taxonomy/history.json"
    # json_data = json.load(open(f_taxonomy_history, 'r', encoding='utf-8'))
    # k_list, t_list = [], []
    # for d in json_data:
    #     for k in d.keys():
    #         for k2 in d[k].keys():
    #             for k3 in d[k][k2].keys():
    #                 k_list.append(k3)
    #                 t_list.append(d[k][k2][k3]['text'])
    # q_list = get_question(k_list, t_list)
    #
    # fw = r"./result/question/history_question_candidates_filtered.txt"
    # fin = open(fw, 'a', encoding='utf-8')
    # q_list = question_filter(k_list, t_list, q_list, fin)
    # fin.close()

    # 需要手工筛选一下

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
    fw = r"./result/question/history_question_answer_multi.txt"
    fin = open(fw, 'a', encoding='utf-8')
    get_answer(k_list[91:92], t_list[91:92], q_list[91:92], fin)
    fin.close()
