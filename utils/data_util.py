import json

# 将data改写为聚类需要的prompt
def d4cluster_pt(json_file):
    prompt = ""
    d = json.load(open(json_file, 'r', encoding='utf-8'))
    for k in d:
        for k1 in k.values():
            for k2 in k1.values():
                for k3 in k2.keys():
                    print(k3)
                    prompt += " - " + k3
    return prompt


# 将question、answer和cutegpt的answer写到待评估的文件中
def gen_file_gptscore(ref_file, answer_file, out_file):

    fr = open(answer_file, 'r', encoding='utf-8')
    fa = open(ref_file, 'r', encoding='utf-8')
    fw = open(out_file, 'w', encoding='utf-8')

    data_json = {}
    l1 = fr.readline()
    l2 = fa.readline()
    i = 0
    while l1.strip():
        print(l1)
        print(l2)
        data_l1 = json.loads(l1.strip())
        data_l2 = json.loads(l2.strip())
        print(data_l1)
        print(data_l2)
        data_json[i] = {
            "src": data_l1['question'],
            "sys_summ": data_l1['answer'],
            "scores": {},
            "ref_summs": data_l2['ref_answer']
        }
        l1 = fr.readline()
        l2 = fa.readline()
        i += 1

    fw.write(json.dumps(data_json, ensure_ascii=False))

    fa.close()
    fr.close()
    fw.close()


gen_file_gptscore('../data/question/history_questions_ref_answers_all.txt', '../result/question/answer_history_gpt35/answer.txt', 'data_for_gpt35.json')


