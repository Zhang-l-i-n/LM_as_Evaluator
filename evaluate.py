import json

range_list = [[0, 16], [16, 29], [29, 52], [52, 70], [70, 85], [85, 100]]

data_gpt35 = json.load(open('gpt35_score_by_cutegpt_gptscore.json', 'r', encoding='utf-8'))
data_cutegpt = json.load(open('cutegpt_score_by_cutegpt_gptscore.json', 'r', encoding='utf-8'))

for i in range_list:
    score_gpt35_sum = 0
    score_cutegpt_sum = 0
    for j in range(i[0], i[1]):
        if data_gpt35[str(j)]["scores"]["ref2sys"] < data_cutegpt[str(j)]["scores"]["ref2sys"]:
            print(data_gpt35[str(j)]["scores"]["ref2sys"])
            print(data_cutegpt[str(j)]["scores"]["ref2sys"])
            print(data_gpt35[str(j)]["src"])
        score_gpt35_sum += data_gpt35[str(j)]["scores"]["ref2sys"]
        score_cutegpt_sum += data_cutegpt[str(j)]["scores"]["ref2sys"]
    print('score_gpt35_sum : ' + str(score_gpt35_sum/(i[1]-i[0])))
    print('score_cutegpt_sum : ' + str(score_cutegpt_sum/(i[1]-i[0])))


