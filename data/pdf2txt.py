# import fitz # PyMuPDF
#
# # docs = ['./课本/历史/中外历史纲要（上）.pdf', './课本/历史/中外历史纲要（下）.pdf', './课本/历史/国家制度与社会治理.pdf', './课本/历史/经济与社会生活.pdf', './课本/历史/文化交流与传播.pdf']
# docs = ['./课本/生物/分子与细胞.pdf', './课本/生物/遗传与进化.pdf', './课本/生物/稳态与调节.pdf', './课本/生物/生物与环境.pdf', './课本/生物/生物技术与工程.pdf']
#
# for i in range(len(docs)):
#     doc = fitz.open(docs[i])
#
#     for page in doc:
#         width, height = page.rect.width, page.rect.height
#         margin_bottom = height * 0.90
#         rect = fitz.Rect(0, 0, width, margin_bottom)
#
#         # 获取该区域的文本
#         text = page.get_textbox(rect)
#         print(text)
#
#         with open('text/history'+str(i+1)+'.txt', 'a', encoding='utf-8') as f:
#             f.write(text + '\n')

import json

f_json = "./taxonomy/history.json"
f_txt = ["./text/history1.txt", "./text/history2.txt", "./text/history3.txt",
         "./text/history4.txt", "./text/history5.txt"]

all_string = ""
for f in f_txt:
    s = open(f, 'r', encoding='utf-8').read()
    all_string += s

k_now, k_next = "", ""
k, k2, k3 = "", "", ""
k2_before, k_before = "", ""
data_json = json.load(open(f_json, 'r', encoding='utf-8'))
d_json = json.load(open(f_json, 'r', encoding='utf-8'))
for i, d in enumerate(data_json):
    k = list(data_json[i].keys())[0]
    for k1 in d.values():
        for k2 in k1.keys():
            for k3 in k1[k2]:
                k_now = k_next
                k_next = k3
                print("********", k_now, k_next)
                if k_now != "":
                    between = all_string[all_string.index('\n'+k_now+'\n'):all_string.index('\n'+k_next+'\n')]
                    # print(between.replace('\n', ' '))
                    print(d_json[i][k][k2].keys())
                    if k_now in data_json[i][k][k2].keys():
                        print(i, k, k2, k_now)
                        d_json[i][k][k2][k_now] = {"text": between.replace('\n', ' ')}
                        # print(d_json[i][k][k2][k_now])
                    else:
                        if k2_before in data_json[i][k].keys():
                            print(i, k, k2_before, k_now)
                            d_json[i][k][k2_before][k_now] = {"text": between.replace('\n', ' ')}
                        else:
                            print(i-1, k_before, k2_before, k_now)
                            d_json[i-1][k_before][k2_before][k_now] = {"text": between.replace('\n', ' ')}
                        # print(d_json[i][k][k2_before][k_now])
            k2_before = k2
        k_before = k


k_now = k_next
k_next = ""
between = all_string[all_string.index('\n'+k_now+'\n'):]
print(between.replace('\n', ' '))
d_json[-1][k][k2][k_now] = {"text": between.replace('\n', ' ')}

# print(json.dumps(data_json, ensure_ascii=False))

with open(f_json, 'w', encoding='utf-8') as f:
    f.write(json.dumps(d_json, ensure_ascii=False))
