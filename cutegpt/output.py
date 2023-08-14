import json

from transformers import LlamaForCausalLM, LlamaTokenizer
import torch


model_name = "./kw-cutegpt-13b-ift"
tokenizer = LlamaTokenizer.from_pretrained(model_name)
print('additional_special_token:', tokenizer.additional_special_tokens)
model = LlamaForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
)
model.eval()
device = torch.device("cuda:0")
model = model.to(device)

data_all = json.load(open('./data/data_for_gpt35.json', 'r', encoding='utf-8'))
data_out = json.load(open('./data/data_for_gpt35.json', 'r', encoding='utf-8'))
for i in range(0, 100):
    s1 = "'你需要对问题的参考答案进行改写。问题“"
    s2 = "”参考答案为“"
    s3 = "”，请给出参考答案的另一个版本，确保新内容保留原始含义和信息。以下是改写的答案版本："
    d = data_all[str(i)]
    question = d["src"]
    sys_summ = d["sys_summ"]
    print(question)
    print(sys_summ)
    queries = [s1 + question + s2 + s + s3 for s in d["ref_summs"]]
    sum_weighted_log_prob = 0
    for prompt in queries:
        print(prompt)

        input_ids = tokenizer(prompt + sys_summ, return_tensors="pt", padding=False, truncation=False,
                              add_special_tokens=False)
        input_ids = input_ids["input_ids"].to(device)

        outputs = model(input_ids)
        logits = outputs.logits
        # 将logits转化为概率
        probs = torch.nn.functional.softmax(logits, dim=-1)
        # 获取"label"部分的token id
        label_ids = tokenizer(sys_summ, return_tensors="pt", padding=False, truncation=False, add_special_tokens=False)
        # 确保label在输入文本中的位置
        start_pos = len(tokenizer.encode(prompt))  # 获取"prompt + "部分的长度作为起始位置
        end_pos = start_pos + len(label_ids)  # 结束位置
        # 假设权重w_{t}为1
        weights = torch.ones(end_pos - start_pos).to(device)
        # 获取"label"部分每个token的概率，并计算对数概率
        label_probs = probs[0, start_pos:end_pos, :]
        label_log_probs = torch.log(label_probs)
        # 计算每一行中自大的log_prob
        max_log_probs = torch.max(label_log_probs, dim=-1).values
        weighted_log_probs = weights * max_log_probs
        total_weighted_log_prob = torch.sum(weighted_log_probs).item()
        # print(total_weighted_log_prob)
        sum_weighted_log_prob += total_weighted_log_prob
    print(sum_weighted_log_prob/5)
    data_out[str(i)]["scores"] = {"ref2sys": sum_weighted_log_prob/5}

with open("./result/gpt35_cutegpt_result.json", "w", encoding="utf-8") as f:
    f.write(json.dumps(data_out, ensure_ascii=False))

# queries = []
# labels = []
#
# queries = [
#     '你需要对问题的参考答案进行改写。问题“中华文明的起源和早期国家在什么时候形成？它们的社会制度、经济结构和文化特点有哪些？”参考答案为“中华文明的起源和早期国家在旧石器时代和新石器时代形成。社会制度方面，旧石器时代晚期出现了母系氏族社会，新石器时代晚期出现了父系氏族社会，私有制和阶级分化日益明显。经济结构方面，新石器时代开始从事原始农业和饲养家畜，生活逐渐稳定，此后的文明中农业成为主要的经济活动。文化特点方面，早期人类遗址和古人类化石的发现说明中国是远古人类的重要起源地，石器时代人类先后以打制和磨制的石器作为工具，青铜器铸造是手工业生产中的主要部门。早期的国家有夏、商、西周等，实行奴隶制，并且分封制与宗法制相结合。”，请给出参考答案的另一个版本，确保新内容保留原始含义和信息。以下是改写的答案版本：',
#     '你需要对问题的参考答案进行改写。问题“中华文明的起源和早期国家在什么时候形成？它们的社会制度、经济结构和文化特点有哪些？”参考答案为“中华文明的起源和早期国家在旧石器时代和新石器时代出现，最早的奴隶制国家夏朝建立于公元前2070年。旧石器时代和新石器时代的社会制度是原始社会，母系氏族社会和父系氏族社会；经济结构是渔猎、采集和原始农业；文化特点是旧石器时代和新石器时代人类最早制造的工具——石器，以及元谋人和北京人等的门齿化石、磨制陶器等文物遗存。在商周时期，社会制度是奴隶制，经济结构以农业为主，政治制度是封建制，文化特点是青铜文化和宗教文化。”，请给出参考答案的另一个版本，确保新内容保留原始含义和消息。以下是改写的答案版本：',
#     '你需要对问题的参考答案进行改写。问题“中华文明的起源和早期国家在什么时候形成？它们的社会制度、经济结构和文化特点有哪些？”参考答案为“中华文明的起源和早期国家在旧石器时代和新石器时代出现，最早的奴隶制国家夏朝建立于公元前2070年。旧石器时代和新石器时代的社会制度是原始社会，母系氏族社会和父系氏族社会；经济结构是渔猎、采集和原始农业；文化特点是旧石器时代和新石器时代人类最早制造的工具——石器，以及元谋人和北京人等的门齿化石、磨制陶器等文物遗存。在商周时期，社会制度是奴隶制，经济结构以农业为主，政治制度是封建制，文化特点是青铜文化和宗教文化。”，请给出参考答案的另一个版本，确保新内容保留原始含义和消息。以下是改写的答案版本：',
#     '你需要对问题的参考答案进行改写。问题“中华文明的起源和早期国家在什么时候形成？它们的社会制度、经济结构和文化特点有哪些？”参考答案为“中华文明的起源和早期国家在旧石器时代和新石器时代出现，最早的奴隶制国家夏朝建立于公元前2070年。旧石器时代和新石器时代的社会制度是原始社会，母系氏族社会和父系氏族社会；经济结构是渔猎、采集和原始农业；文化特点是旧石器时代和新石器时代人类最早制造的工具——石器，以及元谋人和北京人等的门齿化石、磨制陶器等文物遗存。在商周时期，社会制度是奴隶制，经济结构以农业为主，政治制度是封建制，文化特点是青铜文化和宗教文化。”，请给出参考答案的另一个版本，确保新内容保留原始含义和消息。以下是改写的答案版本：',
#     '你需要对问题的参考答案进行改写。问题“中华文明的起源和早期国家在什么时候形成？它们的社会制度、经济结构和文化特点有哪些？”参考答案为“中华文明的起源和早期国家在旧石器时代和新石器时代出现，最早的奴隶制国家夏朝建立于公元前2070年。旧石器时代和新石器时代的社会制度是原始社会，母系氏族社会和父系氏族社会；经济结构是渔猎、采集和原始农业；文化特点是旧石器时代和新石器时代人类最早制造的工具——石器，以及元谋人和北京人等的门齿化石、磨制陶器等文物遗存。在商周时期，社会制度是奴隶制，经济结构以农业为主，政治制度是封建制，文化特点是青铜文化和宗教文化。”，请给出参考答案的另一个版本，确保新内容保留原始含义和消息。以下是改写的答案版本：'
# ]
# labels = [
#     "中华文明的起源和早期国家在旧石器时代和新石器时代形成。社会制度方面，旧石器时代晚期出现了母系氏族社会，新石器时代晚期出现了父系氏族社会，私有制和阶级分化日益明显。经济结构方面，新石器时代开始从事原始农业和饲养家畜，生活逐渐稳定，此后的文明中农业成为主要的经济活动。文化特点方面，早期人类遗址和古人类化石的发现说明中国是远古人类的重要起源地，石器时代人类先后以打制和磨制的石器作为工具，青铜器铸造是手工业生产中的主要部门。早期的国家有夏、商、西周等，实行奴隶制，并且分封制与宗法制相结合。",
#     "中华文明的起源和早期国家在旧石器时代和新石器时代形成。社会制度方面，旧石器时代晚期出现了母系氏族社会，新石器时代晚期出现了父系氏族社会，私有制和阶级分化日益明显。经济结构方面，新石器时代开始从事原始农业和饲养家畜，生活逐渐稳定，此后的文明中农业成为主要的经济活动。文化特点方面，早期人类遗址和古人类化石的发现说明中国是远古人类的重要起源地，石器时代人类先后以打制和磨制的石器作为工具，青铜器铸造是手工业生产中的主要部门。早期的国家有夏、商、西周等，实行奴隶制，并且分封制与宗法制相结合。",
#     "中华文明的起源可以追溯到新石器时代晚期的仰韶文化和龙山文化，约在公元前21世纪左右出现了以夏、商、周为代表的早期国家。这些早期国家的社会制度多为封建制度，社会阶层分化明显，存在天子、贵族、农民、奴隶等不同阶层，奴隶主要用于农业生产和工程建设。经济结构以农业为主，也有手工业和商业发展，如商代的青铜器、商纣王时期的盐铁官营制度等。文化特点方面，尊崇祖先，注重礼仪和道德，有卜筮、祭祀、乐舞等宗教和文化活动，同时也有文字记载和文学创作，如商代的甲骨文和《诗经》、周代的《周易》、《尚书》等。",
#     "中华文明的起源可以追溯到约5000年前的新石器时代晚期，而早期国家的形成则始于公元前221年秦国的建立。这些早期国家的社会制度、经济结构和文化特点因地区和时间的不同而有所不同，但总体上都强调了农业、手工业和商业的发展，以及对神灵和祖先的崇拜。 ",
#     "新文化运动对中国近代思想解放产生了深远的影响。它推动了中国社会从封建主义向现代资本主义的转型，促进了民主、自由和科学的思想在人们心中的觉醒，为中国现代化建设提供了重要的思想纲领和指导方针。 "
# ]
#
# for (prompt, answer) in zip(queries, labels):
#     print(prompt + answer)
#
#     input_ids = tokenizer(prompt + answer, return_tensors="pt", padding=False, truncation=False, add_special_tokens=False)
#     input_ids = input_ids["input_ids"].to(device)
#
#     outputs = model(input_ids)
#     logits = outputs.logits
#     # 将logits转化为概率
#     probs = torch.nn.functional.softmax(logits, dim=-1)
#     # 获取"label"部分的token id
#     label_ids = tokenizer(answer, return_tensors="pt", padding=False, truncation=False, add_special_tokens=False)
#     # 确保label在输入文本中的位置
#     start_pos = len(tokenizer.encode(prompt))  # 获取"prompt + "部分的长度作为起始位置
#     end_pos = start_pos + len(label_ids)  # 结束位置
#     # 假设权重w_{t}为1
#     weights = torch.ones(end_pos - start_pos).to(device)
#     # 获取"label"部分每个token的概率，并计算对数概率
#     label_probs = probs[0, start_pos:end_pos, :]
#     label_log_probs = torch.log(label_probs)
#     # 计算每一行中自大的log_prob
#     max_log_probs = torch.max(label_log_probs, dim=-1).values
#     weighted_log_probs = weights * max_log_probs
#     total_weighted_log_prob = torch.sum(weighted_log_probs).item()
#
#     print(total_weighted_log_prob)
