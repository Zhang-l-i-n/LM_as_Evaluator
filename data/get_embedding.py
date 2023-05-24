import json
import openai
from tenacity import retry, wait_random_exponential, stop_after_attempt
from tqdm import tqdm


openai.api_version = '2023-03-15-preview'
openai.api_type = 'azure'
openai.api_base = "https://conmmunity-openai.openai.azure.com/"
openai.api_key = '3371b75d06a54deabcdd5818629ca833'


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_embedding(text, engine="ada-1"):
    return openai.Embedding.create(engine=engine, input=[text])["data"][0]["embedding"]


def embeddings(f_txt):
    embedding_list = []
    count = 0
    with open(f_txt, 'r', encoding='utf-8') as frd:
        for line in frd:
            count = count + 1

    pbar = tqdm(total=count)
    with open(f_txt, 'r', encoding='utf-8') as f:
        line = f.readline()
        type = 0
        while line:
            if len(line) > 1:
                if line.startswith('        '):
                    level = 2
                elif line.startswith('    '):
                    level = 1
                    type += 1
                elif line.strip():
                    level = 0
                    type = 0
                embedding_list.append(
                    {"label": line.strip(),
                     "level": level,
                     "type": type,
                     # "type": 1,
                     "embedding": get_embedding(line)
                     }
                )
            line = f.readline()
            pbar.update(1)
    pbar.close()

    return embedding_list


def data_prepare(f_raw, f_emb):
    embedding_list = embeddings(f_raw)
    with open(f_emb, 'w', encoding='utf=8') as f:
        f.write(json.dumps(embedding_list, ensure_ascii=False))


if __name__ == '__main__':
    f_raw, f_emb = './kaogang/physics.txt', './embedding/embedding_physics.json'

    ############
    print('data preparing...')
    data_prepare(f_raw, f_emb)
    print('data prepared')
