import os
import openai
from requests.exceptions import RetryError, Timeout
from retrying import retry

# 设置OpenAI API密钥
os.environ["OPENAI_API_KEY"] = "2d6f2d5784c84e079ba30935e2980135"
openai.api_type = "azure"
openai.api_base = "https://community-openai-990.openai.azure.com/"
openai.api_version = "2023-03-15-preview"
openai.api_key = os.getenv("OPENAI_API_KEY")
engine = "gpt-4-990"

# 定义API调用重试参数
retry_kwargs = {
    'wait_exponential_multiplier': 1000,
    'wait_exponential_max': 10000,
    'stop_max_attempt_number': 3,
    'retry_on_exception': lambda x: isinstance(x, RetryError) or isinstance(x, Timeout)
}


@retry(**retry_kwargs)
def call_openai_api(inst, inputs, temperature=1.0, max_tokens=800, top_p=1, n=1):
    if len(inputs) > 5500:
        inputs = inputs[:5500]
    try:
        response = openai.ChatCompletion.create(
            engine=engine,
            messages=[
                {"role": "system", "content": inst},
                {"role": "user", "content": inputs},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            n=n,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None)
        result_list = [response["choices"][i]["message"]["content"] for i in range(n)]
        return result_list
    except Exception as e:
        # 发生异常时会在 retry 库帮助下自动重试
        raise RetryError(f'OpenAI API call failed with exception: {e}')


print(call_openai_api('写一首诗', '写一首关于夏天的诗'))
