
#Note: The openai-python library support for Azure OpenAI is in preview.
import os
import openai
import openai as openai

os.environ["OPENAI_API_KEY"] = '33e8f0c860bc4109825496444bbfed3e'
openai.api_type = "azure"
openai.api_base = "https://community-openai-34.openai.azure.com/"
openai.api_version = "2023-03-15-preview"
openai.api_key = os.getenv("OPENAI_API_KEY")

inst = ""
inputs = ""

response = openai.ChatCompletion.create(
  engine="gpt35-34",
  messages=[
      {"role": "system", "content": inst},
      {"role": "user", "content": inputs}
  ],
  temperature=0.7,
  max_tokens=800,
  top_p=0.95,
  frequency_penalty=0,
  presence_penalty=0,
  stop=None)

with open('../iteration/history/key1.txt', 'r', encoding='utf-8') as f1:
    l = f1.readline()
    while l.strip():
        key = l.strip()

        l = f1.readline()
        