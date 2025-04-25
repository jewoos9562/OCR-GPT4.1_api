import openai
import os


client = openai.OpenAI(api_key="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" )  # 또는 os.getenv("OPENAI_API_KEY")

response = client.chat.completions.create(
    model="gpt-4.1",  # ✅ 이걸로 바꾸기
    messages=[
        {"role": "user", "content": "Hello, world!"}
    ]
)


print(response.choices[0].message.content)
