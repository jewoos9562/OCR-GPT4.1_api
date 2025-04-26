from openai import OpenAI
import base64
from dotenv import load_dotenv
import re, os
import pandas as pd

# === OpenAI API Key 설정 ===
load_dotenv()  # .env 파일에서 OPENAI_API_KEY 불러오기
client = OpenAI()  # 환경변수 OPENAI_API_KEY가 자동 로드됨

# === 이미지 → base64 인코딩 ===
def image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

# === 입력 이미지 경로 ===
image_path = "/workspace/ocr/openai/sample_images/bank_00002.jpg"
image_base64 = image_to_base64(image_path)
save_folder = "/workspace/result"
prompt = f"""
다음은 신청서 문서에서 추출된 표와 OCR 텍스트입니다.

이 문서에서 의미 있는 key-value 정보를 정리해주세요.
각 항목은 다음 형식으로 작성해주세요:

key: value

표의 구조와 OCR 텍스트, 이미지 자체를 모두 고려해서 중요한 정보만 뽑아주세요.
변경 전/후 정보가 있다면 '→' 기호로 구분해주세요.
이미지의 손글씨 정보를 고려해서 value값을 정해주세요 
괄호 공란, 서명란 등이 있는 경우는, 그 부분들만 뺀 단어들로 고려해주세요
"""


# === GPT-4-Vision API 호출 ===
response = client.chat.completions.create(
    model="gpt-4.1",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_base64}"
                    }
                }
            ]
        }
    ],
)

# === 결과 출력 ===
# 응답 파싱
gpt_output = response.choices[0].message.content
gpt_output = re.sub(r"\*\*", "", gpt_output)

print("🧾 GPT 응답:")
print(gpt_output)

# key-value 추출
def extract_key_value_pairs(text):
    lines = text.strip().split('\n')
    pairs = []
    for line in lines:
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if key and value:
                pairs.append((key, value))
    return pairs

key_value_list = extract_key_value_pairs(gpt_output)
df = pd.DataFrame(key_value_list, columns=["Key", "Value"])

# 엑셀 저장
base_name = os.path.splitext(os.path.basename(image_path))[0]
excel_save_path = os.path.join(save_folder, f"{base_name}_key_value_only_vision_llm.xlsx")
df.to_excel(excel_save_path, index=False)
print(f"✅ 엑셀 저장 완료: {excel_save_path}")
