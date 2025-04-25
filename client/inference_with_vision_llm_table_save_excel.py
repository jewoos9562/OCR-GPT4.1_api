import tritonclient.http as httpclient
import numpy as np
import os
import base64
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd
import re

# === 설정 ===
TRITON_URL = "localhost:8000"
MODEL_NAME = "ensemble"
input_image_path = "/workspace/sample_image2/insure/images/insure_00004.jpeg"
#input_image_path = "/workspace/sample_images/IMG_OCR_6_F_0015610.png"
save_folder = "/workspace/result"
os.makedirs(save_folder, exist_ok=True)

# Triton 클라이언트
client = httpclient.InferenceServerClient(url=TRITON_URL)

def encode_image_base64(image_path):
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
        if len(encoded) % 4:
            encoded += "=" * (4 - len(encoded) % 4)
        return encoded

def decode_float(x):
    try:
        return float(x.decode("utf-8")) if isinstance(x, bytes) else float(x)
    except:
        return 0.0

# Triton 추론
encoded_image = encode_image_base64(input_image_path)
inputs = httpclient.InferInput("input_image", [1], "BYTES")
inputs.set_data_from_numpy(np.array([encoded_image], dtype=object))
outputs = [
    httpclient.InferRequestedOutput("final_texts_with_line"),
    httpclient.InferRequestedOutput("html_tags")
]
response = client.infer(model_name=MODEL_NAME, inputs=[inputs], outputs=outputs)
texts = response.as_numpy("final_texts_with_line")
html = response.as_numpy("html_tags")

# OCR 텍스트 정리
ocr_lines = []
for t in texts:
    x1, y1, x2, y2, text, line_num = t
    x1, y1, x2, y2 = map(decode_float, [x1, y1, x2, y2])
    text = text.decode("utf-8") if isinstance(text, bytes) else str(text)
    ocr_lines.append(f"[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}] \"{text}\"")
ocr_text_block = "\n".join(ocr_lines)
html_text = "\n".join(line.decode("utf-8") for line in html)

# OpenAI
load_dotenv()
openai_client = OpenAI()

# GPT Vision 프롬프트
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

# GPT-4 Vision 요청
with open(input_image_path, "rb") as image_file:
    response = openai_client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{encoded_image}"
                    }},
                    {"type": "text", "text": "\n📌 HTML 테이블:\n" + html_text},
                    {"type": "text", "text": "\n📌 OCR 텍스트:\n" + ocr_text_block},
                ]
            }
        ],
        temperature=0.3,
    )

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
base_name = os.path.splitext(os.path.basename(input_image_path))[0]
excel_save_path = os.path.join(save_folder, f"{base_name}_key_value.xlsx")
df.to_excel(excel_save_path, index=False)
print(f"✅ 엑셀 저장 완료: {excel_save_path}")
