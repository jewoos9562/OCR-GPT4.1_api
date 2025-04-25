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
save_folder = "/workspace/result"
os.makedirs(save_folder, exist_ok=True)

# === Triton 클라이언트 생성 ===
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

# === Triton 입력 설정 ===
encoded_image = encode_image_base64(input_image_path)
inputs = httpclient.InferInput("input_image", [1], "BYTES")
inputs.set_data_from_numpy(np.array([encoded_image], dtype=object))
outputs = [
    httpclient.InferRequestedOutput("final_texts_with_line"),
    httpclient.InferRequestedOutput("html_tags")
]

# === Triton 추론 ===
response = client.infer(model_name=MODEL_NAME, inputs=[inputs], outputs=outputs)
texts = response.as_numpy("final_texts_with_line")
html = response.as_numpy("html_tags")

# === OCR 텍스트 정리 ===
ocr_lines = []
for t in texts:
    x1, y1, x2, y2, text, line_num = t
    x1, y1, x2, y2 = map(decode_float, [x1, y1, x2, y2])
    text = text.decode("utf-8") if isinstance(text, bytes) else str(text)
    ocr_lines.append(f"[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}] \"{text}\"")
ocr_text_block = "\n".join(ocr_lines)
html_text = "\n".join(line.decode("utf-8") for line in html)

# === OpenAI GPT 요청 ===
load_dotenv()
openai_client = OpenAI()

prompt = f"""
다음은 한 신청서 문서에서 추출된 HTML 테이블과 OCR 텍스트입니다.

이 문서에서 중요한 정보들을 key-value 형식으로 정리해줘.
항목 개수는 정해져 있지 않고, 문서 내용을 바탕으로 의미 있는 정보만 뽑아줘.
"key: value" 형식으로만 정리해줘.
표 구조를 이해해서 각 항목의 변경 전/후 내용도 함께 고려하고,핵심적인 값들만 명확한 key-value 형태로 정리해주세요.
표에서 변경사항이 있는 경우는 화살표를 이용해서 가시성 좋게 만들어줘
표나 내용이 공란인 경우는 공란으로 표시해주면 돼

📌 표 (HTML):
{html_text}

📌 OCR 텍스트 박스:
{ocr_text_block}
"""

response = openai_client.chat.completions.create(
    model="gpt-4.1",  # 또는 "gpt-4.1"
    messages=[{"role": "user", "content": prompt}],
    temperature=0.3
)

gpt_output = response.choices[0].message.content
# ** 강조 기호 제거
gpt_output = re.sub(r"\*\*", "", gpt_output)

print("🧾 GPT 응답:")
print(gpt_output)

# === key-value 추출 함수 ===
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

# === 엑셀 저장 ===
base_name = os.path.splitext(os.path.basename(input_image_path))[0]
excel_save_path = os.path.join(save_folder, f"{base_name}_key_value.xlsx")
df.to_excel(excel_save_path, index=False)
print(f"✅ 엑셀 저장 완료: {excel_save_path}")
