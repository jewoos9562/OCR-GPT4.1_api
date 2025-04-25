import tritonclient.http as httpclient
import numpy as np
import os
import base64
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv

# === Triton 설정 ===
TRITON_URL = "localhost:8000"
MODEL_NAME = "ensemble"
input_image_path = "/workspace/sample_images/IMG_OCR_6_F_0005207.png"
save_folder = "/workspace/result"
os.makedirs(save_folder, exist_ok=True)

# === Triton 클라이언트 생성 ===
client = httpclient.InferenceServerClient(url=TRITON_URL)

# === 이미지 base64 인코딩 ===
def encode_image_base64(image_path):
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
        if len(encoded) % 4:
            encoded += "=" * (4 - len(encoded) % 4)
        return encoded
    

# === Triton 입력 ===
encoded_image = encode_image_base64(input_image_path)
inputs = httpclient.InferInput("input_image", [1], "BYTES")
inputs.set_data_from_numpy(np.array([encoded_image], dtype=object))

outputs = [
    httpclient.InferRequestedOutput("final_texts_with_line"),
    httpclient.InferRequestedOutput("html_tags")
]

response = client.infer(model_name=MODEL_NAME, inputs=[inputs], outputs=outputs)

# === 결과 추출 ===
texts = response.as_numpy("final_texts_with_line")  # (N, 6)
html = response.as_numpy("html_tags")               # (T,)

# === HTML 텍스트 구성 ===
html_text = "\n".join(line.decode("utf-8") for line in html)

# === OCR 텍스트 리스트 구성 ===
def decode_float(x):
    if isinstance(x, bytes):
        return float(x.decode("utf-8"))
    return float(x)

ocr_lines = []
for t in texts:
    x1, y1, x2, y2, text, line_num = t
    x1, y1, x2, y2 = map(decode_float, [x1, y1, x2, y2])
    text = text.decode("utf-8") if isinstance(text, bytes) else str(text)
    ocr_lines.append(f"[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}] \"{text}\"")

ocr_text_block = "\n".join(ocr_lines)

# === OpenAI 설정 ===
load_dotenv()  # .env에 OPENAI_API_KEY 저장되어 있어야 함
openai_client = OpenAI()

# === 프롬프트 구성 ===
prompt = f"""
다음은 한 신청서 문서에서 추출된 HTML 테이블과 OCR 텍스트입니다.

이 문서에서 중요한 key값과 value를 정리해줘.
표의 구조를 잘 이해해서 정리해줘
엑셀 파일에 저장할거니까 구조화를 잘해주면 좋겠어

📌 표 (HTML):
{html_text}

📌 OCR 텍스트 박스:
{ocr_text_block}
"""

# === GPT 요청 ===
response = openai_client.chat.completions.create(
    model="gpt-4.1",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.3
)

# === 결과 출력 ===
print("🧾 요약 결과:")
print(response.choices[0].message.content)
