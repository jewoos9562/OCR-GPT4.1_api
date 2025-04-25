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

# === Triton client 생성 ===
client_triton = httpclient.InferenceServerClient(url=TRITON_URL)

# === 이미지 base64 인코딩 ===
def encode_image_base64(image_path):
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
        if len(encoded) % 4:
            encoded += "=" * (4 - len(encoded) % 4)
        return encoded

# === Triton 입력 데이터 구성 ===
encoded_image = encode_image_base64(input_image_path)
inputs = httpclient.InferInput("input_image", [1], "BYTES")
inputs.set_data_from_numpy(np.array([encoded_image], dtype=object))

outputs = [
    httpclient.InferRequestedOutput("final_texts_with_line")
]

# === Triton 추론 ===
response = client_triton.infer(model_name=MODEL_NAME, inputs=[inputs], outputs=outputs)
texts = response.as_numpy("final_texts_with_line")  # (N, 6)

# === 위치 정보 + 텍스트 정리 ===
def decode_float(x):
    if isinstance(x, bytes):
        return float(x.decode("utf-8"))
    return float(x)

structured_lines = []
for t in texts:
    x1, y1, x2, y2, text, line_num = t
    x1, y1, x2, y2 = map(decode_float, [x1, y1, x2, y2])
    text = text.decode("utf-8") if isinstance(text, bytes) else str(text)
    structured_lines.append(f"[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}] \"{text}\"")

ocr_summary = "\n".join(structured_lines)

# === LLM 프롬프트 구성 ===
prompt = f"""
다음은 문서 이미지에서 추출된 OCR 텍스트와 위치 정보입니다.
각 텍스트는 해당 문서의 위치 좌표와 함께 주어집니다.
이 정보를 기반으로 문서에 포함된 중요한 key-value 쌍을 추출해줘.
폼의 구조를 이해하고 가능한 정리된 방식으로 요약해줘.

OCR 결과:
{ocr_summary}
"""

# === OpenAI API 호출 ===
load_dotenv()  # .env에서 OPENAI_API_KEY 로딩
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4.1",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.2,
    max_tokens=1500
)

# === 결과 출력 ===
print("📄 추출된 Key-Value 요약:")
print(response.choices[0].message.content)
