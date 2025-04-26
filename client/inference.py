import tritonclient.http as httpclient
import numpy as np
import os
import base64
from PIL import Image, ImageDraw
import re

# ==== 설정 ====
TRITON_URL = "localhost:8000"
MODEL_NAME = "ensemble"
input_image_path = "/workspace/ocr/openai/sample_images/bank_00002.jpg"
save_folder = "/workspace/result"
os.makedirs(save_folder, exist_ok=True)
save_path = os.path.join(save_folder, "document_result.jpg")

# Triton 클라이언트 생성
client = httpclient.InferenceServerClient(url=TRITON_URL)

# 이미지 base64 인코딩 함수
def encode_image_base64(image_path):
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
        if len(encoded) % 4:
            encoded += "=" * (4 - len(encoded) % 4)
        return encoded

# base64 입력 준비
encoded_image = encode_image_base64(input_image_path)
inputs = httpclient.InferInput("input_image", [1], "BYTES")
inputs.set_data_from_numpy(np.array([encoded_image], dtype=object))

# 출력 요청 정의
outputs = [
    httpclient.InferRequestedOutput("final_texts_with_line"),
    httpclient.InferRequestedOutput("figure_result"),
    httpclient.InferRequestedOutput("final_table_bboxes"),
    httpclient.InferRequestedOutput("html_tags")
]

# ==== 추론 실행 ====
response = client.infer(model_name=MODEL_NAME, inputs=[inputs], outputs=outputs)

# ==== 결과 추출 ====
texts = response.as_numpy("final_texts_with_line")     # (N, 6)
figures = response.as_numpy("figure_result")           # (M, 4)
tables = response.as_numpy("final_table_bboxes")       # (K, 4)
html = response.as_numpy("html_tags")                  # (T,)

# ==== 이미지 시각화 ====
image = Image.open(input_image_path).convert("RGB")
draw = ImageDraw.Draw(image)

for t in texts:
    x1, y1, x2, y2, text, line_num = t
    x1, y1, x2, y2 = map(float, (x1, y1, x2, y2))
    text = text.decode("utf-8") if isinstance(text, bytes) else str(text)
    draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
    draw.text((x1, y1 - 10), text, fill="red")

for box in figures:
    x1, y1, x2, y2 = map(float, box)
    draw.rectangle([x1, y1, x2, y2], outline="blue", width=2)

for box in tables:
    x1, y1, x2, y2 = map(float, box)
    draw.rectangle([x1, y1, x2, y2], outline="green", width=2)

# 저장
image.save(save_path)
print(f"✅ 결과 이미지 저장 완료: {save_path}")

# ==== 전체 HTML 저장 ====
html_save_path = save_path.replace(".jpg", ".html")
with open(html_save_path, "w", encoding="utf-8") as f:
    for line in html:
        f.write(line.decode("utf-8") + "\n")
print(f"📝 전체 HTML 저장 완료: {html_save_path}")

# ==== <table> 블록만 따로 저장 ====
table_count = 0
in_table = False
current_table_lines = []

for line in html:
    # ==== 전체 HTML 디코딩 ====
    decoded_html = "\n".join(line.decode("utf-8") for line in html)

    # ==== <table>...</table> 블록 추출 ====
    table_blocks = re.findall(r"<table.*?>.*?</table>", decoded_html, re.DOTALL)

    # ==== 저장 ====
    for i, table_html in enumerate(table_blocks, 1):
        table_filename = f"table_{i}.html"
        table_save_path = os.path.join(save_folder, table_filename)

        with open(table_save_path, "w", encoding="utf-8") as f:
            f.write(table_html)

        print(f"✅ 표 {i} 저장 완료: {table_save_path}")
