
from openai import OpenAI
import base64
from dotenv import load_dotenv
import os
import re
import zipfile

# === OpenAI 설정 ===
load_dotenv()
client = OpenAI()

# === 이미지 → base64 인코딩 ===
def image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

# === HTML 테이블 분리 및 저장 ===
def save_tables_from_html(raw_html: str, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    tables = re.findall(r"(<table[\s\S]*?</table>)", raw_html, re.IGNORECASE)
    html_paths = []

    style = """
    <style>
    table { border-collapse: collapse; width: 100%; margin: 20px 0; font-size: 14px; }
    th, td { border: 1px solid #999; padding: 8px 12px; text-align: left; }
    th { background-color: #f2f2f2; }
    </style>
    """

    for i, table_html in enumerate(tables, 1):
        filename = f"table_{i}.html"
        path = os.path.join(output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"<html><head>{style}</head><body>{table_html}</body></html>")
        html_paths.append(path)

    # zip 저장
    zip_path = os.path.join(output_dir, "tables.zip")
    with zipfile.ZipFile(zip_path, "w") as zipf:
        for html_file in html_paths:
            zipf.write(html_file, os.path.basename(html_file))

    return zip_path

# === 입력 이미지 설정 ===
image_path = "/workspace/ocr/openai/sample_images/bank_00002.jpg"
save_dir = "/workspace/result/tables"
image_base64 = image_to_base64(image_path)

# === GPT 프롬프트 ===
prompt = """
이 문서에서 중요한 표(table)만 추출해서 HTML <table> 형식으로 정리해주세요.
다른 설명이나 텍스트 없이 <table> 요소만 출력해주세요.
"""

# === GPT Vision API 호출 ===
response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
            ]
        }
    ],
    max_tokens=4096
)

# === GPT 응답 추출 ===
gpt_html_output = response.choices[0].message.content.strip()
print("✅ GPT 응답 수신 완료")

# === HTML 테이블 저장 및 압축 ===
zip_path = save_tables_from_html(gpt_html_output, save_dir)
print(f"📦 압축 저장 완료: {zip_path}")
