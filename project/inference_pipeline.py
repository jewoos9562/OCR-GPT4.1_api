import tritonclient.http as httpclient
import numpy as np
import os
import base64
import cv2
from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd
import re
import shutil

# === Triton 설정 ===
TRITON_URL = "202.79.101.81:55124"
MODEL_NAME = "ensemble"

# === Triton 클라이언트 생성 (전역으로 한 번만)
triton_client = httpclient.InferenceServerClient(url=TRITON_URL)

# === OpenAI 클라이언트 생성 (환경변수에서 키 불러옴)
load_dotenv()
openai_client = OpenAI()

# === 유틸: 이미지 base64 인코딩 ===
def encode_image_base64(image_path):
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
        if len(encoded) % 4:
            encoded += "=" * (4 - len(encoded) % 4)
        return encoded

# === 유틸: 바이트형 숫자 디코딩 ===
def decode_float(x):
    try:
        return float(x.decode("utf-8")) if isinstance(x, bytes) else float(x)
    except:
        return 0.0

# === 텍스트 bbox 시각화 함수 ===
def draw_bboxes_on_image(image_path, texts, output_path):
    image = cv2.imread(image_path)
    for t in texts:
        x1, y1, x2, y2, text, _ = t
        x1, y1, x2, y2 = map(decode_float, [x1, y1, x2, y2])
        pt1 = (int(x1), int(y1))
        pt2 = (int(x2), int(y2))
        label = text.decode("utf-8") if isinstance(text, bytes) else str(text)

        cv2.rectangle(image, pt1, pt2, (0, 255, 0), 2)
    
    cv2.imwrite(output_path, image)

# === 핵심 함수 ===
def run_ocr_gpt_pipeline(image_path: str, key_list: list = None) -> tuple[pd.DataFrame, str]:
    # === 1. Triton 추론 ===
    encoded_image = encode_image_base64(image_path)
    inputs = httpclient.InferInput("input_image", [1], "BYTES")
    inputs.set_data_from_numpy(np.array([encoded_image], dtype=object))

    outputs = [
        httpclient.InferRequestedOutput("final_texts_with_line"),
        httpclient.InferRequestedOutput("html_tags")
    ]

    response = triton_client.infer(model_name=MODEL_NAME, inputs=[inputs], outputs=outputs)
    texts = response.as_numpy("final_texts_with_line")
    html = response.as_numpy("html_tags")

    # === 2. OCR 텍스트 구성 ===
    ocr_lines = []
    for t in texts:
        x1, y1, x2, y2, text, line_num = t
        x1, y1, x2, y2 = map(decode_float, [x1, y1, x2, y2])
        text = text.decode("utf-8") if isinstance(text, bytes) else str(text)
        ocr_lines.append(f"[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}] \"{text}\"")

    ocr_text_block = "\n".join(ocr_lines)
    html_text = "\n".join(line.decode("utf-8") for line in html)

    # === 3. 프롬프트 구성 ===
    key_prompt = ""
    clean_keys = [k.strip() for k in key_list if k.strip()] if key_list else []
    if clean_keys:
        key_prompt = (
            "\n\n📌 아래 항목들은 반드시 결과에 포함되어야 합니다. 누락되었다면 반드시 다음 형식으로라도 포함시켜주세요:\n"
            + "\n".join(f"- {key}: 미검출" for key in clean_keys)
        )

    prompt = f"""
    다음은 신청서 문서에서 추출된 표와 OCR 텍스트입니다.

    이 문서에서 의미 있는 key-value 정보를 정리해주세요.
    각 항목은 다음 형식으로 작성해주세요:

    key: value

    표의 구조와 OCR 텍스트, 이미지 자체를 모두 고려해서 중요한 정보를 뽑아주세요.
    이미지의 손글씨 정보를 고려해서 value값을 정해주세요.
    같은 key값이 있을 때, 표의 구조와 문서를 잘 분석해서 다르게 구분해주세요.
    체크리스트도 유의해서 공란인지 체크되어있는지 확인해주세요.
    괄호 공란, 서명란 등이 있는 경우는, 그 부분들만 뺀 단어들로 고려해주세요.{key_prompt}

    📌 표 (HTML):
    {html_text}

    📌 OCR 텍스트 박스:
    {ocr_text_block}
    """

    # === 4. GPT 호출 ===
    response = openai_client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        seed=1
    )

    gpt_output = response.choices[0].message.content
    gpt_output = re.sub(r"\*\*", "", gpt_output)

    # === 5. key-value 파싱
    def extract_key_value_pairs(text):
        lines = text.strip().split('\n')
        pairs = []
        for line in lines:
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()
                if key:
                    pairs.append((key, value))
        return pairs

    kv_pairs = extract_key_value_pairs(gpt_output)

    # === 6. 누락된 key 보완
    if clean_keys:
        existing_keys = {k for k, _ in kv_pairs}
        for k in clean_keys:
            if k not in existing_keys:
                kv_pairs.append((k, "미검출"))

    df = pd.DataFrame(kv_pairs, columns=["Key", "Value"])

    # === 7. bbox 이미지 저장
    output_image_path = image_path.replace(".jpg", "_bbox.jpg").replace(".png", "_bbox.jpg")
    draw_bboxes_on_image(image_path, texts, output_image_path)

    # === 8. static 폴더로 복사 (프론트에서 접근 가능하게)
    result_image_path = os.path.join("static/results", os.path.basename(output_image_path))
    os.makedirs(os.path.dirname(result_image_path), exist_ok=True)
    shutil.copy(output_image_path, result_image_path)

    return df, result_image_path
