from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import os
import io
from inference_pipeline import run_ocr_gpt_pipeline

app = FastAPI()

# 정적 파일 서빙
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("static/index.html", encoding="utf-8") as f:
        return f.read()

@app.post("/upload/")
async def upload(file: UploadFile = File(...), keys: str = Form("")):
    try:
        print("✅ [UPLOAD] 요청 도착")
        print(f"파일 이름: {file.filename}")
        print(f"입력 키: {keys}")

        # 중간 로그도 가능
        image_path = f"/tmp/{file.filename}"
        with open(image_path, "wb") as f_out:
            f_out.write(await file.read())
        print("✅ 이미지 저장 완료:", image_path)

        key_list = [k.strip() for k in keys.strip().splitlines() if k.strip()]
        df, result_image_path = run_ocr_gpt_pipeline(image_path, key_list)

        print("✅ 추론 완료, 결과 이미지:", result_image_path)

        output = io.BytesIO()
        df.to_excel(output, index=False)
        output.seek(0)
        excel_filename = os.path.splitext(file.filename)[0] + "_result.xlsx"
        excel_path = f"static/results/{excel_filename}"
        with open(excel_path, "wb") as f:
            f.write(output.read())

        print("📤 응답 완료:", excel_path)

        return JSONResponse({
            "image_url": f"/static/results/{os.path.basename(result_image_path)}",
            "excel_url": f"/static/results/{excel_filename}"
        })

    except Exception as e:
        print("❌ 에러 발생:", str(e))
        raise
