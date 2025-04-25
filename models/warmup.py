import requests
import os
import threading
import json
import base64
from PIL import Image
import io
import numpy as np
import time

port = 5002
# url = "222.122.67.140"
url = "192.168.159.13"
URL = f"http://{url}:{port}/v2/models/ensemble/infer"

def read_image_data(im_paths):
    encode_ims = []
    for p in im_paths:
        if not os.path.exists(p):
            continue
        with open(p, "rb") as image:
            im_encode = base64.b64encode(image.read()).decode("ascii")
            encode_ims.append(im_encode)
    return encode_ims

def save_results(response, image_name):
    tables = response['outputs'][0]['data']
    for idx, table in enumerate(tables):
        if table != 'failed':
            with open(f'C:/Users/Tmax/Desktop/triton/temp_result/{image_name}/{idx}.html', 'w') as f:
                f.write(table)

    figure_area_result = np.array(response['outputs'][2]['data']).reshape(response['outputs'][2]['shape'])
    table_area_result = np.array(response['outputs'][1]['data']).reshape(response['outputs'][1]['shape'])
    text_result = np.array(response['outputs'][3]['data']).reshape(response['outputs'][3]['shape'])

    print(image_name)
    #print(f'그림 영역: {figure_area_result}\n')
    #print(f'표 영역: {table_area_result}\n')
    #print(f'텍스트 영역 및 인식 결과: {text_result}')

    base64_image = response['outputs'][-1]['data'][0]
    raw_image_str = base64.b64decode(base64_image)
    img = Image.open(io.BytesIO(raw_image_str))
    img.save(f'C:/Users/Tmax/Desktop/triton/temp_result/{image_name}/image.jpg')

def inference(image_data):
    data = {
        "inputs": [
            {
                "name": "input_image",
                "shape": [1],
                "datatype": "BYTES",
                "data": image_data
            }
        ]
    }

    headers = {"content-type": "application/json"}
    try:
        response = requests.post(URL, headers=headers, data=json.dumps(data, ensure_ascii=False))
        response.raise_for_status()  # HTTP 오류 발생 시 예외 처리
        #save_results(response.json(), image_name)
    except requests.exceptions.RequestException as e:
        print(f"Request Exception: {e}")

def inference_thread(image_path):
    image_data = read_image_data([image_path])
    image_name = image_path.split("\\")[-1][:-4]
    os.makedirs(f'C:/Users/Tmax/Desktop/triton/temp_result/{image_name}', exist_ok=True)
    inference(image_data, image_name)

if __name__ == "__main__":
    for _ in range(5):
        image_paths = [
        r'/home/jewoos62/ocr/triton/document_tr/insure_13310.jpg'
        ]
        image_data=read_image_data([image_paths[0]])
        inference(image_data)
    print('warmup_complete')
