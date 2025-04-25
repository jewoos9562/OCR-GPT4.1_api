# PyTorch + CUDA 개발 도구 포함된 Devel 이미지 사용
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel

# 기본 패키지 업데이트 및 필수 라이브러리 설치
RUN apt-get update && apt-get install -y \
    git curl wget unzip nano vim \
    build-essential cmake pkg-config \
    libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Python 라이브러리 설치
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        ultralytics \
        opencv-python-headless \
        matplotlib \
        numpy \
        tqdm

# 마운트할 폴더 생성
RUN mkdir -p /workspace/sample

# 컨테이너 실행 시 기본 쉘 유지
CMD ["/bin/bash"]
