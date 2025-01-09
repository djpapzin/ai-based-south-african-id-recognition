FROM python:3.10-slim-buster

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY ./model.py /app/model.py

ENV MODEL_TYPE="vit_h"
ENV CHECKPOINT="sam_vit_h_4b8939.pth"
ENV DEVICE="cuda"

CMD ["label-studio-ml", "start", "/app/model.py"] 