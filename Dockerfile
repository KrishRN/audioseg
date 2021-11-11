FROM python:3.8-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app/audioseg

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY audioseg ./audioseg/
COPY setup.py LICENSE README.md ./

RUN pip install ./

RUN chmod +x /app/audioseg/audioseg/main.py

ENTRYPOINT ["python", "/app/audioseg/audioseg/main.py"]