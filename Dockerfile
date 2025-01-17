
FROM python:3.8-slim
WORKDIR /app


RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .


RUN pip install --no-cache-dir -r requirements.txt


RUN mkdir -p number_plates


RUN mkdir -p model


COPY . .


RUN chmod 777 number_plates


EXPOSE 5000


CMD ["uvicorn", "number_plate:app", "--host", "0.0.0.0", "--port", "5000"] 