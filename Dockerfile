# Stage 1: The builder stage
FROM python:3.11-slim-bullseye AS builder
WORKDIR /code

RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*


# RUN apt-get update && apt-get install libgl1

# RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 libgl1-mesa-glx libgl1

# RUN apt-get update && apt-get install -y \
#     libgl1-mesa-glx \
#     tesseract-ocr \
#     libtesseract-dev \
#     poppler-utils \
#     && rm -rf /var/lib/apt/lists/*
# OR

# RUN pip uninstall -y opencv-python opencv-contrib-python && pip install opencv-python-headless



# RUN apt-get update && apt-get install libgl1 -y


# RUN apt-get update && apt-get install -y libgl1-mesa-glx

COPY requirements.txt .
RUN python -m venv /opt/venv
RUN /opt/venv/bin/pip install --no-cache-dir -r requirements.txt
RUN /opt/venv/bin/pip install pymupdf
RUN /opt/venv/bin/pip install sentencepiece

# old
# COPY ./app ./app

# COPY /opt/venv ./venv
COPY ./app /code/app


# Stage 2: The final, production-ready runner stage
FROM python:3.11-slim-bullseye

WORKDIR /code # Working directory is the project root
COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

# Copy the entire app directory
COPY --from=builder /code/app ./app

EXPOSE 8000
# The command can now find 'app.main' as a module
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

