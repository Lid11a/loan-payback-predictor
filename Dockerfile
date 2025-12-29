FROM python:3.12-slim

WORKDIR /app

# system deps for LightGBM (fix: libgomp.so.1 missing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-ci.txt /app/requirements-ci.txt
RUN pip install --no-cache-dir -r requirements-ci.txt

COPY . /app

EXPOSE 8000
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
