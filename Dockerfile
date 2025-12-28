FROM python:3.11-slim

WORKDIR /app

# Устанавливаем зависимости (для CI/Docker)
COPY requirements-ci.txt /app/requirements-ci.txt
RUN pip install --no-cache-dir -r requirements-ci.txt

# Копируем код проекта
COPY . /app

EXPOSE 8000

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
