FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY agent.py api.py index.html ./

EXPOSE 8000

CMD ["python", "api.py"]
