FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8000
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY main.py reliability.py ./
COPY resultsskinwise/efficientnet_final.keras resultsskinwise/class_index.pkl ./resultsskinwise/
COPY final_model2.keras ./
RUN useradd --create-home appuser
USER appuser
EXPOSE 8000
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
