FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY app.py .

# Copy model artifacts (expects artifacts/model.pkl to exist at build time)
COPY artifacts/ artifacts/

EXPOSE 5000

CMD ["python", "app.py"]
