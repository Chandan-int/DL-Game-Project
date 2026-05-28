# scoring/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install CPU-only torch (much smaller image)
RUN pip install --no-cache-dir flask==3.0.0

RUN pip install --no-cache-dir \
    torch==2.2.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Copy model + scoring script
COPY difficulty_model.pt .
COPY score.py .

EXPOSE 5001
CMD ["python", "score.py"]