FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY category1-assistant-coach/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared module
COPY shared /app/shared

# Copy application code
COPY category1-assistant-coach/app /app/app

# Set Python path
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
