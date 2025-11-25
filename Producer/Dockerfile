FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the script
COPY producer.py .

# Command to run the script
CMD ["python", "-u", "producer.py"]