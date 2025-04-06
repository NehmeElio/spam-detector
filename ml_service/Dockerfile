
FROM python:3.12-slim

WORKDIR /app

COPY ml_service/ /app/

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential libssl-dev libffi-dev python3-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* || true

RUN pip install -r /app/requirements.txt

EXPOSE 3000

# Command to run the app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "3000"]
