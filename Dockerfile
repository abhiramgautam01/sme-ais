FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt requirements-prod.txt ./
RUN pip install --no-cache-dir -r requirements-prod.txt
COPY . .
ENV HOST=0.0.0.0 PORT=8000 SME_DB=/data/sme.db
# The SQLite database lives in /data. Mount a volume there (docker run -v sme-data:/data ...) or it is lost with the container.
VOLUME /data
EXPOSE 8000
# ONE worker on purpose: the live working copy of the data and models is held in process memory (the database is the saved copy).
# --timeout 180 gives the first-start model training (about 15 s, slower on small servers) room to finish.
CMD ["gunicorn", "--workers", "1", "--threads", "4", "--timeout", "180", "--bind", "0.0.0.0:8000", "app:app"]
