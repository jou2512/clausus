FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# Versions pinned from uv.lock (`uv export ... -o constraints.txt`), so a later build installs what was tested.
COPY pyproject.toml constraints.txt ./
COPY src ./src
RUN pip install --no-cache-dir -c constraints.txt . \
    && useradd --create-home clausus \
    && mkdir -p /data/runs && chown clausus /data/runs
USER clausus
ENV CLAUSUS_RUNS_DIR=/data/runs

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/live', timeout=8).status == 200 else 1)"

CMD ["uvicorn", "--factory", "clausus.app:create_app", "--host", "0.0.0.0", "--port", "8080"]
