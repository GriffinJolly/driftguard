# DriftGuard demo image. Python 3.12 is baked in, so the host machine's Python does not matter.
#
#   docker build -t driftguard-demo .
#   docker run --rm -it -p 8501:8501 -v "${PWD}\demo-data:/data" driftguard-demo
#
# See DEMO.md for the full guide.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DRIFTGUARD_DB=/data/driftguard.db \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_THEME_BASE=light \
    STREAMLIT_THEME_PRIMARY_COLOR="#2a78d6" \
    STREAMLIT_THEME_BACKGROUND_COLOR="#f4f4f1" \
    STREAMLIT_THEME_SECONDARY_BACKGROUND_COLOR="#ffffff" \
    STREAMLIT_THEME_TEXT_COLOR="#0b0b0b"

WORKDIR /app

# The package itself, with the dashboard and proxy extras (streamlit, plotly, fastapi, uvicorn).
COPY pyproject.toml README.md LICENSE ./
COPY driftguard ./driftguard
RUN pip install ".[dashboard,proxy]"

# The live demo page and the helper scripts (e.g. the simulated-data generator).
COPY examples ./examples
COPY scripts ./scripts

# Run as a normal user. All data (databases) lives in /data, which you mount from your machine.
RUN useradd --create-home --uid 1000 driftguard \
 && mkdir /data \
 && chown driftguard /data
USER driftguard
WORKDIR /data
VOLUME /data

# 8501: live demo page.  8502: the full dashboard (started with a different command, see DEMO.md).
EXPOSE 8501 8502
CMD ["streamlit", "run", "/app/examples/live_page.py"]
