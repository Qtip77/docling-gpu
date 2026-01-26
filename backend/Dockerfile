FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# Install Python 3.11 and system deps for docling/easyocr
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    ca-certificates \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast Python package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy custom CA bundle for corporate proxy/firewall
COPY bundle.pem /etc/ssl/certs/bundle.pem
ENV SSL_CERT_FILE=/etc/ssl/certs/bundle.pem
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/bundle.pem

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Copy project files for uv
COPY pyproject.toml uv.lock ./

# Copy langgraph configuration
COPY langgraph.json .

# Copy README for hatchling build
COPY README.md .

# Copy app source for the build
COPY app ./app

# Install all dependencies using uv sync with native TLS
RUN uv sync --native-tls --no-dev --frozen

# Create uploads directory
RUN mkdir -p /app/uploads

# Set model cache directories to persist in the image
ENV HF_HOME=/app/.cache/huggingface
ENV TORCH_HOME=/app/.cache/torch
ENV EASYOCR_MODULE_PATH=/app/.cache/easyocr

# Disable xet/CAS downloads which can fail behind corporate firewalls
# This forces huggingface_hub to use standard HTTP downloads
ENV HF_HUB_ENABLE_HF_TRANSFER=0
ENV HF_HUB_DISABLE_XET=1

# Pre-download all models during build (Docling, EasyOCR, TableFormer)
RUN uv run python -c "\
from docling.document_converter import DocumentConverter, PdfFormatOption; \
from docling.datamodel.base_models import InputFormat; \
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions, TableFormerMode; \
import easyocr; \
print('Downloading Docling models...'); \
pipeline_options = PdfPipelineOptions(); \
pipeline_options.do_ocr = True; \
pipeline_options.do_table_structure = True; \
pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE; \
pipeline_options.ocr_options = EasyOcrOptions(lang=['en']); \
converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}); \
print('Downloading EasyOCR models...'); \
reader = easyocr.Reader(['en'], gpu=False); \
print('All models downloaded successfully!');"

EXPOSE 8000

# Disable timeouts for long-running document processing
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "0"]
