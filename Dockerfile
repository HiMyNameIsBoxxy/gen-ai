# Use an official Python runtime as a parent image
FROM python:3.12-slim-bookworm

# Install curl + image libraries (needed for torchvision and PIL)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates libglib2.0-0 libsm6 libxrender1 libxext6 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Download uv installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Install uv then remove installer
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure uv is on PATH
ENV PATH="/root/.local/bin/:$PATH"

# Set working directory
WORKDIR /code

# Copy dependency files
COPY pyproject.toml uv.lock /code/

# Install dependencies using uv
RUN uv sync --frozen

# Copy application code and helper library
COPY ./app /code/app
COPY ./helper_lib /code/helper_lib

# Copy trained CNN weights
COPY ./cnn_weights.pth /code/

# Copy trained GAN weights
COPY ./gan_weights.pth /code/

# Copy trained Diffusion weights
COPY ./diffusion_weights.pth /code/

# Copy trained EBM weights
COPY ./ebm_weights.pth /code/

# Expose port
EXPOSE 8000

# Start FastAPI with uvicorn
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
