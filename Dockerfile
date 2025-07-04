FROM nvcr.io/nvidia/pytorch:25.04-py3

# Set non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    tini \
    ffmpeg \
    libgl1 \
    vim \
    htop \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install UV
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

WORKDIR /src

ENV PYTHONUNBUFFERED=true
ENV PYTHONDONTWRITEBYTECODE=1

# Copy pyproject.toml and uv.lock first for better caching
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen

# Copy application code
COPY . /src

# Set up the Python environment
ENV PYTHONPATH="/src:$PYTHONPATH"

# Copy and set up entrypoint script
COPY docker-entrypoint.sh /src/
RUN chmod +x /src/docker-entrypoint.sh

# Expose the port the app runs on (if needed)
EXPOSE 8000

# Set the entrypoint and default command
ENTRYPOINT ["tini", "--", "/src/docker-entrypoint.sh"]
CMD ["uv", "run", "python", "scripts/dinet/train.py"]