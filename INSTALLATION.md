# Installation Guide

This guide provides detailed instructions for setting up the ISV-512A project.

## Prerequisites

- Python 3.10 or higher
- CUDA 12.8 (for GPU support)
- Git
- UV package manager

## Local Installation

### 1. Install UV

UV is a fast Python package manager that replaces Poetry in this project.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone the Repository

```bash
git clone https://github.com/your-repo/isv-512a.git
cd isv-512a
```

### 3. Install Dependencies

UV will automatically create a virtual environment and install all dependencies:

```bash
uv sync
```

For development dependencies:

```bash
uv sync --group dev
```

### 4. Verify Installation

Test if PyTorch and CUDA are working correctly:

```bash
uv run python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## SLURM Installation

For running on HPC clusters with SLURM job scheduler.

### 1. Build Docker Image

```bash
bin/run slurm.build
```

### 2. Convert to Enroot Format

Set the Enroot storage path and convert the Docker image:

```bash
export ENROOT_PATH=/path/to/your/enroot/storage
bin/run slurm.enroot
```

### 3. Submit Training Jobs

Submit a job to SLURM:

```bash
bin/run slurm.train scripts/config/your_config.py
```

## Remote Setup

For setting up on a new remote machine:

```bash
bin/run remote.setup
```

This will install:
- ASDF version manager
- Python
- System dependencies
- UV package manager

## Common Issues

### CUDA Not Available

If you see "CUDA not available" but have an NVIDIA GPU:

1. Check NVIDIA drivers:
   ```bash
   nvidia-smi
   ```

2. Ensure you have the correct CUDA version (12.8) installed

3. For NVIDIA B200 GPUs, make sure you're using PyTorch 2.7.1 with CUDA 12.8

### MediaPipe Installation

MediaPipe only supports Python 3.10-3.12. If using a different version:

```bash
uv python pin 3.10
uv sync
```

### Memory Issues

For large models or datasets, you may need to increase shared memory:

```bash
docker run --shm-size=8g ...
```

## Environment Variables

Create a `.env` file for configuration:

```bash
# Example .env file
WANDB_API_KEY=your_wandb_key
CUDA_VISIBLE_DEVICES=0,1,2,3
```

## Development Tools

### Pre-commit Hooks

Install pre-commit hooks for code quality:

```bash
uv run pre-commit install
```

### Running Tests

```bash
uv run pytest
```

### Code Formatting

```bash
uv run ruff check --fix
uv run ruff format
```

## Useful Commands

- `bin/run deps` - Install/update dependencies
- `bin/run update.lock` - Update the lock file
- `bin/run dev.reset` - Reset the development environment
- `bin/run download.models` - Download required model checkpoints

## Troubleshooting

For issues, check:
1. Python version compatibility (3.10 recommended)
2. CUDA installation and compatibility
3. Available system memory and GPU memory
4. Dependency conflicts in `uv.lock`

For more help, please open an issue on the project repository.