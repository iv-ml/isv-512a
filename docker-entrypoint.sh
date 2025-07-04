#!/bin/bash
set -e

# Activate the virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Execute the command passed to the container
exec "$@"