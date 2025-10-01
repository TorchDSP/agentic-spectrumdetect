# Stage 1: Build the application
FROM ghcr.io/astral-sh/uv:python3.12-alpine AS builder

# Set the working directory in the container
WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Copy the current directory contents into the container at /app
COPY . /app

RUN uv build --wheel


# Stage 2: Create the final container
FROM python:3.12-slim

RUN apt-get update && apt-get install -y curl

# Set the working directory in the container
WORKDIR /app


# Copy the built wheel from the builder stage
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
COPY --from=builder /app/dist/*.whl .

# Install the application
RUN uv pip install --system --no-cache-dir *.whl
# Expose port 80
EXPOSE 8000

# Run the application
CMD ["python", "-m", "spectrum_server"]
