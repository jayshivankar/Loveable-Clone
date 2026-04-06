FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
ENV APP_HOME=/app

WORKDIR $APP_HOME

# Install system dependencies including those needed for psycopg and git if needed
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv globally
RUN pip install --no-cache-dir uv

# Copy uv dependency files
COPY pyproject.toml uv.lock ./

# Install python dependencies using uv
RUN uv sync --frozen

# Copy the application source code
COPY . .

# Expose port
EXPOSE 8000

# Start server
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
