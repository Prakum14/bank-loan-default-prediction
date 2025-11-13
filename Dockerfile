# Use a Python 3.10 slim image as the base
FROM python:3.10-slim

# Set non-interactive mode for Debian/Ubuntu environments
ENV DEBIAN_FRONTEND=noninteractive

# --- Git LFS Installation Block (Crucial for large files) ---
# Install git and git-lfs, which are necessary if the container needs
# to clone the repository or interact with LFS pointers later.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        git-lfs && \
    rm -rf /var/lib/apt/lists/*
# -----------------------------------------------------------

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies first (for efficient Docker caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code and data
# This assumes your large file is located at data/train.csv on your local machine
COPY src/ src/
COPY data/ data/
COPY setup.py .
COPY artifacts/ artifacts/

# Install the project locally (assuming a setup.py for project installation)
RUN pip install .

# Define the command to run your model prediction service
# Replace 'loan_predict' with your actual entry point script or command
CMD ["loan_predict"]
