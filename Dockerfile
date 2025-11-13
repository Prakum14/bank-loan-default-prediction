# ----------------------------------------------------------------------
# Stage 1: Build Stage
# This stage installs all necessary packages and builds the Python environment.
# ----------------------------------------------------------------------
FROM python:3.10-slim AS builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
# We use /tmp to stage installation to avoid polluting the final image layers
COPY requirements.txt .

# Install dependencies using pip (including build dependencies if any)
RUN pip install --no-cache-dir -r requirements.txt

# ----------------------------------------------------------------------
# Stage 2: Production Stage
# This stage copies the installed packages and application code into a final, 
# minimal runtime image, reducing the attack surface and image size.
# ----------------------------------------------------------------------
FROM python:3.10-slim

# Set the working directory in the final image
WORKDIR /app

# Copy application files (source code, setup file)
# Note: We do NOT copy the data files (train/test.csv) as they are only needed
# for the training process, not for prediction inference.
COPY src src/
COPY setup.py .
COPY requirements.txt . # Need this for package metadata

# Install the package using setup.py
# This makes the 'loan_train' and 'loan_predict' commands available via the 
# entry_points defined in setup.py.
RUN pip install .

# Copy the trained artifacts (model and preprocessor)
# We assume 'artifacts' is created when 'loan_train' is run outside the container
# or when we train inside the container (see entrypoint options below).
# For deployment, the artifacts should already exist.
# If they don't exist, this step will fail, which is correct for a prediction container.
COPY artifacts artifacts/

# Expose a port if you later wrap this prediction script in a web server (e.g., Flask)
# EXPOSE 8080 

# Define the entry point for running the prediction script.
# Use the executable name defined in setup.py
# This will execute the 'main' function in src/predict.py
CMD ["loan_predict"]

# Alternative CMD for training the model inside the container:
# CMD ["loan_train"]
# Make sure to include the data files (train.csv/test.csv) if you choose this route.
