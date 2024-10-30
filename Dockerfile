FROM python:3.10-slim-buster

# Create a non-root user and switch to it
RUN useradd -m nonroot
USER nonroot

# Copy the requirements file and source code
COPY --chown=nonroot:nonroot requirements.txt .
COPY --chown=nonroot:nonroot . .

# Install build-essential and make with root permissions
USER root
RUN apt update && apt --no-install-recommends install -y build-essential make

# Install Python dependencies
RUN pip install -U pip && pip install -r requirements.txt

# Switch back to non-root user
USER nonroot

# Copy the Makefile and model file
COPY --chown=nonroot:nonroot Makefile .
COPY --chown=nonroot:nonroot models/best_model.pkl ./models/best_model.pkl

# Expose the application port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
