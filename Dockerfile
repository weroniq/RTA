# This file defines the Docker container that will contain the Flask app.

# From the source image
FROM python:3.9.1-slim

# Identify maintainer
LABEL maintainer = "wb72698@student.sgh.waw.pl"

# Set the default working directory
WORKDIR /app/

# Copy requirements.txt outside the container
# to /app/ inside the container
COPY requirements.txt /app/

# Install required packages
RUN pip install -r ./requirements.txt

# Copy app.py and__init__.py outside the container
# to /app/ inside the container
COPY app.py __init__.py /app/

# Copy model.pkl outside the container
# to /app/ inside the container
COPY model.pkl /app/

# Expose the container's port 8889
EXPOSE 8889

# When the container starts, run this
ENTRYPOINT python ./app.py