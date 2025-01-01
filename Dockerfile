# Use Ubuntu base image
FROM ubuntu:latest

# Set environment variables to avoid user interaction during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Update package list
RUN apt-get update

# Install essential utilities for adding repositories
RUN apt-get install -y software-properties-common

# Add universe repository
RUN add-apt-repository universe && apt-get update

# Install required packages: Python, Pip, Git, gettext-base
RUN apt-get install -y python3 python3-pip git gettext-base

# Clean up to reduce image size
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# Switch to root user
USER root

# Set working directory
WORKDIR /app

# Copy application files
COPY . /app

# Install Python dependencies
RUN pip3 install --no-cache-dir --upgrade --requirement requirements.txt --exists-action=i --break-system-packages

# Expose ports for NGINX and Streamlit
EXPOSE ${NGINX_PORT} ${STREAMLIT_PORT}
