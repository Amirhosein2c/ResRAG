# ResRAG Docker Setup Guide

This guide will help you dockerize and run your ResRAG application using Docker and Docker Compose on WSL2 Ubuntu 20.04.

## Prerequisites

Make sure you have the following installed on your WSL2 Ubuntu system:

### 1. Docker Installation
```bash
# Update package index
sudo apt update

# Install required packages
sudo apt install apt-transport-https ca-certificates curl gnupg lsb-release

# Add Docker's official GPG key
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# Add Docker repository
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker
sudo apt update
sudo apt install docker-ce docker-ce-cli containerd.io

# Add your user to docker group (to avoid using sudo)
sudo usermod -aG docker $USER

# Restart your terminal or run:
newgrp docker
```

### 2. Docker Compose Installation
```bash
# Download Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose

# Make it executable
sudo chmod +x /usr/local/bin/docker-compose

# Verify installation
docker-compose --version
```

## Project Structure

Make sure your project has this structure:
```
resrag-app/
├── app.py              # Your updated Streamlit app
├── requirements.txt    # Python dependencies
├── Dockerfile         # Docker image definition
├── docker-compose.yml # Multi-service setup
├── setup.sh          # Automated setup script
├── .dockerignore     # Files to ignore in Docker build
├── .env              # Environment variables (optional)
├── DBPATH/           # ChromaDB storage (created automatically)
└── DATA/             # Uploaded files storage (created automatically)
```

## Quick Start

### Option 1: Automated Setup (Recommended)
```bash
# Make setup script executable
chmod +x setup.sh

# Run setup script
./setup.sh
```

### Option 2: Manual Setup
```bash
# Build and start services
docker-compose up -d

# Pull required Ollama models (this may take 10-15 minutes)
docker exec ollama ollama pull llama3.2
docker exec ollama ollama pull nomic-embed-text

# Check if everything is running
docker-compose ps
```

## Access Your Application

- **ResRAG Web App**: http://localhost:8501
- **Ollama API**: http://localhost:11434


## Useful Commands

### Service Management
```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down

# Restart services
docker-compose restart

# View logs
docker-compose logs -f

# View logs for specific service
docker-compose logs -f resrag-app
docker-compose logs -f ollama
```

### Model Management
```bash
# List installed models
docker exec ollama ollama list

# Pull a new model
docker exec ollama ollama pull <model-name>

# Remove a model
docker exec ollama ollama rm <model-name>
```

### Data Management
```bash
# Clear all data (be careful!)
docker-compose down
sudo rm -rf DBPATH/ DATA/
docker-compose up -d
```

## Troubleshooting

### 1. "Ollama Disconnected" Error
- Wait 2-3 minutes after starting - models need time to load
- Check if Ollama container is running: `docker-compose ps`
- Check Ollama logs: `docker-compose logs ollama`

### 2. Models Not Found
```bash
# Pull missing models
docker exec ollama ollama pull llama3.2
docker exec ollama ollama pull nomic-embed-text
```

### 3. Port Already in Use
```bash
# Check what's using the port
sudo netstat -tulpn | grep :8501

# Stop any existing Streamlit processes
pkill -f streamlit
```

###