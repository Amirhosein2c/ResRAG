#!/bin/bash

# Setup script for ResRAG Docker deployment

echo "🚀 Setting up ResRAG Docker environment..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p DBPATH DATA

# Build and start the services
echo "🔨 Building and starting Docker containers..."
docker-compose up -d

# Wait for Ollama to start
echo "⏳ Waiting for Ollama to start..."
sleep 10

# Check if Ollama is running
if ! docker exec ollama ollama list &> /dev/null; then
    echo "⏳ Ollama is starting up, waiting a bit more..."
    sleep 20
fi

# Pull required models
echo "📥 Pulling required Ollama models..."
echo "This may take a while depending on your internet connection..."

echo "Pulling llama3.2 model..."
docker exec ollama ollama pull llama3.2

echo "Pulling nomic-embed-text model..."
docker exec ollama ollama pull nomic-embed-text

# Check service status
echo "🔍 Checking service status..."
docker-compose ps

echo ""
echo "✅ Setup complete!"
echo ""
echo "🌐 Your ResRAG app should be available at: http://localhost:8501"
echo "🤖 Ollama API is available at: http://localhost:11434"
echo ""
echo "📋 Useful commands:"
echo "  - Stop services: docker-compose down"
echo "  - View logs: docker-compose logs -f"
echo "  - Restart services: docker-compose restart"
echo "  - Update models: docker exec ollama ollama pull <model-name>"
echo ""
echo "🔧 If the app shows 'Ollama Disconnected', wait a few minutes for models to load."