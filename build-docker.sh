#!/bin/bash

# Build script for EST ASR Pipeline with Speaker-Turn-Based N-Best
# This creates a new Docker image with the speaker-turn-based segmentation functionality

set -e

IMAGE_NAME="est-asr-pipeline:speaker-turn-nbest"

echo "Building Docker image: $IMAGE_NAME"
echo "This image includes the new speaker-turn-based segmentation with n-best list alignment"
echo ""
echo "⚠️  Note: This build may take some time as it downloads conda, models, and dependencies"
echo "🐳 Starting Docker build process..."
echo ""

# Build the Docker image
docker build -t "$IMAGE_NAME" . || {
    echo ""
    echo "❌ Docker build failed!"
    echo ""
    echo "Common issues and solutions:"
    echo "  1. Check if Docker is running: docker ps"
    echo "  2. Ensure you have sufficient disk space (>10GB recommended)"
    echo "  3. If conda TOS issues persist, try: docker system prune"
    echo "  4. Check Docker logs for specific error details"
    echo ""
    exit 1
}

echo ""
echo "✅ Docker image built successfully: $IMAGE_NAME"
echo ""
echo "To use this image with the pipeline, run:"
echo "  nextflow run transcribe.nf -profile docker --in /path/to/audio.wav"
echo ""
echo "The pipeline will now:"
echo "  - Generate n-best alternatives aligned to complete speaker turns"
echo "  - Respect speaker boundaries from diarization"
echo "  - Process each speaker segment individually"
echo ""