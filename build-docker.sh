#!/bin/bash

# Build script for EST ASR Pipeline with Speaker-Turn-Based N-Best
# This creates a new Docker image with the speaker-turn-based segmentation functionality

set -e

IMAGE_NAME="est-asr-pipeline:speaker-turn-nbest"

echo "Building Docker image: $IMAGE_NAME"
echo "This image includes the new speaker-turn-based segmentation with n-best list alignment"
echo ""
echo "⚠️  Note: First time will download base image (europe-north1-docker.pkg.dev/speech2text-218910/repo/est-asr-pipeline:1.1b)"
echo "🐳 Starting Docker build process (extending existing image)..."
echo ""

# Build the Docker image using the speaker-turn-nbest Dockerfile with buildx (fallback to legacy build)
docker buildx build -f Dockerfile.speaker-turn-nbest -t "$IMAGE_NAME" --load . 2>/dev/null || \
docker build -f Dockerfile.speaker-turn-nbest -t "$IMAGE_NAME" . || {
    echo ""
    echo "❌ Docker build failed!"
    echo ""
    echo "Common issues and solutions:"
    echo "  1. Check if Docker is running: docker ps"
    echo "  2. Pull the base image first: docker pull europe-north1-docker.pkg.dev/speech2text-218910/repo/est-asr-pipeline:1.1b"
    echo "  3. Ensure you have sufficient disk space (>5GB recommended)"
    echo "  4. If buildx issues, try: docker buildx install"
    echo "  5. Check Docker logs for specific error details"
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