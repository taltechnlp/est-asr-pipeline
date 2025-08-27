#!/bin/bash

# Build script for EST ASR Pipeline with Speaker-Turn-Based N-Best
# This creates a new Docker image with the speaker-turn-based segmentation functionality

set -e

IMAGE_NAME="est-asr-pipeline:speaker-turn-nbest"

echo "Building Docker image: $IMAGE_NAME"
echo "This image includes the new speaker-turn-based segmentation with n-best list alignment"
echo ""

# Build the Docker image
docker build -t "$IMAGE_NAME" .

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