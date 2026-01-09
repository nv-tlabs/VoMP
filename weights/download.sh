#!/bin/bash

set -e

REPO_URL="https://huggingface.co/JeffreyXiang/TRELLIS-image-large"
TARGET_DIR="weights/TRELLIS-image-large"

if ! command -v git-lfs &> /dev/null; then
    echo "ERROR: git-lfs is not installed."
    exit 1
fi

echo "Cloning repository from $REPO_URL to $TARGET_DIR..."
git clone $REPO_URL $TARGET_DIR
