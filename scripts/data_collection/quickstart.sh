#!/bin/bash
# Linux/Mac Quick Start Script for Data Collection
# =================================================

set -e  # Exit on error

echo "========================================"
echo "Data Collection Quick Start (Linux/Mac)"
echo "========================================"
echo ""

# Check Python
echo "[1/5] Checking Python..."
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python3 not found!"
    echo "Please install Python 3.8+ from https://www.python.org/"
    exit 1
fi
python3 --version
echo "OK: Python installed"
echo ""

# Install dependencies
echo "[2/5] Installing dependencies..."
pip3 install -r requirements.txt || {
    echo "ERROR: Failed to install dependencies"
    exit 1
}
echo "OK: Dependencies installed"
echo ""

# Check .env file
echo "[3/5] Checking .env file..."
if [ ! -f .env ]; then
    echo "WARNING: .env file not found!"
    echo "Creating template .env file..."
    cat > .env << EOF
REDDIT_CLIENT_ID=DgmKFCNd5jIDZGZmx7HcUg
REDDIT_CLIENT_SECRET=BtY3O1iq_emy_yyA73gdHzUqvtGgig
REDDIT_USER_AGENT=DeepfakeDataCollector/1.0
COMFYUI_URL=http://127.0.0.1:8188
OUTPUT_DIR=../../dataset
FFHQ_COUNT=10000
CELEBAHQ_COUNT=8000
YOUTUBE_COUNT=7000
REDDIT_COUNT=3000
GALLERY_COUNT=2000
FLUX_COUNT=3000
EOF
    echo "OK: .env file created"
else
    echo "OK: .env file exists"
fi
echo ""

# Setup Kaggle API
echo "[4/5] Setting up Kaggle API..."
mkdir -p ~/.kaggle
if [ -f kaggle.json ]; then
    cp kaggle.json ~/.kaggle/kaggle.json
    chmod 600 ~/.kaggle/kaggle.json
    echo "OK: Kaggle API configured"
else
    echo "WARNING: kaggle.json not found"
    echo "Download from https://www.kaggle.com/settings"
fi
echo ""

# Check ffmpeg (optional for YouTube)
echo "[Optional] Checking ffmpeg..."
if command -v ffmpeg &> /dev/null; then
    echo "OK: ffmpeg installed"
else
    echo "WARNING: ffmpeg not found (needed for YouTube crawling)"
    echo "Install: sudo apt install ffmpeg (Ubuntu) or brew install ffmpeg (Mac)"
fi
echo ""

# Run data collection
echo "[5/5] Starting data collection..."
echo ""
echo "WARNING: This will take 25-34 hours!"
echo "- Real datasets: 3-5 hours"
echo "- YouTube crawling: 3-5 hours"
echo "- Reddit crawling: 1-2 hours"
echo "- Galleries crawling: 1-2 hours"
echo "- ComfyUI generation: SKIPPED (use without --skip-comfyui to include)"
echo ""
echo "Press Ctrl+C to cancel, or Enter to continue..."
read

python3 run_all.py --skip-comfyui

echo ""
echo "========================================"
echo "Data Collection Complete!"
echo "========================================"
echo ""
echo "Check results in: ../../dataset"
echo "Metadata: ../../dataset/metadata/combined_dataset.csv"
echo ""

