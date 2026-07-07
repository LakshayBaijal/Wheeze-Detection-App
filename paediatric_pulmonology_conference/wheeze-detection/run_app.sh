#!/bin/bash
# Mac/Linux: Double-click this file to start the testing app

cd "$(dirname "$0")"
echo "Starting Wheeze Detection Testing App..."
echo ""
echo "⏳ If this is your first time, Python packages are installing (takes ~1 min)"
echo ""

# Install requirements if not already done
pip install -q -r requirements.txt 2>/dev/null

# Start the app
echo "🚀 Opening the app in your browser..."
streamlit run src/app.py

# Keep the terminal open if there's an error
read -p "Press Enter to close..."
