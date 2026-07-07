#!/bin/bash
# Double-click this file to run the Wheeze Detection app on your own Mac.
# First run sets things up (a few minutes); after that it starts in seconds.

cd "$(dirname "$0")"

echo "==================================================="
echo "  Wheeze Detection - starting on your computer"
echo "==================================================="
echo ""

# Create a private environment the first time only.
if [ ! -d venv ]; then
  echo "First-time setup: building the environment."
  echo "This downloads the needed software and takes ~3-5 minutes."
  echo "Please wait..."
  echo ""
  python3 -m venv venv
  ./venv/bin/pip install --upgrade pip
  ./venv/bin/pip install streamlit numpy scipy scikit-learn soundfile librosa joblib plotly
  echo ""
  echo "Setup complete!"
fi

echo ""
echo "Opening the app in your web browser..."
echo "(To stop the app later: come back to this window and press Control + C)"
echo ""
./venv/bin/streamlit run src/app.py

# Keep the window open if something goes wrong.
echo ""
read -p "Press Enter to close this window..."
