# Quick Start Guide — Wheeze Detection Testing App

## For Dr. Geetesh Baijal (Non-Technical Users)

This guide shows how to test the wheeze detection model on your own patient recordings using a simple web interface. **No coding required.**

---

## Before You Start

1. Make sure you have **Python 3.10+** installed on your computer
   - Download from https://www.python.org/downloads/
   - On Mac: use Homebrew (`brew install python3`)

2. Have the trained model files ready (you should have downloaded them from Colab)

---

## Step 1: Install the Software (One-Time Setup)

### On Windows:
```bash
cd Desktop/Personal\ Work/Geetesh\ Baijal/paediatric\ pulmonology\ conference/wheeze-detection
pip install -r requirements.txt
```

### On Mac:
```bash
cd ~/Desktop/"Personal Work"/"Geetesh Baijal"/"paediatric pulmonology conference"/wheeze-detection
pip install -r requirements.txt
```

### On Linux:
```bash
cd ~/Desktop/Personal\ Work/Geetesh\ Baijal/paediatric\ pulmonology\ conference/wheeze-detection
pip install -r requirements.txt
```

⏳ **This takes 3–5 minutes.** Grab a coffee.

---

## Step 2: Start the Web App

Run this command from the same folder:

```bash
streamlit run src/app.py
```

You'll see:
```
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
```

A browser window will automatically open. If not, go to **http://localhost:8501**

---

## Step 3: Test on Patient Recordings

### Option A: Upload a recording
1. Click **"Upload a file"**
2. Select a `.wav` file (2 seconds of breathing sound)
3. Wait 1–2 seconds
4. See the result: **🔴 WHEEZE DETECTED** or **🟢 No Wheeze**

### Option B: Record directly
1. Click **"Record a clip"** (requires browser microphone access)
2. Hold your phone/microphone to patient's chest
3. Record 2 seconds of breathing
4. Click stop
5. See the result instantly

### After Each Test
- You'll see: **Confidence percentage** (how sure the model is)
- A chart showing prediction strength
- **Download button** to save the result as a file

---

## Workflow for Validation Testing

### Day 1: Get ~50 recordings
1. Record 50 patient breathing clips on your phone (or use a digital stethoscope)
2. Label each one: wheeze or no-wheeze (based on your clinical assessment)
3. Save them as `.wav` files with names like: `patient001_wheeze.wav`, `patient002_no_wheeze.wav`

### Day 2–3: Test the model
1. Start the app (run the command above)
2. For each recording:
   - Upload the file
   - Note the model's prediction
   - Compare it to your clinical assessment
   - Download the result

### Day 4: Analyze
- Open the `results/` folder (in your wheeze-detection folder)
- Count how many the model got right vs wrong
- Calculate accuracy = (correct / total) × 100%

---

## Troubleshooting

### "streamlit: command not found"
**Solution:** The software didn't install properly. Try:
```bash
python -m pip install streamlit plotly
python -m streamlit run src/app.py
```

### "Model not found"
**Solution:** The trained model needs to be in the `outputs/` folder. Make sure you:
1. Ran the full training pipeline (downloaded from Colab)
2. Extracted all files to this folder
3. Check that `outputs/model_random_forest.joblib` exists

### Audio won't upload
**Solution:** Make sure the file is `.wav` format. If it's `.mp3`, convert it first:
- Use an online converter like https://cloudconvert.com
- Or use ffmpeg: `ffmpeg -i file.mp3 file.wav`

### Slow predictions
**Solution:** This is normal for the first prediction (model is loading). After that, predictions are instant.

---

## Tips for Best Results

✅ **Do:**
- Record at the same location (e.g., right anterior chest, left posterior)
- Use the same microphone/stethoscope each time
- Record during quiet breathing (not coughing or talking)
- Aim for exactly 2 seconds

❌ **Don't:**
- Record with loud background noise
- Move the microphone during recording
- Use poor-quality microphones

---

## Next Steps

Once you've tested 50–100 recordings and calculated accuracy:
1. Share your results with your collaborators
2. Write up the validation as a conference abstract
3. If accuracy is high (>80%), you're ready for the full study

---

## Questions?

If something doesn't work, check:
1. Is Python installed? → `python --version`
2. Are packages installed? → `pip list | grep streamlit`
3. Is the model file there? → Check the `outputs/` folder

---

**That's it!** You now have a tool to test the model on real patient data. No coding required. 🎉
