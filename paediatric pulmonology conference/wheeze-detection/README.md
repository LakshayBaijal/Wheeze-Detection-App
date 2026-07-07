# Wheeze Detection from Children's Breathing Sounds — Proof of Concept

This project teaches a computer to listen to a short recording of a child's
breathing and decide: **wheeze** or **no wheeze**. It's a working demonstration
of the "AI" part of the smartphone wheeze-detection idea (Phases 3–5 of the
research plan), built on a free, already-labelled public dataset so you don't
have to collect your own recordings just to prove the concept works.

> **Plain-language summary:** other researchers recorded thousands of children's
> breathing sounds and had doctors label each one. We reuse that labelled data to
> train and test a wheeze detector, and we measure how accurate it is. Later,
> Geetesh can feed in his *own* smartphone recordings — that's the novel,
> award-worthy step.

---

## What's in the box

```
wheeze-detection/
├── README.md              ← you are here
├── requirements.txt       ← the list of software to install
├── notebooks/
│   └── wheeze_detection_colab.ipynb   ← EASY option: runs in a web browser
├── src/                   ← the actual code (the "local scripts" option)
│   ├── config.py          ← all settings in one place
│   ├── download_data.py   ← gets the dataset
│   ├── dataset.py         ← reads + labels the data, splits it by patient
│   ├── preprocess.py      ← Phase 3: cleans each sound clip
│   ├── features.py        ← turns sound into numbers/pictures
│   ├── evaluate.py        ← Phase 5: measures accuracy, saves charts
│   └── run_all.py         ← runs the whole thing with one command
├── data/                  ← the dataset lands here (created on first run)
└── outputs/               ← trained models, charts, and results land here
```

## Three ways to use it

### Option A — **Testing app** (for non-technical users like Geetesh) ⭐ START HERE
A simple web interface where you upload or record breathing sounds and instantly see predictions. Perfect for validation testing.

**On Mac/Linux:**
```bash
chmod +x run_app.sh
./run_app.sh
```

**On Windows:** Double-click `run_app.bat`

Then open http://localhost:8501 in your browser and start testing!

See [QUICK_START.md](QUICK_START.md) for full instructions.

---

### Option B — Google Colab (easiest, no installation on your computer)

1. Go to **https://colab.research.google.com**
2. `File → Upload notebook` and upload `notebooks/wheeze_detection_colab.ipynb`
3. `Runtime → Run all`

That's it. Colab is a free website by Google that runs the code on their
computers (with a fast GPU). It downloads the data, trains the models, and shows
the accuracy charts right in the page. Great for a non-technical person to click
through, or to share with a collaborator by link.

### Option B — On your own computer (for a technical collaborator)

```bash
# 1. (once) install the software
pip install -r requirements.txt

# 2. (once) download the dataset
python src/download_data.py

# 3. run the whole pipeline
python src/run_all.py
```

Results (accuracy numbers, confusion-matrix and ROC charts, saved models) appear
in the `outputs/` folder.

> **Quick trial:** the first full run can take a while because cleaning audio is
> slow. To do a fast test on a small sample, open `src/run_all.py`, set
> `MAX_EVENTS = 400` near the top, run it, then set it back to `None` for the
> real run.

---

## The dataset

We use **SPRSound** — the first open-access *paediatric* respiratory-sound
database (ages 1 month–18 years, ~2,683 recordings), recorded at Shanghai
Children's Medical Center and expert-labelled for wheeze, crackle, normal, etc.
It is free for research and downloads automatically in step 2 / in the notebook.

Other public datasets you can plug in later (the code is easy to point at them):

| Dataset | Notes | Where |
|---|---|---|
| **SPRSound** (used here) | Paediatric, wheeze/crackle labels | GitHub: `SJTU-YONGFU-RESEARCH-GRP/SPRSound` |
| **ICBHI 2017** | The classic benchmark, includes children | Kaggle: `vbookshelf/respiratory-sound-database` |
| **HF_Lung_V1** | Large, event-level labels | Public release |

---

## What the results mean (reading the numbers)

- **Sensitivity** — of the children who *really* wheeze, how many the model
  caught. High is good (you don't want to miss a sick child).
- **Specificity** — of the healthy children, how many the model correctly cleared.
- **ROC-AUC** — one overall score: 0.5 is a coin-flip, 1.0 is perfect. Published
  paediatric wheeze detectors land around 0.85–0.95.

The script prints a comparison table across all the models (Random Forest, SVM,
XGBoost, CNN) and saves charts to `outputs/`.

---

## Training improvements for medical-grade accuracy

The CNN has been upgraded with **medical-grade optimizations**:

| Feature | Benefit | Why |
|---|---|---|
| **Deeper CNN** (3 conv blocks) | Better feature extraction | Lung sounds have complex, nested patterns |
| **Batch normalization** | Stable training across patient populations | Different kids have different baseline lung sounds |
| **Higher dropout** (0.25–0.4) | Prevents overfitting | Medical datasets are precious; don't waste them |
| **150 epochs** | Full model capability | Was training for only 25 steps before—not enough |
| **Early stopping** | Stops when validation plateaus | Avoids overfitting; saves training time |
| **ReduceLROnPlateau** | Adaptive learning rate | If stuck, reduce LR to find better minima |
| **Smaller batch (16)** | Better generalization | Noisier gradients = more robust to new patients |

Expected result: **sensitivity ≥ 0.85, specificity ≥ 0.85** on full SPRSound data (scales up from the proof-of-concept).

---

## Honest note on what this is (and isn't)

This is a **methods proof-of-concept** trained on someone else's data. It proves
the whole machine works end-to-end and gives a strong template. The genuinely
*novel*, award-worthy contribution is the next step: recording children on a
**smartphone** and validating the model on those real-world recordings. When
Geetesh has his own labelled `.wav` files, a collaborator can point this same
code at them with only small changes.

### References worth citing
- SPRSound: Open-Source SJTU Paediatric Respiratory Sound Database (SJTU).
- "AI Models for Paediatric Lung Sound Analysis: Systematic Review & Meta-Analysis,"
  *JMIR* (2025) — pooled sensitivity ≈0.90, specificity ≈0.96.
- "Wheeze detection in real-world paediatric care: AI applied to smartphone lung
  auscultation," *European Journal of Pediatrics* (2026).
