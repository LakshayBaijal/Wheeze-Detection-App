"""
Central settings for the whole project.

Everything you might want to change (folders, audio settings, model choices)
lives here so you don't have to hunt through the other files.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# FOLDERS
# ---------------------------------------------------------------------------
# The project root = the folder that contains this "src" folder.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"          # where the downloaded dataset lives
SPRSOUND_DIR = DATA_DIR / "SPRSound"      # the cloned SPRSound GitHub repo
OUTPUT_DIR = PROJECT_ROOT / "outputs"     # models, plots, and results go here

# Which SPRSound split to use. BioCAS2022 has the biggest labelled TRAIN set.
WAV_SUBDIR = SPRSOUND_DIR / "BioCAS2022" / "train2022_wav"
JSON_SUBDIR = SPRSOUND_DIR / "BioCAS2022" / "train2022_json"

# ---------------------------------------------------------------------------
# AUDIO / PREPROCESSING (Phase 3)
# ---------------------------------------------------------------------------
SAMPLE_RATE = 8000        # SPRSound is recorded at 8 kHz; we keep it consistent
CLIP_SECONDS = 2.0        # each respiratory event is cut/padded to this length
BANDPASS_LOW_HZ = 80      # lung sounds mostly live between ~80 Hz ...
BANDPASS_HIGH_HZ = 2000   # ... and ~2000 Hz; filter out the rest

# ---------------------------------------------------------------------------
# FEATURES
# ---------------------------------------------------------------------------
N_MFCC = 40               # number of MFCC coefficients (a common choice)
N_MELS = 64               # mel bands for the spectrogram image (for the CNN)

# ---------------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------------
RANDOM_SEED = 42          # fixed so results are reproducible
TEST_FRACTION = 0.15      # share of PATIENTS held out for the final test
VAL_FRACTION = 0.15       # share of PATIENTS used for validation
CNN_EPOCHS = 150          # with early stopping, will stop earlier if validation plateaus
CNN_BATCH_SIZE = 16       # smaller batch = noisier gradients = better generalization

# The two classes we predict first (binary task).
CLASSES = ["no_wheeze", "wheeze"]

# Event labels in SPRSound JSON that count as "wheeze" for our binary task.
# (Wheeze and Wheeze+Crackle both contain a wheeze.)
WHEEZE_EVENT_LABELS = {"Wheeze", "Wheeze+Crackle"}
