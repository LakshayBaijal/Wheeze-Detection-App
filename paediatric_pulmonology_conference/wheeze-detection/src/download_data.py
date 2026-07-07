"""
Step 0 - Get the dataset.

This downloads the SPRSound paediatric respiratory-sound database (free, for
research) by cloning its public GitHub repository into the project's data folder.

Run it once:
    python src/download_data.py

If you don't have `git`, you can instead download the ZIP by hand from
    https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound
and unzip it into   wheeze-detection/data/SPRSound
"""

import subprocess
import sys

from config import DATA_DIR, SPRSOUND_DIR, WAV_SUBDIR

REPO_URL = "https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound.git"


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if SPRSOUND_DIR.exists():
        print(f"[skip] SPRSound already present at: {SPRSOUND_DIR}")
    else:
        print(f"[clone] Downloading SPRSound into {SPRSOUND_DIR} ...")
        print("        (this is a few hundred MB and may take a few minutes)")
        # --depth 1 = only the latest version, so it downloads faster.
        result = subprocess.run(
            ["git", "clone", "--depth", "1", REPO_URL, str(SPRSOUND_DIR)]
        )
        if result.returncode != 0:
            sys.exit(
                "\n[error] git clone failed. Either install git, or download the "
                "ZIP manually (see the note at the top of this file)."
            )

    # Sanity check that the expected audio folder is there.
    if WAV_SUBDIR.exists():
        n_wav = len(list(WAV_SUBDIR.glob("*.wav")))
        print(f"[ok] Found {n_wav} .wav files in {WAV_SUBDIR}")
    else:
        print(
            f"[warn] Expected folder not found: {WAV_SUBDIR}\n"
            "       The repo layout may have changed - open the data/SPRSound "
            "folder and update WAV_SUBDIR/JSON_SUBDIR in src/config.py."
        )


if __name__ == "__main__":
    main()
