import os
import glob
import pandas as pd

# ============================================================
# TASK 1: Read all 1440 audio files (sorted for reproducibility)
# ============================================================
DATASET_PATH = "/Users/devanshbansal/Desktop/ser/dataset/Audio_Speech_Actors_01-24"  # <-- update this to your dataset folder

audio_files = sorted(
    glob.glob(os.path.join(DATASET_PATH, "**", "*.wav"), recursive=True)
)

print(f"Total audio files found: {len(audio_files)}")
assert len(audio_files) == 1440, "Expected 1440 files, check your dataset path!"

# ============================================================
# TASK 2: Extract full metadata from each filename
# ============================================================
# RAVDESS filename format (7 numeric identifiers separated by '-'):
# modality-vocal_channel-emotion-intensity-statement-repetition-actor.wav
# Example: 03-01-06-01-02-01-12.wav

emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

intensity_map = {
    "01": "normal",
    "02": "strong"
}

statement_map = {
    "01": "Kids are talking by the door",
    "02": "Dogs are sitting by the door"
}

def extract_metadata(file_path):
    filename = os.path.basename(file_path)
    name_no_ext = filename.replace(".wav", "")
    parts = name_no_ext.split("-")

    if len(parts) != 7:
        return None  # skip malformed filenames

    modality_code, vocal_channel_code, emotion_code, intensity_code, \
        statement_code, repetition_code, actor_code = parts

    actor_num = int(actor_code)
    gender = "female" if actor_num % 2 == 0 else "male"

    return {
        "file_path": file_path,
        "filename": filename,
        "emotion": emotion_map.get(emotion_code, "unknown"),
        "emotion_code": emotion_code,
        "actor": actor_num,
        "gender": gender,
        "intensity": intensity_map.get(intensity_code, "unknown"),
        "statement": int(statement_code),
        "statement_text": statement_map.get(statement_code, "unknown"),
        "repetition": int(repetition_code),
        "vocal_channel": "speech" if vocal_channel_code == "01" else "song",
        "modality": {"01": "full_AV", "02": "video_only", "03": "audio_only"}.get(modality_code, "unknown")
    }

# ============================================================
# TASK 3: Create a DataFrame
# ============================================================
records = [extract_metadata(fp) for fp in audio_files]
records = [r for r in records if r is not None]  # drop malformed entries

df = pd.DataFrame(records)

# Reorder columns to match the recommended layout
df = df[[
    "file_path", "filename", "emotion", "emotion_code",
    "actor", "gender", "intensity", "statement",
    "statement_text", "repetition", "vocal_channel", "modality"
]]

print("\nSample rows:")
print(df.head())
print(f"\nDataFrame shape: {df.shape}")

# ============================================================
# TASK 4: Verify the dataset
# ============================================================
print("\n--- Verification ---")

# Row count
print("Total rows:", len(df))
assert len(df) == 1440, "Row count mismatch!"

# Unknown/missing emotion check
print("Unknown emotion count:", (df["emotion"] == "unknown").sum())

# Unique emotion labels (should be exactly 8, alphabetically listed)
print("\nUnique emotions:")
print(sorted(df["emotion"].unique()))

# Emotion distribution (in official RAVDESS order, not frequency order)
emotion_order = [
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprised"
]

print("\nEmotion distribution:")
print(df["emotion"].value_counts().reindex(emotion_order))

# Gender distribution
print("\nGender distribution:")
print(df["gender"].value_counts())

# Intensity distribution
print("\nIntensity distribution:")
print(df["intensity"].value_counts())

# Duplicate file check
print("\nDuplicate files:", df.duplicated("file_path").sum())

# Statement balance check (should be 720 / 720)
print("\nStatement distribution:")
print(df["statement_text"].value_counts())

# Actor list and count
print("\nActors present:", sorted(df["actor"].unique()))
print("Number of unique actors:", df["actor"].nunique())

# Folder/actor consistency check (every actor should have 60 files)
print("\nFiles per actor:")
print(df.groupby("actor").size().sort_index())

incomplete_actors = df.groupby("actor").size().sort_index()
incomplete_actors = incomplete_actors[incomplete_actors != 60]
if len(incomplete_actors) > 0:
    print("\nWARNING: Actors with incomplete file counts:")
    print(incomplete_actors)
else:
    print("\nAll actors have exactly 60 files. Dataset is complete.")

# Null value check
print("\nMissing values:\n", df.isnull().sum())

# Confirm files actually exist on disk
missing_files = df[~df["file_path"].apply(os.path.exists)]
print(f"\nMissing files on disk: {len(missing_files)}")

# ============================================================
# TASK 5: Save for next stages
# ============================================================
OUTPUT_DIR = "../../outputs"  # <-- adjust relative path to fit your project structure

os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_CSV = os.path.join(OUTPUT_DIR, "ravdess_dataset.csv")
OUTPUT_PKL = os.path.join(OUTPUT_DIR, "ravdess_dataset.pkl")

df.to_csv(OUTPUT_CSV, index=False)
df.to_pickle(OUTPUT_PKL)  # preserves dtypes for later stages

print(f"\nSaved dataset to '{OUTPUT_CSV}' and '{OUTPUT_PKL}'")

# ============================================================
# COMPLETION SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("DATASET VERIFICATION COMPLETED SUCCESSFULLY")
print("=" * 60)
print(f"Total Samples   : {len(df)}")
print(f"Unique Actors   : {df['actor'].nunique()}")
print(f"Unique Emotions : {df['emotion'].nunique()}")
print("Dataset Status  : READY FOR PREPROCESSING")
print("=" * 60)