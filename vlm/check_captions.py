from pathlib import Path
import pandas as pd

CSV_FILE = Path("datasets/captions.csv")

if not CSV_FILE.exists():
    raise FileNotFoundError(
        "datasets/captions.csv was not found."
    )

df = pd.read_csv(CSV_FILE)

print("=" * 60)
print("CAPTION DATASET CHECK")
print("=" * 60)

print(f"Total images: {len(df)}")
print(f"Columns: {list(df.columns)}")

print("\nFirst 20 captions:")
print("-" * 60)

for i, row in df.head(20).iterrows():

    print(f"\n{i + 1}. {row['image']}")
    print(f"   {row['caption']}")

print("\n" + "=" * 60)

empty = df[
    df["caption"].fillna("").str.strip() == ""
]

print(f"Empty captions: {len(empty)}")

duplicate_images = df[
    df["image"].duplicated()
]

print(
    f"Duplicate image entries: "
    f"{len(duplicate_images)}"
)

duplicate_captions = df[
    df["caption"].duplicated()
]

print(
    f"Duplicate captions: "
    f"{len(duplicate_captions)}"
)

print("=" * 60)