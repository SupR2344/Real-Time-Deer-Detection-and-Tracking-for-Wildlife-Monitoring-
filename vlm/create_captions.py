from pathlib import Path
import pandas as pd
import torch

from PIL import Image
from tqdm import tqdm

from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_DIR = Path("datasets/images")
OUTPUT_FILE = Path("datasets/captions.csv")

MODEL_NAME = "microsoft/Florence-2-base-ft"
TASK_PROMPT = "<MORE_DETAILED_CAPTION>"

BATCH_SIZE = 8
MAX_NEW_TOKENS = 50


# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():

    device = torch.device("cuda")

    print("=" * 60)
    print("GPU DETECTED")
    print("=" * 60)

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

else:

    device = torch.device("cpu")

    print("=" * 60)
    print("WARNING: GPU NOT DETECTED")
    print("=" * 60)

    print(
        "Caption generation will run on CPU "
        "and may be considerably slower."
    )


# ============================================================
# CHECK IMAGE DIRECTORY
# ============================================================

if not IMAGE_DIR.exists():

    raise FileNotFoundError(
        f"Image directory not found:\n{IMAGE_DIR.resolve()}"
    )


# ============================================================
# FIND IMAGES
# ============================================================

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".JPG",
    ".JPEG",
    ".PNG"
}


image_paths = sorted(
    [
        path
        for path in IMAGE_DIR.rglob("*")
        if (
            path.is_file()
            and path.suffix in VALID_EXTENSIONS
        )
    ]
)


print()
print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print(
    "Images found:",
    len(image_paths)
)


if len(image_paths) == 0:

    raise RuntimeError(
        "No images were found inside datasets/images/"
    )


# ============================================================
# LOAD BLIP
# ============================================================

print()
print("=" * 60)
print("LOADING IMAGE CAPTIONING MODEL")
print("=" * 60)

print(
    f"Model: {MODEL_NAME}"
)

processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    attn_implementation="eager",
)

model.to(device)

model.eval()

print("BLIP loaded successfully.")


# ============================================================
# GENERATE CAPTIONS
# ============================================================

results = []

print()
print("=" * 60)
print("GENERATING CAPTIONS")
print("=" * 60)

for start in tqdm(
    range(
        0,
        len(image_paths),
        BATCH_SIZE
    ),
    desc="Processing images"
):

    batch_paths = image_paths[
        start:start + BATCH_SIZE
    ]

    images = []
    valid_paths = []

    # --------------------------------------------------------
    # LOAD IMAGES
    # --------------------------------------------------------

    for image_path in batch_paths:

        try:

            image = Image.open(
                image_path
            ).convert("RGB")

            images.append(image)
            valid_paths.append(image_path)

        except Exception as error:

            print()
            print(
                f"Skipping image: {image_path}"
            )

            print(
                f"Reason: {error}"
            )


    if not images:
        continue


    # --------------------------------------------------------
    # PROCESS IMAGES
    # --------------------------------------------------------

    inputs = processor(
        images=images,
        text=[TASK_PROMPT] * len(images),
        return_tensors="pt"
    )


    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }


    # --------------------------------------------------------
    # GENERATE CAPTIONS
    # --------------------------------------------------------

    with torch.no_grad():

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            num_beams=5,
            repetition_penalty=1.15,
            no_repeat_ngram_size=3
        )


    captions = [
        processor.post_process_generation(
            processor.decode(generated_id, skip_special_tokens=True),
            task=TASK_PROMPT,
            image_size=images[index].size,
        )
        for index, generated_id in enumerate(generated_ids)
    ]


    # --------------------------------------------------------
    # STORE RESULTS
    # --------------------------------------------------------

    for image_path, caption in zip(
        valid_paths,
        captions
    ):

        # Store path relative to datasets/images
        relative_path = image_path.relative_to(
            IMAGE_DIR
        )

        results.append(
            {
                "image": str(
                    relative_path
                ).replace("\\", "/"),

                "caption": caption.strip()
            }
        )


# ============================================================
# CREATE DATAFRAME
# ============================================================

if not results:

    raise RuntimeError(
        "No captions were generated."
    )


captions_df = pd.DataFrame(
    results
)


# ============================================================
# SAVE CSV
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

captions_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)


# ============================================================
# FINISHED
# ============================================================

print()
print("=" * 60)
print("CAPTION GENERATION COMPLETE")
print("=" * 60)

print(
    "Images processed:",
    len(captions_df)
)

print(
    "CSV saved at:",
    OUTPUT_FILE.resolve()
)

print()
print("First 10 captions:")
print()

print(
    captions_df.head(10).to_string(
        index=False
    )
)