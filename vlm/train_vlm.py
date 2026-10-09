from pathlib import Path
import json
import os
import random

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import (
	BlipForConditionalGeneration,
	BlipProcessor,
)
from tqdm import tqdm


SEED = 42
DATASET_ROOT = Path("datasets")
IMAGE_ROOT = DATASET_ROOT / "images"
CAPTIONS_FILE = DATASET_ROOT / "captions.csv"
OUTPUT_DIR = Path("outputs/vlm-blip")
MODEL_NAME = "Salesforce/blip-image-captioning-base"

MAX_TEXT_LENGTH = 96
TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 4
NUM_EPOCHS = int(os.getenv("VLM_NUM_EPOCHS", "1"))
LEARNING_RATE = float(os.getenv("VLM_LEARNING_RATE", "5e-5"))
MAX_TRAIN_EXAMPLES = int(os.getenv("VLM_MAX_TRAIN_EXAMPLES", "32"))


def set_seed(seed: int) -> None:
	random.seed(seed)
	torch.manual_seed(seed)


def load_examples() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
	if not CAPTIONS_FILE.exists():
		raise FileNotFoundError(f"Caption file not found: {CAPTIONS_FILE.resolve()}")

	frame = pd.read_csv(CAPTIONS_FILE).fillna("")
	required_columns = {"image", "caption"}
	if not required_columns.issubset(frame.columns):
		raise ValueError("captions.csv must contain image and caption columns")

	frame["image"] = frame["image"].astype(str).str.replace("\\", "/")
	frame["caption"] = frame["caption"].astype(str).str.strip()
	frame = frame[(frame["caption"].str.len() >= 12) & frame["image"].ne("")]
	frame = frame.drop_duplicates(subset=["image"])

	examples = []
	for row in frame.itertuples(index=False):
		image_path = IMAGE_ROOT / row.image
		if image_path.is_file():
			examples.append({"image": row.image, "caption": row.caption})

	if len(examples) < 2:
		raise ValueError("At least two valid image-caption pairs are required")

	train, validation = train_test_split(
		examples,
		test_size=0.1,
		random_state=SEED,
	)
	if MAX_TRAIN_EXAMPLES > 0:
		train = train[:MAX_TRAIN_EXAMPLES]
	return train, validation


class CaptionDataset(Dataset):
	def __init__(
		self,
		examples: list[dict[str, str]],
		processor: BlipProcessor,
	):
		self.examples = examples
		self.processor = processor

	def __len__(self) -> int:
		return len(self.examples)

	def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
		example = self.examples[index]
		with Image.open(IMAGE_ROOT / example["image"]) as image:
			image = image.convert("RGB")

		encoded = self.processor(
			images=image,
			text=example["caption"],
			padding="max_length",
			truncation=True,
			max_length=MAX_TEXT_LENGTH,
			return_tensors="pt",
		)
		item = {key: value.squeeze(0) for key, value in encoded.items()}
		item["labels"] = item["input_ids"].clone()
		item["labels"][item["attention_mask"] == 0] = -100
		return item


def main() -> None:
	set_seed(SEED)
	train_examples, validation_examples = load_examples()
	device = "cuda" if torch.cuda.is_available() else "cpu"
	print(f"Training examples: {len(train_examples)}")
	print(f"Validation examples: {len(validation_examples)}")
	print(f"Device: {device}")

	processor = BlipProcessor.from_pretrained(MODEL_NAME, use_fast=True)
	model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME)
	model.to(device)
	train_loader = DataLoader(
		CaptionDataset(train_examples, processor),
		batch_size=TRAIN_BATCH_SIZE,
		shuffle=True,
		num_workers=0,
	)
	validation_loader = DataLoader(
		CaptionDataset(validation_examples, processor),
		batch_size=EVAL_BATCH_SIZE,
		shuffle=False,
		num_workers=0,
	)
	optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

	for epoch in range(NUM_EPOCHS):
		model.train()
		train_loss = 0.0
		for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}"):
			batch = {key: value.to(device) for key, value in batch.items()}
			optimizer.zero_grad(set_to_none=True)
			loss = model(**batch).loss
			loss.backward()
			optimizer.step()
			train_loss += loss.item()

		print(f"epoch={epoch + 1} train_loss={train_loss / len(train_loader):.4f}")
		model.save_pretrained(OUTPUT_DIR)
		processor.save_pretrained(OUTPUT_DIR)

	processor.save_pretrained(OUTPUT_DIR)
	(OUTPUT_DIR / "dataset_summary.json").write_text(
		json.dumps(
			{
				"model": MODEL_NAME,
				"train_examples": len(train_examples),
				"validation_examples": len(validation_examples),
				"seed": SEED,
			},
			indent=2,
		),
		encoding="utf-8",
	)
	print(f"Saved fine-tuned model to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
	main()
