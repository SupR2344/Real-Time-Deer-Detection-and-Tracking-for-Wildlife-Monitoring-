from pathlib import Path

import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor


MODEL_DIR = Path("outputs/vlm-blip")
IMAGE_ROOT = Path("datasets/images")
IMAGE_NAME = "03d871997f.jpg"


def main() -> None:
	if not MODEL_DIR.exists():
		raise FileNotFoundError(
			f"Trained model not found at {MODEL_DIR.resolve()}. Run train_vlm.py first."
		)

	processor = BlipProcessor.from_pretrained(MODEL_DIR)
	model = BlipForConditionalGeneration.from_pretrained(MODEL_DIR)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)
	model.eval()

	image_path = IMAGE_ROOT / IMAGE_NAME
	with Image.open(image_path) as image:
		inputs = processor(images=image.convert("RGB"), return_tensors="pt")
	inputs = {key: value.to(device) for key, value in inputs.items()}
	with torch.no_grad():
		output = model.generate(
			**inputs,
			max_new_tokens=96,
			num_beams=5,
			repetition_penalty=1.15,
			no_repeat_ngram_size=3,
		)
	print(processor.decode(output[0], skip_special_tokens=True).strip())


if __name__ == "__main__":
	main()
