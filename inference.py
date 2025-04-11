from model.paligemma import PaliGemmaForConditionalGeneration
from model.config import PaliGemmaConfig, SiglipVisionConfig, GemmaConfig

from transformers import AutoTokenizer
import json
import glob
from safetensors import safe_open
from typing import Tuple
import os


def load_hf_model(
    model_id: str = "google/paligemma-3b-pt-224",
    model_path="./weights",
    device: str = "cpu",
) -> Tuple[PaliGemmaForConditionalGeneration, AutoTokenizer]:
    # Load the tokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="right")
    assert tokenizer.padding_side == "right"

    # # Find all the *.safetensors files
    safetensors_files = glob.glob(os.path.join(model_path, "*.safetensors"))

    # # ... and load them one by one in the tensors dictionary
    tensors = {}
    for safetensors_file in safetensors_files:
        with safe_open(safetensors_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

    config = PaliGemmaConfig(
        text_config=GemmaConfig(),
        vision_config=SiglipVisionConfig(),
    )

    model = PaliGemmaForConditionalGeneration(config).to(device)

    # Load the state dict of the model
    model.load_state_dict(tensors, strict=False)

    # Tie weights
    model.tie_weights()

    return (model, tokenizer)


if __name__ == "__main__":
    model, tokenizer = load_hf_model(model_path="./weights", device="cpu")
    print("Model loaded successfully")
