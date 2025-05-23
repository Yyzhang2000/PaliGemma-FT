from PIL import Image
import torch
import fire


from typing import Optional

from process_images import PaliGemmaProcessor
from model.kv_cache import KVCache
from model.paligemma import PaliGemmaForConditionalGeneration

from load_weights import load_hf_model


def move_inputs_to_device(model_inputs: dict, device: str):
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
    return model_inputs


def get_model_inputs(
    processor: PaliGemmaProcessor, prompt: str, image_file_path: str, device: str
):
    image = Image.open(image_file_path)
    images = [image]
    prompts = [prompt]
    model_inputs = processor(text=prompts, images=images)
    model_inputs = move_inputs_to_device(model_inputs, device)
    return model_inputs


def sample_top_p(
    probs: torch.Tensor,
    top_p: float,
):
    probs_sort, probs_idx = torch.sort(probs, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)

    mask = probs_sum - probs_sort > top_p

    probs_sort[mask] = 0.0

    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token_id = torch.multinomial(probs_sort, num_samples=1)
    next_token_id = probs_idx.gather(-1, next_token_id)

    return next_token_id


def inference(
    model: PaliGemmaForConditionalGeneration,
    processor: PaliGemmaProcessor,
    device: str,
    prompt: Optional[str],
    image_file_path: Optional[str],
    max_tokens_to_generate: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
):
    model_inputs = get_model_inputs(processor, prompt, image_file_path, device)
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    pixel_values = model_inputs["pixel_values"]

    kv_cache = KVCache()

    stop_token = processor.tokenizer.eos_token_id
    generated_tokens = []

    if do_sample:
        print("Top-P Sampling")
    else:
        print("Greedy")

    for _ in range(max_tokens_to_generate):
        outputs = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
        )

        kv_cache = outputs["kv_cache"]
        next_token_logits = outputs["logits"][:, -1, :]

        if do_sample:
            next_token_logits = torch.softmax(next_token_logits / temperature, dim=-1)
            next_token_id = sample_top_p(next_token_logits, top_p=top_p)
        else:
            next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        next_token_id = next_token_id.squeeze(0)
        generated_tokens.append(next_token_id)
        if next_token_id == stop_token:
            break
        # Append the next token to the input_ids and attention_mask
        input_ids = next_token_id.unsqueeze(0)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), device=input_ids.device)], dim=-1
        )

        tempt_text = torch.cat(generated_tokens, dim=-1)
        decoded = processor.tokenizer.decode(tempt_text, skip_special_tokens=True)
        print(prompt + decoded)

    generated_tokens = torch.cat(generated_tokens, dim=-1)
    decoded = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    print(prompt + decoded)


def main(
    model_path: str = "./weights",
    prompt: Optional[str] = None,
    image_file_path: Optional[str] = None,
    max_tokens_to_generate: int = 100,
    temperature: float = 0.8,
    top_p: float = 0.9,
    do_sample: bool = False,
    only_cpu: bool = False,
):
    device = "cpu"

    if not only_cpu:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"

    print("Device in use: ", device)

    print(f"Loading model")
    model, tokenizer = load_hf_model(model_path=model_path, device=device)
    model = model.to(device).eval()
    print("Model Loaded successful")

    num_image_tokens = model.config.vision_config.num_image_tokens
    image_size = model.config.vision_config.image_size
    processor = PaliGemmaProcessor(tokenizer, num_image_tokens, image_size)

    print("Running inference")
    with torch.no_grad():
        inference(
            model,
            processor,
            device,
            prompt,
            image_file_path,
            max_tokens_to_generate,
            temperature,
            top_p,
            do_sample,
        )


if __name__ == "__main__":
    fire.Fire(main)
