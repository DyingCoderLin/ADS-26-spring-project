# Load model directly — streaming generation (stdout) via Transformers TextStreamer.
# See: https://github.com/huggingface/transformers/blob/main/src/transformers/generation/streamers.py
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    TextStreamer,
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen3.5-0.8B")
model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3.5-0.8B")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Please tell me a short story."},
        ],
    },
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

tokenizer = getattr(processor, "tokenizer", processor)
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

model.generate(
    **inputs,
    streamer=streamer,
    max_new_tokens=4096,
    do_sample=True,
    temperature=0.7,
)
