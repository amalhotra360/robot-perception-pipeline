from PIL import Image
import requests
from transformers import AutoProcessor, LlavaForConditionalGeneration
import torch

print("📥 Loading LLaVA model (first time: ~1-2 min)...")
model_id = "llava-hf/llava-1.5-7b-hf"
processor = AutoProcessor.from_pretrained(model_id)
model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load image
image = Image.open('test_image1.jpeg')

# Step 1: Get detailed description
print("\n🔍 Analyzing image with LLaVA...\n")
prompt = "What objects are in this image? List each object you can see."
inputs = processor(prompt, image, return_tensors="pt").to(model.device)

output = model.generate(**inputs, max_new_tokens=200)
description = processor.decode(output[0], skip_special_tokens=True)

# Extract just the response (remove prompt)
response = description.split("ASSISTANT:")[-1].strip()
print("📝 What LLaVA sees:")
print(response)

# Step 2: Ask it to filter out uncertain objects
print("\n" + "="*50)
prompt2 = """Now, look at your description above.
Only list objects you are CONFIDENT are actually in the image.
Remove anything you're uncertain about or might be a shadow/reflection.

Final objects:"""

inputs2 = processor(prompt2, image, return_tensors="pt").to(model.device)
output2 = model.generate(**inputs2, max_new_tokens=100)
filtered = processor.decode(output2[0], skip_special_tokens=True)

print("\n✅ Confident objects only:")
print(filtered.split("ASSISTANT:")[-1].strip())