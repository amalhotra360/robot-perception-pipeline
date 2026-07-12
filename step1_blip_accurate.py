from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

print("📥 Loading BLIP-2 model (only ~350MB)...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# Load image
image = Image.open('test_image1.jpeg')

# Generate description
print("\n🔍 Analyzing image with BLIP-2...\n")
inputs = processor(image, return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=300)
caption = processor.decode(out[0], skip_special_tokens=True)

print("📝 What BLIP-2 sees:")
print(caption)

# Filter false positives
print("\n" + "="*50)
print("\n✅ Objects in the image:")

keywords = {
    'cup': ['cup', 'mug', 'glass'],
    'potted plant': ['plant', 'pot', 'potted', 'flower'],
    'scissors': ['scissors', 'scissor'],
    'cell phone': ['phone', 'cell', 'mobile'],
}

found = []
for obj, kw_list in keywords.items():
    for kw in kw_list:
        if kw.lower() in caption.lower():
            found.append(obj)
            break

for obj in found:
    print(f"  ✓ {obj}")

# Show what was correctly rejected
print("\n❌ False positives that were filtered:")
rejected = ['mouse', 'donut', 'carrot', 'remote', 'clock']
for item in rejected:
    if item.lower() not in caption.lower():
        print(f"  • Rejected: {item}")