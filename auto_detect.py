"""
Fully automatic object detection — no fixed word list, no typing from me.

The pipeline (the program invents its own vocabulary from the photo):
  Step 1: BLIP looks at the photo and freely describes it, a few different
          ways (no human prompting).
  Step 2: The program pulls the object words out of those sentences
          ("a red desk lamp, a phone and a cactus" -> lamp, phone, cactus).
  Step 3: OWL-ViT searches the photo for each of those words and returns
          a BOX for every one it finds (this replaces YOLO's 80-word limit).
  Step 4: If a person is in the photo, BLIP describes what they're DOING.

Also saves a copy of the photo with the boxes drawn on it: result_<name>.jpg

How to run it:
  python auto_detect.py                  <- uses test_image1.jpeg
  python auto_detect.py kitchen.jpg      <- uses your own picture
"""

import sys
import re
import torch
from PIL import Image, ImageDraw
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    OwlViTProcessor, OwlViTForObjectDetection,
)

# ---------- Settings you can play with ----------
BOX_CUTOFF = 0.15      # how sure OWL-ViT must be to keep a box (0.1 = loose, 0.3 = strict)
NUM_CAPTIONS = 3       # how many different descriptions to ask BLIP for

# ---------- Pick which picture to analyze ----------
image_path = sys.argv[1] if len(sys.argv) > 1 else 'test_image1.jpeg'
print(f"🖼️  Analyzing: {image_path}")

# ---------- Load the models ----------
print("\n(loading models, this takes a moment...)")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
blip = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
owl_processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
owl = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

image = Image.open(image_path).convert('RGB')


def blip_describe(img, starter_text=None, creative=False):
    """Ask BLIP to describe a picture (or finish a sentence about it)."""
    if starter_text:
        inputs = blip_processor(img, text=starter_text, return_tensors="pt")
    else:
        inputs = blip_processor(img, return_tensors="pt")
    if creative:
        # let BLIP be a little random, so repeated runs mention different things
        out = blip.generate(**inputs, max_new_tokens=40, do_sample=True, top_p=0.9)
    else:
        out = blip.generate(**inputs, max_new_tokens=40)
    return blip_processor.decode(out[0], skip_special_tokens=True)


# ---------- STEP 1: BLIP freely describes the photo ----------
print("\n--- STEP 1: BLIP describes the photo (no prompting) ---")

captions = [blip_describe(image)]                       # one careful description
for _ in range(NUM_CAPTIONS - 1):
    captions.append(blip_describe(image, creative=True))  # extra creative ones

for cap in captions:
    print(f"  \"{cap}\"")

# ---------- STEP 2: Pull the object words out of the sentences ----------
print("\n--- STEP 2: Extracting object words from the descriptions ---")


def extract_object_phrases(caption):
    """Chop a sentence like 'there is a red lamp, a phone and a cactus'
    into object phrases: ['red lamp', 'phone', 'cactus']"""
    text = caption.lower().strip(' .')
    # remove sentence starters
    text = re.sub(r'^(there (is|are)|this is|it is|a photo of|an image of)\s*', '', text)
    # split into chunks wherever objects are usually separated
    chunks = re.split(r',| and | with | on | next to | in front of | behind | near ', text)
    phrases = []
    for chunk in chunks:
        chunk = chunk.strip(' .')
        # remove leading 'a', 'an', 'the', 'some', 'two'...
        chunk = re.sub(r'^(a|an|the|some|two|three|four|several|many)\s+', '', chunk)
        if 2 < len(chunk) < 40:  # skip empty or weird chunks
            phrases.append(chunk)
    return phrases


queries = []
for cap in captions:
    for phrase in extract_object_phrases(cap):
        if phrase not in queries:
            queries.append(phrase)

print(f"  Words the program came up with by itself: {queries}")

# ---------- STEP 3: OWL-ViT finds a box for each word ----------
print("\n--- STEP 3: OWL-ViT searches for each word ---")

inputs = owl_processor(text=[queries], images=image, return_tensors="pt")
with torch.no_grad():
    outputs = owl(**inputs)

# convert the raw output into boxes in normal pixel coordinates
target_size = torch.tensor([image.size[::-1]])  # (height, width)
results = owl_processor.post_process_object_detection(
    outputs, threshold=BOX_CUTOFF, target_sizes=target_size
)[0]

detections = []
for score, label_idx, box in zip(results['scores'], results['labels'], results['boxes']):
    detections.append({
        'name': queries[int(label_idx)],
        'confidence': float(score),
        'box': [round(v) for v in box.tolist()],  # [left, top, right, bottom]
    })

# sort: most confident first
detections.sort(key=lambda d: d['confidence'], reverse=True)

for det in detections:
    print(f"  ✓ {det['name']} ({det['confidence']:.0%} sure) at {det['box']}")
if not detections:
    print("  (no boxes found — try lowering BOX_CUTOFF)")

# ---------- Draw the boxes on a copy of the photo ----------
annotated = image.copy()
draw = ImageDraw.Draw(annotated)
for det in detections:
    left, top, right, bottom = det['box']
    draw.rectangle([left, top, right, bottom], outline='red', width=3)
    draw.text((left + 4, top + 4), f"{det['name']} {det['confidence']:.0%}", fill='red')

result_path = 'result_' + image_path.split('/')[-1].rsplit('.', 1)[0] + '.jpg'
annotated.save(result_path)
print(f"\n💾 Saved photo with boxes drawn on it: {result_path}")

# ---------- STEP 4: If there's a person, describe the action ----------
person_words = ['person', 'man', 'woman', 'boy', 'girl', 'people',
                'someone', 'guy', 'lady', 'chef', 'child']
all_text = ' '.join(captions) + ' ' + ' '.join(d['name'] for d in detections)
person_present = any(word in all_text.lower().split() or word in all_text.lower()
                     for word in person_words)

action_description = None
if person_present:
    action_description = blip_describe(image, starter_text="the person is")

# ---------- FINAL REPORT ----------
print("\n" + "=" * 50)
print("📋 FINAL REPORT")
print("=" * 50)

print("\nObjects found (with locations):")
if detections:
    for det in detections:
        print(f"  • {det['name']} ({det['confidence']:.0%})")
else:
    print("  (none)")

print("\nWhat the scene looks like:")
print(f"  \"{captions[0]}\"")

if action_description:
    print("\nWhat the person is doing:")
    print(f"  \"{action_description}\"")
else:
    print("\nNo person detected, so no action to describe.")
