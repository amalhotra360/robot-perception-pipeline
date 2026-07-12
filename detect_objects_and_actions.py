"""
Detect OBJECTS and ACTIONS in a single picture — with smart double-checking.

The team of 3 models (each does what it's best at):
  YOLO  — finds objects and WHERE they are (boxes)
  CLIP  — the fact-checker: looks at each box and scores how well every
          possible object name matches it (no sentences, so no wording problems)
  BLIP  — the describer: writes a sentence about the scene, and about
          what a person is DOING (the action)

The rules:
  * If YOLO is very sure (80%+), we trust it. No check needed.
  * If YOLO is unsure, we zoom into the box and ask CLIP to rank all
    80 object names (plus decoys like "a shadow"). If YOLO's answer is
    in CLIP's top 3, it stays. Otherwise it's thrown out.

How to run it:
  python detect_objects_and_actions.py                  <- uses test_image1.jpeg
  python detect_objects_and_actions.py kitchen.jpg      <- uses your own picture
"""

import sys
import torch
from ultralytics import YOLO
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    CLIPProcessor, CLIPModel,
)
from PIL import Image

# ---------- Settings you can play with ----------
TRUST_CUTOFF = 0.80   # YOLO this sure? -> trust it, skip the double-check
GUESS_CUTOFF = 0.40   # YOLO less sure than this? -> ignore it completely
TOP_K = 3             # YOLO's answer must be in CLIP's top 3 to survive

# ---------- Pick which picture to analyze ----------
image_path = sys.argv[1] if len(sys.argv) > 1 else 'test_image1.jpeg'
print(f"🖼️  Analyzing: {image_path}")

# ---------- Load the three models ----------
print("\n(loading models, this takes a moment...)")
yolo = YOLO('yolov8s.pt')
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
blip = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

image = Image.open(image_path).convert('RGB')

# The 80 object names YOLO knows, plus "decoy" answers.
# If a box is really just a shadow or clutter, CLIP can pick a decoy
# instead of being forced to pick some object.
yolo_labels = list(yolo.names.values())
decoys = ["a shadow", "a blurry background", "an empty table surface",
          "a wall", "random clutter"]
choices = [f"a photo of a {name}" for name in yolo_labels] + decoys
all_names = yolo_labels + decoys


def clip_rank(crop):
    """Ask CLIP: which of the possible names matches this crop best?
    Returns the names sorted from best match to worst."""
    inputs = clip_processor(text=choices, images=crop,
                            return_tensors="pt", padding=True)
    with torch.no_grad():
        scores = clip_model(**inputs).logits_per_image[0]
    order = scores.argsort(descending=True)
    return [all_names[i] for i in order]


def blip_describe(img, starter_text=None):
    """Ask BLIP to describe a picture (or finish a sentence about it)."""
    if starter_text:
        inputs = blip_processor(img, text=starter_text, return_tensors="pt")
    else:
        inputs = blip_processor(img, return_tensors="pt")
    out = blip.generate(**inputs, max_new_tokens=40)
    return blip_processor.decode(out[0], skip_special_tokens=True)


# ---------- STEP 1: YOLO finds candidate objects ----------
print("\n--- STEP 1: YOLO's guesses ---")

results = yolo.predict(image_path, verbose=False)

candidates = []
for result in results:
    for box in result.boxes:
        confidence = float(box.conf)
        name = result.names[int(box.cls)]
        if confidence >= GUESS_CUTOFF:
            candidates.append({
                'name': name,
                'confidence': confidence,
                'box': box.xyxy[0].tolist(),  # [left, top, right, bottom]
            })
            print(f"  ? {name} ({confidence:.0%} sure)")

if not candidates:
    print("  (YOLO found nothing)")

# ---------- STEP 2: Double-check the unsure ones with CLIP ----------
print("\n--- STEP 2: Double-checking with CLIP ---")

confirmed = []
rejected = []

for obj in candidates:
    # Very confident? Trust YOLO, skip the check.
    if obj['confidence'] >= TRUST_CUTOFF:
        confirmed.append(obj['name'])
        print(f"  ✓ {obj['name']} — YOLO is {obj['confidence']:.0%} sure, trusted")
        continue

    # Otherwise: crop the box (grown 15% for context) and let CLIP rank it
    left, top, right, bottom = obj['box']
    pad_x = (right - left) * 0.15
    pad_y = (bottom - top) * 0.15
    crop = image.crop((
        max(0, left - pad_x),
        max(0, top - pad_y),
        min(image.width, right + pad_x),
        min(image.height, bottom + pad_y),
    ))

    ranking = clip_rank(crop)

    if obj['name'] in ranking[:TOP_K]:
        confirmed.append(obj['name'])
        print(f"  ✓ {obj['name']} CONFIRMED — CLIP's top guesses: {ranking[:TOP_K]}")
    else:
        rejected.append((obj['name'], ranking[0]))
        print(f"  ✗ {obj['name']} REJECTED  — CLIP thinks it's: {ranking[:TOP_K]}")

# remove repeats but keep the order
confirmed = list(dict.fromkeys(confirmed))

# ---------- STEP 3: Describe the scene and the action ----------
print("\n--- STEP 3: Describing the scene (BLIP) ---")

scene_description = blip_describe(image)

action_description = None
if 'person' in confirmed:
    action_description = blip_describe(image, starter_text="the person is")

# ---------- FINAL REPORT ----------
print("\n" + "=" * 50)
print("📋 FINAL REPORT")
print("=" * 50)

print("\nObjects:")
if confirmed:
    for obj in confirmed:
        print(f"  • {obj}")
else:
    print("  (none)")

if rejected:
    print("\nThrown out as probably wrong:")
    for name, clip_guess in rejected:
        print(f"  • YOLO said \"{name}\", but CLIP thinks it's \"{clip_guess}\"")

print("\nWhat the scene looks like:")
print(f"  \"{scene_description}\"")

if action_description:
    print("\nWhat the person is doing:")
    print(f"  \"{action_description}\"")
else:
    print("\nNo person detected, so no action to describe.")
