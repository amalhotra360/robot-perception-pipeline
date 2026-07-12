"""
perceive.py — fully automatic scene understanding (no word lists from the user!)

The pipeline:
  Step 1: BLIP looks at the photo (and each quarter of it, to catch small
          things) and describes what it sees — no prompting needed.
  Step 2: The program chops BLIP's sentences into search words
          ("a red desk lamp, a phone and a cactus" -> lamp, phone, cactus).
  Step 3: OWL-ViT takes those words and finds a BOX for each one.
          Duplicate finds (same box, two names) get merged.
  Step 4: Using the boxes, the program works out spatial relationships:
          "the knife is above the cutting board", "the lamp is on the desk".
  Step 5: If a person is in the photo, BLIP finishes the sentence
          "the person is..." to describe the ACTION.
  Step 6: A copy of the photo with the boxes drawn on it is saved,
          so you can SEE what was found.

How to run it:
  python perceive.py                <- uses test_image1.jpeg
  python perceive.py cooking.jpg    <- any photo you want
"""

import re
import sys
import torch
from PIL import Image, ImageDraw
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    OwlViTProcessor, OwlViTForObjectDetection,
)

# ---------- Settings you can play with ----------
BOX_CUTOFF = 0.08     # how sure OWL-ViT must be to keep a box (0 to 1)
                      # note: OWL-ViT's scores run low — 0.2 is a strong match
SAME_OBJECT = 0.50    # if two boxes overlap more than this, treat them
                      # as the same object found under two names
WIGGLE = 0.08         # ignore position differences smaller than 8% of the
                      # photo size when deciding left/right/above/below

# ---------- Pick which picture to analyze ----------
image_path = sys.argv[1] if len(sys.argv) > 1 else 'test_image1.jpeg'
print(f"🖼️  Analyzing: {image_path}")

# ---------- Load the two models ----------
print("\n(loading models, this takes a moment...)")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
blip = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
owl_processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
owl = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

image = Image.open(image_path).convert('RGB')


def blip_describe(img, starter_text=None):
    """Ask BLIP to describe a picture (or finish a sentence about it)."""
    if starter_text:
        inputs = blip_processor(img, text=starter_text, return_tensors="pt")
    else:
        inputs = blip_processor(img, return_tensors="pt")
    out = blip.generate(**inputs, max_new_tokens=40)
    return blip_processor.decode(out[0], skip_special_tokens=True)


def caption_to_phrases(caption):
    """Chop a sentence like 'a red desk lamp, a phone and a cactus'
    into search words: ['red desk lamp', 'phone', 'cactus']"""
    text = caption.lower()

    # remove filler openings
    for junk in ['there is ', 'there are ', 'this is ', 'an image of ',
                 'a picture of ', 'a photo of ']:
        text = text.replace(junk, '')

    # turn connecting words into commas, so we can split on commas
    # (longer patterns first, so ' that is ' gets handled before ' is ')
    for connector in [' that is ', ' that are ', ' and ', ' with ',
                      ' next to ', ' on top of ', ' in front of ',
                      ' beside ', ' near ', ' in ', ' on ', ' at ',
                      ' is ', ' are ', ' sitting ', ' standing ',
                      ' playing ', ' holding ', ' talking ']:
        text = text.replace(connector, ' , ')

    # words that aren't objects and shouldn't be searched for
    junk_words = {'someone', 'something', 'it', 'they', 'them', 'that',
                  'one', 'her', 'his', 'top'}

    phrases = []
    for p in text.split(','):
        p = p.strip().rstrip('.')
        # remove articles from the front ("a phone" -> "phone")
        for article in ('a ', 'an ', 'the ', 'some ', 'his ', 'her '):
            if p.startswith(article):
                p = p[len(article):]
        # keep short, thing-like phrases only
        if p and p not in junk_words and len(p.split()) <= 3 and p not in phrases:
            phrases.append(p)
    return phrases


# ---------- Box math helpers ----------
def box_area(box):
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def overlap_amount(box_a, box_b):
    """How much two boxes overlap: 0 = not at all, 1 = exactly the same box."""
    left = max(box_a[0], box_b[0])
    top = max(box_a[1], box_b[1])
    right = min(box_a[2], box_b[2])
    bottom = min(box_a[3], box_b[3])
    if right <= left or bottom <= top:
        return 0.0
    shared = (right - left) * (bottom - top)
    total = box_area(box_a) + box_area(box_b) - shared
    return shared / total


def mostly_inside(box_a, box_b):
    """Is box A mostly sitting inside a much bigger box B?
    (like a lamp on a desk, or a knife on a table)"""
    left = max(box_a[0], box_b[0])
    top = max(box_a[1], box_b[1])
    right = min(box_a[2], box_b[2])
    bottom = min(box_a[3], box_b[3])
    if right <= left or bottom <= top:
        return False
    shared = (right - left) * (bottom - top)
    a_is_inside = shared / box_area(box_a) > 0.85   # 85% of A is within B
    b_is_bigger = box_area(box_b) > 3 * box_area(box_a)
    return a_is_inside and b_is_bigger


def where_in_photo(box):
    """Turn a box into plain words like 'bottom-left' or 'center-middle'."""
    cx = (box[0] + box[2]) / 2 / image.width   # 0 = left edge, 1 = right edge
    cy = (box[1] + box[3]) / 2 / image.height  # 0 = top edge,  1 = bottom edge
    horiz = 'left' if cx < 1/3 else ('right' if cx > 2/3 else 'middle')
    vert = 'top' if cy < 1/3 else ('bottom' if cy > 2/3 else 'center')
    return f"{vert}-{horiz}"


def relationship(obj_a, obj_b):
    """Where is object A compared to object B, in plain words?"""
    ax = (obj_a['box'][0] + obj_a['box'][2]) / 2
    ay = (obj_a['box'][1] + obj_a['box'][3]) / 2
    bx = (obj_b['box'][0] + obj_b['box'][2]) / 2
    by = (obj_b['box'][1] + obj_b['box'][3]) / 2

    wiggle_x = image.width * WIGGLE
    wiggle_y = image.height * WIGGLE

    parts = []
    if ax < bx - wiggle_x:
        parts.append("left of")
    elif ax > bx + wiggle_x:
        parts.append("right of")
    if ay < by - wiggle_y:
        parts.append("above")
    elif ay > by + wiggle_y:
        parts.append("below")

    return " and ".join(parts) if parts else "right next to"


# ---------- STEP 1: BLIP names what it sees ----------
print("\n--- STEP 1: BLIP describes the photo (and its 4 quarters) ---")

captions = [blip_describe(image)]  # the whole photo
print(f"  Whole photo: \"{captions[0]}\"")

# zoom into each quarter to catch smaller objects
w, h = image.size
quarters = [(0, 0, w//2, h//2), (w//2, 0, w, h//2),
            (0, h//2, w//2, h), (w//2, h//2, w, h)]
for i, qbox in enumerate(quarters, 1):
    q_caption = blip_describe(image.crop(qbox))
    captions.append(q_caption)
    print(f"  Quarter {i}:   \"{q_caption}\"")

# ---------- STEP 2: Turn the sentences into search words ----------
search_words = []
for cap in captions:
    for phrase in caption_to_phrases(cap):
        if phrase not in search_words:
            search_words.append(phrase)

print(f"\n--- STEP 2: Search words the program made by itself ---")
print(f"  {search_words}")

# ---------- STEP 3: OWL-ViT finds a box for each search word ----------
print("\n--- STEP 3: OWL-ViT hunts for each word ---")

# OWL-ViT understands words best in the form "a photo of a ..."
queries = [f"a photo of a {word}" for word in search_words]
inputs = owl_processor(text=[queries], images=image, return_tensors="pt")
with torch.no_grad():
    outputs = owl(**inputs)

target_sizes = torch.tensor([[image.height, image.width]])
found = owl_processor.post_process_grounded_object_detection(
    outputs=outputs, target_sizes=target_sizes, threshold=BOX_CUTOFF)[0]

# keep only the BEST box for each word (highest score)
best = {}
for score, label, box in zip(found['scores'], found['labels'], found['boxes']):
    name = search_words[int(label)]
    s = float(score)
    if name not in best or s > best[name]['score']:
        best[name] = {'score': s, 'box': [float(x) for x in box]}

for name, info in best.items():
    print(f"  ✓ {name}  ({info['score']:.0%} match, {where_in_photo(info['box'])})")

missing = [wrd for wrd in search_words if wrd not in best]
if missing:
    print(f"  (no box found for: {', '.join(missing)})")

# Merge duplicates: if two names landed on (almost) the same box, it's one
# object described two ways — keep the name with the higher score.
strongest_first = sorted(best, key=lambda n: -best[n]['score'])
objects = []
for name in strongest_first:
    info = best[name]
    is_duplicate = any(
        overlap_amount(info['box'], kept['box']) > SAME_OBJECT
        for kept in objects
    )
    if is_duplicate:
        print(f"  (merged \"{name}\" — same box as an object already found)")
    else:
        objects.append({'name': name, 'score': info['score'], 'box': info['box']})

# ---------- STEP 4: Spatial relationships ----------
print("\n--- STEP 4: Where things are, relative to each other ---")

relations = []
for i, a in enumerate(objects):
    for b in objects[i + 1:]:
        if mostly_inside(a['box'], b['box']):
            relations.append(f"the {a['name']} is on/inside the {b['name']}")
        elif mostly_inside(b['box'], a['box']):
            relations.append(f"the {b['name']} is on/inside the {a['name']}")
        else:
            relations.append(f"the {a['name']} is {relationship(a, b)} the {b['name']}")

for r in relations:
    print(f"  • {r}")
if not relations:
    print("  (need at least 2 objects for relationships)")

# ---------- STEP 5: Describe the action if there's a person ----------
all_caption_words = set(re.findall(r'[a-z]+', ' '.join(captions).lower()))
person_words = {'person', 'man', 'woman', 'boy', 'girl', 'child', 'kid',
                'baby', 'people', 'chef', 'guy', 'lady'}

action_description = None
if all_caption_words & person_words:  # any person word mentioned?
    action_description = blip_describe(image, starter_text="the person is")

# ---------- STEP 6: Save a copy of the photo with boxes drawn on ----------
drawn = image.copy()
pen = ImageDraw.Draw(drawn)
colors = ['red', 'lime', 'deepskyblue', 'yellow', 'magenta', 'orange', 'cyan']
for i, obj in enumerate(objects):
    color = colors[i % len(colors)]
    left, top, right, bottom = obj['box']
    pen.rectangle([left, top, right, bottom], outline=color, width=4)
    pen.text((left + 5, max(0, top - 15)), obj['name'], fill=color)

out_path = 'detected_' + image_path.split('/')[-1].rsplit('.', 1)[0] + '.jpg'
drawn.save(out_path)

# ---------- FINAL REPORT ----------
print("\n" + "=" * 50)
print("📋 FINAL REPORT")
print("=" * 50)

print("\nObjects found (with where they are):")
for obj in objects:
    print(f"  • {obj['name']}  —  {where_in_photo(obj['box'])} of the photo")

print("\nHow they're arranged:")
for r in relations:
    print(f"  • {r}")

print("\nWhat the scene looks like:")
print(f"  \"{captions[0]}\"")

if action_description:
    print("\nWhat the person is doing:")
    print(f"  \"{action_description}\"")
else:
    print("\nNo person detected, so no action to describe.")

print(f"\n🖼️  Photo with boxes drawn: {out_path}")
