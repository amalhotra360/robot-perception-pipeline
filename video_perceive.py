"""
video_perceive.py — Phase 2a: run perception on a VIDEO, frame by frame.

Big idea: a video is just MANY photos ("frames") played fast. So we reuse
the same object + action detection from perceive.py — we just run it on
several frames spread across the video.

What this script does:
  1. opens a video file
  2. grabs a few frames evenly spaced through it (default 4)
  3. runs object + action detection on each frame
  4. prints what it saw in each frame
  5. saves everything to video_perception_<name>.json

(A later step will COMPARE the frames to see what CHANGED over time.)

How to run it:
  python video_perceive.py my_clip.mp4          <- 4 frames
  python video_perceive.py my_clip.mp4 6         <- 6 frames instead

Note: to keep it fast per frame, this uses BLIP on the whole frame only
(not the 4-quarter trick from perceive.py) and leans on the built-in
COMMON_OBJECTS list to catch small things.
"""

import json
import sys
import cv2
import torch
from PIL import Image, ImageDraw
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    OwlViTProcessor, OwlViTForObjectDetection,
)

# ---------- Settings (same idea as perceive.py) ----------
BOX_CUTOFF = 0.10
SAME_OBJECT = 0.50
COMMON_OBJECTS = [
    "person", "chair", "table", "couch", "lamp", "clock", "book", "laptop",
    "phone", "cup", "mug", "bottle", "bowl", "plate", "fork", "knife",
    "spoon", "cutting board", "pan", "pot", "jar", "glass", "scissors",
    "pen", "pencil", "eraser", "plant", "flower", "vase", "bag", "ball",
    "toy blocks", "remote", "apple", "banana", "orange", "lemon", "carrot",
    "tomato", "cucumber", "pepper", "lettuce", "onion", "bread", "cake",
    "donut", "egg", "pumpkin", "shelf",
]
SCENE_WORDS = {"kitchen", "room", "background", "wall", "floor", "ceiling",
               "scene", "counter", "countertop", "living room", "bedroom",
               "office", "space", "area"}
ACTION_VERBS = {"cutting", "brushing", "playing", "holding", "talking",
                "standing", "sitting", "eating", "drinking", "cooking",
                "washing", "reading", "writing", "looking", "smiling"}
PERSON_WORDS = {"person", "man", "woman", "boy", "girl", "child", "kid",
                "baby", "people", "chef", "guy", "lady"}

# ---------- Read the command arguments ----------
if len(sys.argv) < 2:
    print("Usage: python video_perceive.py <video_file> [number_of_frames]")
    sys.exit(1)
video_path = sys.argv[1]
num_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 4

# ---------- Load the models once (not per frame!) ----------
print("(loading models, one moment...)")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
blip = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
owl_processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
owl = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")


def blip_describe(img, starter_text=None):
    if starter_text:
        inputs = blip_processor(img, text=starter_text, return_tensors="pt")
    else:
        inputs = blip_processor(img, return_tensors="pt")
    out = blip.generate(**inputs, max_new_tokens=40)
    return blip_processor.decode(out[0], skip_special_tokens=True)


def caption_to_phrases(caption):
    """Chop a caption into clean object words (with Phase 1f filters)."""
    text = caption.lower().replace('arafed ', '').replace('arafed', '')
    for junk in ['there is ', 'there are ', 'this is ', 'an image of ',
                 'a picture of ', 'a photo of ']:
        text = text.replace(junk, '')
    for connector in [' that is ', ' that are ', ' and ', ' with ',
                      ' next to ', ' on top of ', ' in front of ',
                      ' beside ', ' near ', ' in ', ' on ', ' at ',
                      ' is ', ' are ', ' playing ', ' holding ', ' talking ']:
        text = text.replace(connector, ' , ')
    junk_words = {'someone', 'something', 'it', 'they', 'them', 'that', 'one',
                  'her', 'his', 'top'}
    phrases = []
    for p in text.split(','):
        p = p.strip().rstrip('.')
        for article in ('a ', 'an ', 'the ', 'some ', 'his ', 'her '):
            if p.startswith(article):
                p = p[len(article):]
        first = p.split()[0] if p else ''
        if (not p or p in junk_words or p in SCENE_WORDS
                or first in ACTION_VERBS or len(p.split()) > 3 or p in phrases):
            continue
        phrases.append(p)
    return phrases


def box_area(b):
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def overlap_amount(a, b):
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    if right <= left or bottom <= top:
        return 0.0
    shared = (right - left) * (bottom - top)
    return shared / (box_area(a) + box_area(b) - shared)


def where_in_photo(box, W, H):
    cx = (box[0] + box[2]) / 2 / W
    cy = (box[1] + box[3]) / 2 / H
    horiz = 'left' if cx < 1/3 else ('right' if cx > 2/3 else 'middle')
    vert = 'top' if cy < 1/3 else ('bottom' if cy > 2/3 else 'center')
    return f"{vert}-{horiz}"


def analyze_frame(image):
    """Run the whole detector on ONE frame -> facts dict."""
    W, H = image.size

    # 1. BLIP describes the frame
    scene = blip_describe(image)

    # 2. search words = BLIP's words + the built-in common objects
    search_words = caption_to_phrases(scene)
    for obj in COMMON_OBJECTS:
        if obj not in search_words:
            search_words.append(obj)

    # 3. OWL-ViT finds a box for each word
    queries = [f"a photo of a {w}" for w in search_words]
    inputs = owl_processor(text=[queries], images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = owl(**inputs)
    target_sizes = torch.tensor([[H, W]])
    found = owl_processor.post_process_grounded_object_detection(
        outputs=outputs, target_sizes=target_sizes, threshold=BOX_CUTOFF)[0]

    best = {}
    for score, label, box in zip(found['scores'], found['labels'], found['boxes']):
        name = search_words[int(label)]
        s = float(score)
        if name not in best or s > best[name]['score']:
            best[name] = {'score': s, 'box': [float(x) for x in box]}

    # merge duplicate boxes (same object, two names)
    objects = []
    for name in sorted(best, key=lambda n: -best[n]['score']):
        b = best[name]['box']
        if not any(overlap_amount(b, o['box']) > SAME_OBJECT for o in objects):
            objects.append({'name': name, 'box': b,
                            'where': where_in_photo(b, W, H)})

    # 4. action if a person is present
    action = None
    if any(w in scene.lower().split() for w in PERSON_WORDS) or \
       any(o['name'] in PERSON_WORDS for o in objects):
        action = blip_describe(image, starter_text="the person is")

    # keep the boxes too, so the caller can draw them on the frame
    return {'scene': scene, 'action': action, 'objects': objects}


# ---------- Open the video and pick which frames to look at ----------
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"❌ Could not open video: {video_path}")
    sys.exit(1)

total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30
duration = total / fps
print(f"🎬 {video_path}: {total} frames, {fps:.0f} fps, {duration:.1f} seconds")
print(f"   Sampling {num_frames} frames spread across the video\n")

# short name of the video, used for the output file names
name = video_path.split('/')[-1].rsplit('.', 1)[0]

# evenly spaced frame numbers (the +0.5 centers them in each chunk)
indices = [int(total * (i + 0.5) / num_frames) for i in range(num_frames)]

report = []
for order, idx in enumerate(indices, 1):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    if not ok:
        continue
    # OpenCV loads colors as BGR; PIL/our models expect RGB -> convert
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    t = idx / fps

    print(f"--- Frame {order}/{num_frames}  (at {t:.1f}s) ---")
    facts = analyze_frame(pil)
    facts['frame'] = order
    facts['time_sec'] = round(t, 1)

    # draw the boxes on this frame and save it so we can SEE the result
    drawn = pil.copy()
    pen = ImageDraw.Draw(drawn)
    colors = ['red', 'lime', 'deepskyblue', 'yellow', 'magenta', 'orange', 'cyan']
    for k, o in enumerate(facts['objects']):
        left, top, right, bottom = o['box']
        pen.rectangle([left, top, right, bottom], outline=colors[k % len(colors)], width=3)
        pen.text((left + 4, max(0, top - 12)), o['name'], fill=colors[k % len(colors)])
    frame_path = f"videoframe_{name}_{order}.jpg"
    drawn.save(frame_path)

    report.append(facts)
    print(f"  scene:   \"{facts['scene']}\"")
    print(f"  objects: {', '.join(o['name'] for o in facts['objects']) or '(none)'}")
    if facts['action']:
        print(f"  action:  \"{facts['action']}\"")
    print(f"  🖼️  saved frame with boxes: {frame_path}")
    print()

cap.release()

# ---------- Save all frames' facts ----------
out_path = f"video_perception_{name}.json"
with open(out_path, 'w') as f:
    json.dump({'video': video_path, 'frames': report}, f, indent=2)

print(f"💾 Saved all frames: {out_path}")
print(f"   (next step: compare frames to see what CHANGED)")
