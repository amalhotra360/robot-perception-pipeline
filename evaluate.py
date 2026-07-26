"""
evaluate.py — Phase 1e: how good is the pipeline, in real numbers?

Instead of eyeballing the results, this SCORES them against the truth.

You give it two things:
  1. a "truth" file  -> what's REALLY in the photo (you write this by hand)
  2. a perception_*.json file -> what the program THOUGHT was there

It then counts three things and prints the famous scores:
  - CORRECT  (true positive)  = program found a real object
  - MADE UP  (false positive) = program found something that isn't there
  - MISSED   (false negative) = a real object the program didn't find

  Precision = correct / (everything the program said)   -> "how often it's right"
  Recall    = correct / (everything really there)        -> "how much it caught"
  F1        = one number that balances the two

How to make a truth file:
  Create a text file like truth_test_image1.txt with ONE object per line:
      cactus
      telephone
      donut
      ...
  (blank lines and lines starting with # are ignored)

How to run it:
  python evaluate.py truth_test_image1.txt perception_test_image1.json
"""

import json
import re
import sys

# ---------- Read the two file names from the command ----------
if len(sys.argv) < 3:
    print("Usage: python evaluate.py <truth_file.txt> <perception_file.json>")
    sys.exit(1)

truth_path = sys.argv[1]
perception_path = sys.argv[2]

# ---------- Words that mean the same thing ----------
# so "telephone" in the truth still matches "phone" from the program
SYNONYMS = [
    {"phone", "telephone", "cell phone", "mobile"},
    {"potted plant", "plant", "cactus", "cactus plant", "flower", "pot"},
    {"cup", "mug", "glass", "glass of wine"},
    {"couch", "sofa"},
    {"tv", "television", "monitor"},
    {"person", "man", "woman", "girl", "boy", "arafed woman", "lady", "child"},
    {"bottle", "bottle of wine", "bunch of bottles", "wine"},
    {"table", "counter", "desk", "dining table"},
    {"glasses", "eyeglasses", "spectacles"},
]


def clean(name):
    """lowercase, remove extra spaces and simple plurals -> a tidy word."""
    name = name.lower().strip()
    name = re.sub(r"[^a-z ]", "", name)     # drop punctuation/numbers
    name = name.rstrip("s") if len(name) > 3 else name  # rough de-plural
    return name.strip()


def same_thing(a, b):
    """Do two object names mean the same thing?"""
    a, b = clean(a), clean(b)
    if a == b:
        return True
    # Compare by WHOLE WORDS, not raw letters. (The old version checked if
    # "table" was inside "vegetables" — which matched by accident! Splitting
    # into words first fixes that: {table} is not inside {cutting, vegetables}.)
    wa, wb = set(a.split()), set(b.split())
    if wa and wb and (wa <= wb or wb <= wa):  # all words of one are in the other
        return True
    for group in SYNONYMS:                     # or they're in a synonym group
        cleaned = {clean(w) for w in group}
        if any(a == c or a in c.split() or c in wa for c in cleaned) and \
           any(b == c or b in c.split() or c in wb for c in cleaned):
            return True
    return False


# ---------- Load the truth list (what's really there) ----------
truth = []
with open(truth_path) as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#"):
            truth.append(line)

# ---------- Load what the program detected ----------
with open(perception_path) as f:
    scene = json.load(f)
detected = [obj["name"] for obj in scene["objects"]]

print(f"📸 Photo: {scene['image']}")
print(f"   Truth has {len(truth)} objects | Program found {len(detected)}\n")

# ---------- Match them up (each item can match only once) ----------
truth_left = list(truth)      # real objects not yet matched
correct = []                  # (detected, matched-truth) pairs

for d in detected:
    match = next((t for t in truth_left if same_thing(d, t)), None)
    if match:
        correct.append((d, match))
        truth_left.remove(match)   # used up, can't match again

made_up = [d for d in detected if d not in [c[0] for c in correct]]
missed = truth_left               # truth items nothing matched

# ---------- Print what happened ----------
print("✅ CORRECT (found a real object):")
for d, t in correct:
    print(f"   • {d}" + (f'  (= truth "{t}")' if clean(d) != clean(t) else ""))
print("\n❌ MADE UP (not really there):")
for d in made_up:
    print(f"   • {d}")
print("\n⚠️  MISSED (real, but not found):")
for t in missed:
    print(f"   • {t}")

# ---------- The scores ----------
n_correct = len(correct)
precision = n_correct / len(detected) if detected else 0
recall = n_correct / len(truth) if truth else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

print("\n" + "=" * 45)
print("📊 SCORES")
print("=" * 45)
print(f"Precision (how often it's right): {precision:.0%}")
print(f"Recall    (how much it caught):   {recall:.0%}")
print(f"F1        (balance of the two):   {f1:.0%}")
