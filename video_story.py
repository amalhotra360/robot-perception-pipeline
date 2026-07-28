"""
video_story.py — Phase 2c: have the language model NARRATE the video.

Steps 2a and 2b turned the video into clean facts (what's in each frame,
and what changed). This step feeds those facts to a small local language
model (Qwen 2.5 — free, runs on this computer) and asks it to write a
short plain-English story of what happened over time.

  video_perceive.py -> facts per frame -> video_story.py -> "the story"

How to run it:
  python video_story.py video_perception_test_video.json
"""

import json
import sys

# ---------- Load the per-frame facts ----------
path = sys.argv[1] if len(sys.argv) > 1 else 'video_perception_test_video.json'
with open(path) as f:
    data = json.load(f)

frames = data['frames']
print(f"🎬 Narrating: {data['video']}  ({len(frames)} frames)\n")


def names(frame):
    return {o['name'].lower() for o in frame['objects']}


# ---------- Turn the frames into a written timeline for the model ----------
lines = []
for i, fr in enumerate(frames):
    obj_list = ", ".join(o['name'] for o in fr['objects']) or "nothing clear"
    line = f"At {fr['time_sec']}s — objects: {obj_list}."
    if fr.get('action'):
        line += f" Action: {fr['action']}."
    if i > 0:  # note what changed since the previous frame
        appeared = names(fr) - names(frames[i - 1])
        gone = names(frames[i - 1]) - names(fr)
        if appeared:
            line += f" (newly appeared: {', '.join(sorted(appeared))})"
        if gone:
            line += f" (no longer seen: {', '.join(sorted(gone))})"
    lines.append(line)

timeline = "\n".join(lines)

prompt = f"""A camera watched a short video. Here is what a vision system
detected in each frame, in time order:

{timeline}

In 3-4 simple sentences, tell the story of what happened over time. Focus
on what CHANGED and what the person (if any) was doing. Keep it plain and
practical — do not just list objects."""

# ---------- Load the local language model and ask it ----------
print("(loading the language model — first time downloads ~3GB...)")
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype="auto")

chat = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt")

print("🤖 Writing the story (a minute or two on a laptop)...\n")
output = model.generate(**inputs, max_new_tokens=300)
answer = tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

print("=" * 50)
print("📖 THE STORY OF THE VIDEO")
print("=" * 50)
print(answer)
