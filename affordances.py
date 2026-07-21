"""
affordances.py — Phase 1d: what could a robot DO in this scene? (100% free & local)

An "affordance" is what an object lets you do with it: a cup affords
picking up and drinking from, a knife affords cutting, a chair affords
sitting on.

This is the robot's REASONING brain. The pipeline is:

  perceive.py  ->  perception_<photo>.json  ->  affordances.py
  (eyes: what/where)   (structured facts)       (brain: what can I do?)

perceive.py already did the seeing; this script feeds those facts to a
small language model (Qwen 2.5, running entirely on this computer — no
internet account, no API key, no cost) and asks it to reason like a
robot: what can I do with each object, and what would be useful here?

How to run it:
  python affordances.py                          <- uses perception_test_image1.json
  python affordances.py perception_cooking.json  <- any saved scene
"""

import json
import os
import sys

# ---------- Load the scene facts that perceive.py saved ----------
facts_path = sys.argv[1] if len(sys.argv) > 1 else 'perception_test_image1.json'

if not os.path.exists(facts_path):
    print(f"❌ Can't find {facts_path}")
    print("   Run perceive.py on a photo first — it saves the scene facts file.")
    sys.exit(1)

with open(facts_path) as f:
    scene = json.load(f)

print(f"🧠 Reasoning about: {scene['image']}")

# ---------- Turn the scene facts into a prompt for the model ----------
object_lines = "\n".join(
    f"- {obj['name']} (located {obj['where']} of the scene)"
    for obj in scene['objects']
)
relation_lines = "\n".join(f"- {r}" for r in scene['relations'])
action_line = scene['action'] if scene['action'] else "no person detected"

prompt = f"""You are the reasoning brain of a robot with one gripper arm and a camera.
Your vision system just analyzed a scene and reported these facts:

Scene description: "{scene['scene']}"

Objects detected:
{object_lines}

How they are arranged:
{relation_lines}

Human activity: {action_line}

Based only on these facts, answer in three short sections:

1. AFFORDANCES — for each detected object, list 2-3 physical actions the robot
   could realistically do with it (e.g., "pick up", "push", "pour from").

2. RISKS — anything the robot should be careful about in this scene.

3. USEFUL TASK — ONE helpful task the robot could do here, as a short numbered
   list of steps referencing the detected objects and their positions.

Keep the language simple and practical."""

# ---------- Load the local language model ----------
print("(loading the reasoning model — first time downloads ~3GB...)")

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype="auto")

# Chat models expect the conversation wrapped in a special format —
# apply_chat_template does that wrapping for us.
chat = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

inputs = tokenizer(text, return_tensors="pt")

print("🤖 Thinking (this takes a minute or two on a laptop)...\n")
output = model.generate(**inputs, max_new_tokens=600)

# The output contains our prompt + the answer; slice off the prompt part
answer_tokens = output[0][inputs.input_ids.shape[1]:]
answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)

# ---------- Print the robot's reasoning ----------
print("=" * 50)
print("📋 ROBOT REASONING REPORT")
print("=" * 50)
print(answer)
