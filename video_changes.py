"""
video_changes.py — Phase 2b: what CHANGED between the frames?

This is the heart of video understanding. A single photo is frozen; a
sequence of frames lets us notice CHANGE:
  - an object APPEARED (wasn't there, now it is)
  - an object DISAPPEARED (was there, now it's gone)
  - the ACTION changed (person went from "cutting" to "washing")

It reads the video_perception_*.json that video_perceive.py saved, walks
through the frames in order, and prints a timeline of changes.

How to run it:
  python video_changes.py video_perception_test_video.json
"""

import json
import sys

# ---------- Load the per-frame facts ----------
path = sys.argv[1] if len(sys.argv) > 1 else 'video_perception_test_video.json'
with open(path) as f:
    data = json.load(f)

frames = data['frames']
print(f"🎬 Video: {data['video']}  ({len(frames)} frames analyzed)\n")


def names(frame):
    """The set of object names in one frame (lowercased)."""
    return {o['name'].lower() for o in frame['objects']}


# ---------- Walk through the frames in order ----------
print("=" * 50)
print("📽️  TIMELINE OF CHANGES")
print("=" * 50)

# First frame: just say what's there to start
first = frames[0]
print(f"\n▶ START (frame 1, {first['time_sec']}s)")
print(f"   objects: {', '.join(sorted(names(first))) or '(none)'}")
if first.get('action'):
    print(f"   action:  {first['action']}")

# Every next frame: compare to the one before it
for i in range(1, len(frames)):
    prev, curr = frames[i - 1], frames[i]
    before, after = names(prev), names(curr)

    appeared = after - before       # in curr but not prev
    disappeared = before - after    # in prev but not curr

    print(f"\n▶ frame {curr['frame']} ({curr['time_sec']}s)")
    if appeared:
        print(f"   ➕ appeared:    {', '.join(sorted(appeared))}")
    if disappeared:
        print(f"   ➖ disappeared: {', '.join(sorted(disappeared))}")
    if not appeared and not disappeared:
        print("   (no object changes)")

    # did the action change?
    if curr.get('action') and curr['action'] != prev.get('action'):
        print(f"   ~ action now: {curr['action']}")

# ---------- A simple overall summary ----------
print("\n" + "=" * 50)
print("📋 SUMMARY")
print("=" * 50)

# objects seen in EVERY frame = the stable, always-there things
stable = set.intersection(*[names(f) for f in frames]) if frames else set()
# objects seen in only SOME frames = things that came and went
all_objects = set.union(*[names(f) for f in frames]) if frames else set()
changing = all_objects - stable

print(f"Always present: {', '.join(sorted(stable)) or '(none)'}")
print(f"Came and went:  {', '.join(sorted(changing)) or '(none)'}")
