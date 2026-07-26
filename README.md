# Robot Perception Pipeline

Teaching a computer to look at a single photo and understand it the way a robot would need to:
**what** objects are there, **where** they are, and **what the people in it are doing**.

This is my first AI project — I'm a 2nd-year Computer Science undergrad at UC Riverside,
building this from scratch and documenting everything I learn along the way.

## What it does

Give `perceive.py` any photo, and it outputs something like this (real output from a
photo of a man chopping onions):

```
Objects found (with where they are):
  • cutting board  —  bottom-right of the photo
  • man  —  center-middle of the photo
  • knife  —  bottom-middle of the photo

How they're arranged:
  • the knife is on/inside the cutting board
  • the cutting board is right of and below the man

What the person is doing:
  "the person is cutting onions on a cutting board in the kitchen"
```

It also saves a copy of your photo with labeled boxes drawn around everything it found.

## How it works

No word lists, no fixed categories — the pipeline generates its own vocabulary from
the photo itself:

1. **BLIP** describes the photo in plain sentences (plus each *quarter* of the photo,
   to catch smaller objects)
2. The program **chops those sentences into search words** — "a red desk lamp, a phone
   and a cactus" becomes `lamp, phone, cactus`
3. **OWL-ViT** (an open-vocabulary detector) finds a **bounding box** for each word,
   and overlapping duplicates get merged
4. **Box math** turns coordinates into plain English: *"the pot is on the table"*,
   *"the lamp is right of the clock"*
5. If a person is in the photo, BLIP finishes the sentence *"the person is..."*
   to describe the **action**

The key idea: each model covers the other's weakness. BLIP knows *what* is in a photo
but not *where*; OWL-ViT knows *where* but has to be told *what*. Chained together,
they're a fully automatic detector with no fixed vocabulary.

## Why not just one model?

I tried that first! The journey is the interesting part:

| Attempt | What went wrong |
|---|---|
| YOLOv8 alone | Only knows 80 object types — called my telephone a "remote" and my glasses "scissors" |
| + confidence filters | Threw away real objects along with fake ones |
| + BLIP double-checking | Rejected correct detections when the two models used different words ("mug" vs "cup") |
| + CLIP fact-checking | Better, but still capped by YOLO's 80-word vocabulary |
| **BLIP → OWL-ViT pipeline** | **Current version — open vocabulary, ~2x more correct detections** |

The full story, with ground-truth evaluations and everything I learned, is in
[phase1_model_comparison.md](phase1_model_comparison.md) and
[evaluation_test_image1.md](evaluation_test_image1.md).

## Try it

```bash
git clone https://github.com/amalhotra360/robot-perception-pipeline.git
cd robot-perception-pipeline
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

python perceive.py your_photo.jpg
```

The models (~2.5 GB total) download automatically on first run. No GPU needed —
it runs on a regular laptop, just takes a minute or two per photo.

## Files

| File | What it is |
|---|---|
| `perceive.py` | **The main pipeline** — objects + boxes + relationships + actions (saves scene facts to JSON) |
| `affordances.py` | **The robot's reasoning brain** — a free local LLM (Qwen 2.5) reads the scene facts and plans what a robot could do |
| `evaluate.py` | **Scoring** — compares detections against a hand-written answer key and reports precision / recall / F1 |
| `truth_*.txt` | Ground-truth answer keys (what's really in each photo) |
| `auto_detect.py` | My own experimental take on the BLIP → OWL-ViT idea (creative captions + regex word extraction) |
| `detect_objects_and_actions.py` | Earlier version: YOLO + CLIP fact-checking + BLIP |
| `step1_*.py` | My first experiments with YOLO, BLIP, and LLaVA |
| `phase1_model_comparison.md` | My research journal — every model tried, and what I learned |
| `evaluation_test_image1.md` | Ground-truth evaluation: scoring the pipeline against reality |
| `code_notes.md` | Plain-English explanation of how all the code works |

## Roadmap

- [x] Phase 1a: object detection (YOLO → multi-model pipeline)
- [x] Phase 1b: spatial relationships from bounding boxes
- [x] Phase 1c: action recognition ("the person is cutting onions")
- [x] Phase 1d: affordances — what could a robot *do* with each object? (local LLM reasoning)
- [x] Phase 1e: evaluation — precision/recall/F1 against ground truth (baseline: ~29% F1)
- [ ] Phase 2: video — understanding how scenes change over time
- [ ] Phase 3: simulation (PyBullet) — perception driving a virtual robot arm
