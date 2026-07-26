# Code Notes — How All My Code Works

Simple explanations of every script in this project.
(This file gets updated whenever code is added or changed.)

---

## Part 1: Python Survival Kit (coming from C++)

Things Python does differently that show up in my code:

| Python | C++ equivalent | Notes |
|---|---|---|
| `import torch` | `#include <...>` | brings in a library |
| `from PIL import Image` | — | imports just ONE thing from a library |
| indentation | `{ }` braces | Python uses spaces to mark blocks. No semicolons. |
| `my_list = []` | `std::vector` | grows automatically, can hold anything |
| `my_dict = {}` | `std::map` | key → value pairs, e.g. `best['clock'] = 0.89` |
| `(a, b, c)` | struct-ish | a "tuple" — a small bundle of values |
| `def my_func(x):` | function definition | no return type needed |
| `f"score: {s:.2f}"` | `printf("%.2f", s)` | f-string: put variables inside `{ }` in the text |

Patterns that appear everywhere in my scripts:

- **`for box in result.boxes:`** — Python loops directly over the items,
  no index variable needed (like C++ `for (auto& box : boxes)`).
- **`sys.argv`** — the words typed after the script name in the terminal.
  `python perceive.py cooking.jpg` → `sys.argv[1]` is `"cooking.jpg"`.
  (Same as `argv` in C++ `main`.)
- **Negative indexes:** `my_list[-1]` = last item. `text.split(":")[-1]` =
  the part after the last colon.
- **Slices:** `objects[i+1:]` = "everything from position i+1 to the end."
- **`any(...)`** — true if at least one thing in a list is true.
- **String helpers:** `.lower()` (lowercase), `.strip()` (trim spaces),
  `.replace(a, b)` (swap text), `.split(",")` (chop into a list),
  `.startswith("a ")` (check the beginning).
- **`{'a', 'b'} & other_set`** — set overlap: which items appear in BOTH
  groups. I use this to check "did any person-word appear in the captions?"

---

## Part 2: The AI Pattern That Repeats in Every Script

Every AI script in this project follows the same 4 beats:

```
1. LOAD the model          (downloads it the first time, then cached)
2. PREPARE the input       (the "processor" turns image/text into numbers)
3. RUN the model           (predict / generate)
4. DECODE the output       (turn the model's numbers back into words/boxes)
```

Key ideas:

- **Processor = translator.** Models only understand numbers (tensors).
  The processor converts a photo or sentence into numbers on the way in,
  and numbers back into words on the way out.
- **`max_new_tokens=40`** — stop the model after 40 tokens (~30 words).
  Bigger = longer answers, slower runs.
- **`with torch.no_grad():`** — tells PyTorch "we're not training, don't
  keep the extra bookkeeping" → faster and uses less memory.
- **Boxes are 4 numbers:** `[left, top, right, bottom]` in pixels.
  ⚠️ y grows DOWNWARD in images: y=0 is the TOP of the photo.
  That's why "above" means a SMALLER y value.

---

## Part 3: File by File

### step1_object_detection.py — my first script
What it does: YOLO looks at a photo and prints every object it finds.

```python
model = YOLO('yolov8s.pt')                  # load the model
results = model.predict('test_image1.jpeg') # run it on my photo
```

- `box.cls` = which of the 80 object types (as a number, e.g. 74)
- `result.names[...]` = the lookup table from number → word ("clock")
- `box.conf` = confidence, 0 to 1
- `int(...)` and `float(...)` convert the model's tensor-numbers into
  normal Python numbers so we can print them

### step1_blip_accurate.py — BLIP describes the photo
What it does: BLIP writes one sentence about the photo, then simple
keyword-checking guesses which objects are present.

The 4-beat pattern in action:
```python
processor = BlipProcessor.from_pretrained(...)   # 1. load
inputs = processor(image, return_tensors="pt")   # 2. prepare (photo -> numbers)
out = model.generate(**inputs, ...)              # 3. run
caption = processor.decode(out[0], ...)          # 4. decode (numbers -> words)
```
- `**inputs` = "unpack this bundle and pass everything in it as arguments"

### step1_llava_perception.py — LLaVA (ChatGPT with eyes)
What it does: sends the photo plus a QUESTION to LLaVA and prints its answer.
Unlike BLIP (which only captions), LLaVA answers whatever you ask.

- `torch_dtype=torch.float16` = store the model's numbers at half
  precision — half the memory, tiny accuracy cost
- `device_map="auto"` = put the model on the GPU if there is one
- `description.split("ASSISTANT:")[-1]` = LLaVA's raw output contains the
  whole conversation; keep only the part after "ASSISTANT:" (its answer)

### detect_objects_and_actions.py — the 3-model team (YOLO + CLIP + BLIP)
What it does: YOLO guesses objects → CLIP fact-checks the unsure ones by
zooming into each box → BLIP describes the scene and the person's action.

The clever parts:
- **Trust levels:** YOLO 80%+ sure → believe it. 40-80% → zoom in and
  make CLIP rank all 80 object names for that crop. YOLO's answer must
  be in CLIP's top 3 to survive.
- **Decoy answers:** CLIP's choices include fakes like "a shadow" and
  "random clutter" — so if a box is really nothing, CLIP can say so
  instead of being forced to pick a real object.
- `image.crop((left, top, right, bottom))` = cut out a rectangle
- `scores.argsort(descending=True)` = order the choices from best to worst

### auto_detect.py — my own version of the BLIP → OWL-ViT idea
What it does: same core idea as perceive.py (BLIP writes the words,
OWL-ViT finds the boxes) but with two different design choices:

- **Creative captions instead of quarters:** asks BLIP to describe the
  photo 3 times with `do_sample=True, top_p=0.9` — a little randomness so
  each description mentions different objects. (Trade-off vs the quarters
  trick: catches variety, but random captions can also invent things.)
- **Regex for word-chopping:** uses `re.sub` and `re.split` (regular
  expressions — text patterns) instead of chains of `.replace()`.
  `re.split(r',| and | with ', text)` splits on ANY of those separators
  in one shot.

### perceive.py — THE MAIN PIPELINE (current best version)
What it does: photo in → objects + boxes + relationships + action out.
No word lists from me: BLIP writes the search words, OWL-ViT finds boxes.

The 6 steps:
1. **BLIP describes the photo AND its 4 quarters** (zoomed views make it
   mention smaller objects it would skip in the full photo)
2. **`caption_to_phrases()` chops sentences into search words.** It works
   by swapping connector words ("and", "with", "next to", "that is"...)
   for commas, splitting on commas, then cleaning each piece (remove "a ",
   "the ", throw away junk like "someone"). Pure string surgery — no AI.
3. **OWL-ViT hunts for each word.** We ask for "a photo of a clock"
   instead of "clock" (matches how it was trained = better scores).
   For each word we keep only its best box. Then duplicates get merged:
   if two boxes overlap >50%, it's one object with two names — keep the
   higher-scoring one. (Overlap measure = shared area / total area,
   called "IoU" — used everywhere in vision AI.)
4. **Box math turns coordinates into plain English:**
   - `where_in_photo()` — divides the photo into a 3x3 grid, names the
     cell ("bottom-left")
   - `relationship()` — compares two box centers. Higher center = "above",
     more-left center = "left of". A "wiggle" zone (8% of photo size)
     stops tiny differences from counting.
   - `mostly_inside()` — if 85%+ of box A sits inside a box B that's 3x
     bigger → "A is on/inside B" (how we get "the pot is ON the table")
5. **Action:** if any person-word (man/woman/girl/...) appeared in the
   captions, ask BLIP to finish the sentence "the person is..."
6. **Draw the boxes** on a copy of the photo with `ImageDraw` and save it
   as `detected_<photo name>.jpg`

Settings at the top of the file (safe to play with):
- `BOX_CUTOFF = 0.08` — min OWL-ViT score to keep a box
- `SAME_OBJECT = 0.50` — overlap needed to merge two boxes into one
- `WIGGLE = 0.08` — dead zone for left/right/above/below decisions

Known weaknesses (found by testing):
- BLIP sometimes imagines things in zoomed quarter views
  ("brushing her teeth" — there was no toothbrush)
- 2D boxes can't tell "inside" from "in front of" (no depth in a photo) —
  it said "the knife is on/inside the man" because he leans over the table

New in the Phase 1d update: perceive.py now also SAVES its results to
`perception_<photo>.json` using `json.dump(report, f)` — turning Python
dicts/lists into a text file another program can read later.

### affordances.py — the robot's reasoning brain (Phase 1d)
What it does: reads the scene facts perceive.py saved, feeds them to a
small language model (Qwen 2.5, running entirely on this computer —
free, no account, no API key), and asks what a robot could DO here.

The new ideas in this file:

- **`json.load(f)`** — the reverse of `json.dump`: reads the saved file
  back into Python dicts and lists.
- **Building the prompt** — f-strings assemble the scene facts into one
  big instruction text. The prompt gives the model a ROLE (robot brain),
  the FACTS (objects, positions, relations, action), and a FORMAT
  (affordances / risks / useful task).
- **A language model in the same 4-beat pattern:**
  ```python
  tokenizer = AutoTokenizer.from_pretrained(...)          # 1. load
  model = AutoModelForCausalLM.from_pretrained(...)       #    load
  text = tokenizer.apply_chat_template(chat, ...)         # 2. prepare
  output = model.generate(**inputs, max_new_tokens=600)   # 3. run
  answer = tokenizer.decode(output[0][...], ...)          # 4. decode
  ```
- **`apply_chat_template`** — chat models were trained on conversations
  wrapped in special markers (who's the user, where the reply starts).
  This function does that wrapping so you don't have to know the markers.
- **Slicing off the prompt** — `output[0][inputs.input_ids.shape[1]:]`
  looks scary but just means: the model's output contains our prompt
  PLUS its answer, so skip the first N tokens (the prompt) and keep
  the rest (the answer).
- **The trade-off** — a 1.5B-knob local model is free and private but
  much less smart than a cloud model like Claude. Same small-vs-big
  trade-off as YOLO vs LLaVA, now on the language side.

### evaluate.py — scoring the pipeline (Phase 1e)
What it does: compares a hand-written "truth" file (what's really in the
photo) against a perception_*.json (what the program found), and prints
precision, recall, and F1.

How to run it:
  python evaluate.py truth_test_image1.txt perception_test_image1.json

The new ideas in this file:

- **Two input files from the command line** — `sys.argv[1]` and
  `sys.argv[2]` are the truth file and the results file you type after
  the script name.
- **Reading a plain text list** — loop over the file's lines, skip blanks
  and lines starting with `#` (comments). One object per line.
- **`clean()`** — tidies a name: lowercase, drop punctuation, remove a
  trailing "s" so "apples" and "apple" match.
- **`same_thing(a, b)`** — the heart of it. Decides if two names mean the
  same object. THREE ways to match:
  1. exactly equal
  2. all the words of one appear in the other (compared as WHOLE WORDS —
     see the bug note below)
  3. they're both in a hand-written SYNONYMS group (so "telephone" and
     "phone" count as the same)
- **The whole-word bug (great lesson):** my first version checked if one
  name was a substring of the other with `a in b`. That matched "table"
  inside "vege**table**s"! Fixed by splitting into word sets first:
  `set(a.split()) <= set(b.split())` — now "table" is not "inside"
  {cutting, vegetables}. Moral: sanity-check your evaluator, not just
  your model.
- **Greedy matching** — each real object can be matched only ONCE, so if
  the program finds the cactus twice, only one counts as correct and the
  other becomes a "made up". `truth_left.remove(match)` does this.
- **The math:**
  ```python
  precision = correct / len(detected)   # of what it said, how much was right
  recall    = correct / len(truth)      # of what's real, how much it caught
  f1 = 2 * p * r / (p + r)              # balance of the two
  ```
