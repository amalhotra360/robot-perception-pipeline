# Phase 1: Object Detection — Models I Tried

My notes on the different AI models I tested for detecting objects in a single image.

## Quick Comparison

| Model | What it is | Speed | Pros | Cons |
|---|---|---|---|---|
| **YOLOv8n** (nano) | A small, fast object detector | Super fast (under 1 second) | • Small download, runs on any laptop<br>• Easiest one to set up<br>• Tells you *where* each object is in the picture (a box around it)<br>• Gives a confidence score (how sure it is) | • Least accurate — it made up objects that weren't there (a "carrot" and a "donut")<br>• Lots of low-confidence guesses<br>• Only knows 80 everyday object types (can't detect anything outside that list) |
| **YOLOv8s** (small) | A slightly bigger, smarter version of YOLOv8n | Fast (a bit slower than nano) | • More accurate than nano<br>• Same easy code — just change one line<br>• Still gives boxes + confidence scores | • Still made things up (a fake "mouse")<br>• Still limited to the same 80 object types<br>• I had to filter out anything below 60% confidence to make it usable |
| **LLaVA** | An AI that looks at a picture and *describes it in words*, like ChatGPT with eyes | Slow (15–30 seconds per image) | • Most accurate of the three — it actually "thinks" about the scene instead of guessing<br>• Not limited to a list — it can describe anything<br>• Says when it's unsure ("this appears to be...")<br>• You can ask it follow-up questions about the image | • Slow<br>• Big model — takes a lot of memory to run<br>• Doesn't tell you *where* objects are (no boxes), so I can't figure out spatial relationships (left of, above, etc.) from it alone<br>• Answers are plain sentences, which are harder for my code to work with |
| **BLIP-2** (looked into it, didn't try it) | Another describe-the-picture AI, like LLaVA but lighter | Medium (5–10 seconds) | • Faster than LLaVA<br>• Easy to set up | • Descriptions are less detailed than LLaVA's<br>• More likely to mention objects it's not sure about<br>• Also no boxes, same problem as LLaVA |

## What I Learned

**There's a trade-off between speed and accuracy:**

- The **YOLO models** are fast and tell you *where* things are, but they guess wrong sometimes (fake carrots, donuts, and a mouse).
- **LLaVA** is accurate and doesn't make stuff up, but it's slow and can't tell you *where* things are.

**Confidence scores matter:** anything below ~0.50 is basically the model guessing. Filtering out low-confidence detections (I used 0.60) removed most of the fake objects.

**Possible next step:** combine them — use YOLO to find *where* objects are, then use a smarter model (like LLaVA or Claude) to double-check *what* they actually are.

---

# The Story So Far (Plain-English Recap)

1. **First try:** used YOLO to detect objects in a photo. It worked, but shouted out wrong things (a "carrot" and a "mouse" that weren't there).
2. **Tried to fix it:** ignored low-confidence guesses, upgraded to a bigger YOLO, then added a "second opinion" — zoom into each object YOLO claims and ask BLIP "what is this?" If they disagree, throw it out.
3. **Added actions:** if a person is detected, ask BLIP to finish the sentence *"the person is..."* — that gives the action (like "cutting onions").
4. **The second opinion had its own problem:** it also threw out some REAL objects, because the two models sometimes use different words for the same thing ("mug" vs "cup"). Swapped BLIP for CLIP as the fact-checker, since CLIP is built for matching pictures to words.
5. **The big reveal:** I wrote down what was ACTUALLY in the photo (ground truth) and scored everything. Turns out 7 of my 10 objects aren't even in YOLO's 80-word vocabulary — it never had a chance. Its "mistakes" were near-misses: it called my telephone a "remote" and my glasses "scissors" because those are the closest words it knows.

**Two big lessons:**
- **You can't fix what you haven't measured.** Tuning before knowing the true answer is just guessing.
- **Every fix has a price.** Filters that remove fake objects also remove real ones (precision vs recall).

---

# What Each Model Actually Does (Their Different Jobs)

These aren't better/worse versions of the same thing — each one answers a
different question about a photo:

| Model | The question it answers | Like a person who... | Output |
|---|---|---|---|
| **YOLO** | "What's here, and *where*?" | ...points at things and names them, fast — but only knows 80 words | Object names + a box around each + a sureness score |
| **BLIP** | "Describe this photo." | ...writes one sentence about what they see | A sentence, like "a red desk lamp, a phone, and a cactus" |
| **CLIP** | "Does this picture match this word?" | ...plays a matching game: hand them a picture and a word list, they rank which word fits best | A ranking of words (no boxes, no sentences) |
| **LLaVA** | "Look at this and answer my question." | ...you can have a conversation with about the photo | Full answers, like ChatGPT with eyes |

Key differences to remember:

- **YOLO is the only one that gives locations (boxes)** — needed for spatial
  relationships like "the cup is left of the laptop."
- **BLIP and CLIP are opposites in direction.** BLIP goes picture → words
  (writes a sentence from scratch). CLIP goes words → picture (you supply
  the words, it scores how well each one matches). That's why CLIP is the
  better fact-checker: no wording mismatch problems.
- **LLaVA is a different species.** The other three are small single-trick
  models; LLaVA is a full language model with vision bolted on, so it can
  *reason* — but it's huge and slow. Claude-with-vision is the same idea,
  bigger and smarter, running in the cloud.
- **Size explains speed.** Roughly: YOLOv8n ~3M parameters, YOLOv8s ~11M,
  CLIP ~150M, BLIP ~470M, LLaVA ~7 BILLION. Each jump buys more
  understanding and costs more time.

That's why the final pipeline uses a team: YOLO for **where**, CLIP for
**double-checking what**, BLIP for **describing the scene and the action**.

---

# What Are Tokens?

Tokens are the little chunks AI models break text into — usually a word,
part of a word, or punctuation. Models read and write ONE token at a time.

- "The person is cutting onions" → `The` | ` person` | ` is` | ` cutting` | ` on` | `ions`
- Rare words get split into pieces ("onions" → "on" + "ions").
- Rule of thumb: 1 token ≈ 3/4 of a word.

Why it works this way: models only understand numbers, so every token has
an ID number. The model's whole job is: look at the tokens so far, predict
the most likely NEXT token, repeat — like texting with super-aggressive
autocomplete.

Where tokens show up in MY code:

```python
out = blip.generate(**inputs, max_new_tokens=40)
```

This says "stop after 40 tokens" (~30 words) — it's why BLIP's descriptions
are short. Raising it gives longer descriptions but slower runs.

Where else tokens matter:

- **Pricing:** APIs like Claude charge per token (in and out). An image +
  short answer is a few thousand tokens ≈ about a penny.
- **Limits:** every model has a max number of tokens it can handle at once
  (the "context window") — like its working memory.
- **Images become tokens too:** vision models like LLaVA and Claude slice a
  photo into a grid of little patches, and each patch becomes a token. To
  the model, a photo is a "paragraph" of image-tokens — that's the trick
  that lets a language model "see."

---

# The Automatic Pipeline (perceive.py)

The problem: OWL-ViT can find ANY object, but you have to tell it what
words to look for — and typing words myself defeats the purpose.

**The trick: let one model write the words for the other.**

1. BLIP describes the photo in a sentence (no prompting needed)
2. The program chops that sentence into search words
   ("a red desk lamp, a phone and a cactus" → lamp, phone, cactus)
3. OWL-ViT finds a box for each word

So the program invents its own vocabulary from the photo itself.
Photo in → objects with boxes out. No word lists from me, no 80-word limit.

Two extra tricks that made it work much better:

- **The quarters trick:** BLIP only mentions the 4-5 most obvious things,
  so we also show it each QUARTER of the photo separately. Zoomed-in
  views make it mention smaller objects.
- **Ask nicely:** OWL-ViT was trained on phrases like "a photo of a clock",
  so searching for "a photo of a clock" works much better than just "clock".

What I learned from the first (bad) run:

- **Low scores don't always mean wrong.** OWL-ViT's match scores just run
  low — 20% is a STRONG match for it. My cutoff of 15% was throwing away
  good detections. Every model has its own "score personality."
- **Garbage in, garbage out.** My sentence-chopper was producing junk
  search words like "red lamp that is" — and OWL-ViT can't find a good
  box for a junk phrase. Cleaning the words fixed half the problem.
- **Partial views make models imagine things.** When BLIP saw just one
  quarter of the legos photo (a girl's face with her hands up), it guessed
  "brushing her teeth with a toothbrush." No toothbrush exists. Models
  fill gaps with the most familiar story.

Results: went from 1 box per photo to 4-8 correct boxes per photo,
including objects YOLO could never find (telephone, lamp, cutting board).

---

# Spatial Relationships (Box Math)

Once every object has a box, figuring out WHERE things are relative to
each other is just simple math — no AI needed:

- **Every box has a center point** (middle of the rectangle).
- **Compare centers:** if the knife's center is higher up than the cutting
  board's center → "the knife is above the cutting board."
- **Wiggle room:** if two centers are within 8% of the photo size of each
  other, we say "right next to" instead of calling a tiny difference
  "left of." Without this, everything is left/right/above/below of
  everything by a few pixels.

Two extra rules that made the output much nicer:

- **Merging duplicates:** the same object often gets found under two names
  ("plate of donuts" AND "plate of food"). Fix: if two boxes overlap more
  than 50%, they're the same object — keep the higher-scoring name.
  (The overlap measure — shared area divided by total area — is called
  IoU, "Intersection over Union," and it's used everywhere in vision AI.)
- **The on/inside rule:** if one box sits almost entirely inside a much
  bigger box, say "the pot is ON the table" instead of the useless
  "the pot is below-left of the table."

Real output from my photos:
- "the knife is on/inside the cutting board" ✓
- "the pot is on/inside the table" ✓
- "the little girl is above the toy set" ✓

One funny failure that taught me something big:
**"the knife is on/inside the man."** The man is leaning over the table,
so the knife's box IS inside his box — in 2D. The photo has no depth, so
box math can't tell "inside" from "in front of." Understanding 3D space
from a flat photo needs more than rectangles — that's a preview of why
robots use depth cameras.

---

# Affordances (Phase 1d) — The Robot's Brain

An **affordance** is what an object lets you DO with it: a cup affords
picking up and drinking from, a knife affords cutting, a chair affords
sitting on. The word comes from psychology — we see objects in terms of
what we can do with them, and a robot needs to as well.

**The architecture — eyes vs brain:**

```
perceive.py  ->  perception_<photo>.json  ->  affordances.py
(the EYES:        (structured facts          (the BRAIN: what
what & where)      saved to a file)           can I do here?)
```

This separation is real robotics design: perception turns messy pixels
into clean facts, then reasoning works with the facts — it never has to
look at pixels. The JSON file in between means the brain can think about
a scene without re-running all the slow vision models.

**Why a language model for the brain?** Affordances need world
knowledge: knowing that knives cut, that things near a table edge can
fall, that you shouldn't grab a knife by the blade. That knowledge isn't
in the photo — it's in language. So the reasoning step feeds the scene
facts to a language model and asks: "you're a robot with one gripper
arm — what could you do here, what's risky, what would be useful?"

**Cloud API vs local model — a real engineering decision:**

| Option | Cost | Quality | Privacy |
|---|---|---|---|
| Big cloud model (e.g. Claude, via an API) | pennies per request, needs an account + card | smartest | your data goes to a server |
| Small local model (Qwen 2.5, 1.5B) | free forever | good enough for simple reasoning | everything stays on my laptop |

I chose the **free local model** (Qwen 2.5 with 1.5 billion knobs, run
through the same `transformers` library as BLIP). It's the same
trade-off as YOLO vs LLaVA all over again: small-and-free vs
big-and-smart. Knowing when "good enough" is actually good enough is
an engineering skill in itself.

**What I learned about prompting a language model:**
- The pattern is simple: build a text prompt -> run the model -> read
  the reply. The craft is in writing a good prompt.
- Good prompts give a ROLE ("you are the reasoning brain of a robot
  with one gripper arm"), the FACTS (objects, positions, relations),
  and a clear FORMAT for the answer (numbered sections).
- Chat models need their input wrapped in a special conversation format —
  the tokenizer's "chat template" does that wrapping for you.

**The funniest failure of the whole project (so far):** for the photo
of the man cutting onions, the model listed affordances for the MAN:
"pick up, push, carry to storage area." And its task plan began with
"Secure the Man: ensure the man is standing firmly against the wall."

Why this matters: the small model followed my FORMAT perfectly ("2-3
actions for each detected object") but doesn't have the common sense to
realize a person isn't a thing you carry to storage. Instructions were
followed; judgment was missing. Bigger models have more judgment —
that's a big part of what those extra billions of knobs buy. Real robot
systems add safety rules on top of the model for exactly this reason:
never trust the model alone with decisions about people.

---

# Evaluation (Phase 1e) — Real Numbers At Last

Instead of eyeballing results, I scored the pipeline against the truth.
The recipe (this is how EVERY AI system gets measured):

1. **Ground truth** — I write down what's REALLY in each photo, by hand.
2. **Detection** — the program writes down what it thinks is there.
3. **Compare and count** three things:
   - CORRECT (found a real object)
   - MADE UP (found something that isn't there — "false positive")
   - MISSED (a real object it didn't find — "false negative")

**The three scores** (every AI paper reports these):
- **Precision** = correct / everything it said = "how often it's right"
- **Recall** = correct / everything really there = "how much it caught"
- **F1** = one number balancing the two

**My pipeline's report card (baseline):**

| Photo | Precision | Recall | F1 |
|---|---|---|---|
| test_image1 | 50% | 30% | 37% |
| cooking | 56% | 25% | 34% |
| legos | 33% | 11% | 17% |
| **Average** | **46%** | **22%** | **29%** |

This is my BASELINE — the number to beat when I improve things later.
A ~29% F1 for a first hand-built pipeline is an honest starting point.

**What the numbers taught me:**

- **Recall (22%) is the weak spot.** The pipeline misses far more than it
  makes up. Small objects (eraser, pencil, cucumber, knife) get missed
  because BLIP never mentions them, so OWL-ViT never looks for them.
- **The cluttered legos photo scored worst (17%).** More stuff = more to
  miss. Busy scenes are just harder.
- **Evaluation itself can have bugs!** My first scorer matched
  "cutting vegetables" to "table" — because "table" is literally inside
  "vege-TABLE-s". Lesson: it was comparing letters, not whole words.
  Fixed it to compare word-by-word. ALWAYS sanity-check your evaluator.
- **Ground truth is a judgment call.** I listed "smile", "braid", and
  "girl with two pigtails" as truth — but those aren't objects a detector
  can box, so they counted as "missed" and dragged recall down. Real
  datasets define "what counts as an object" BEFORE scoring.

**Ideas to raise the score later** (all measurable now that I have a baseline):
- Give OWL-ViT a built-in list of common objects so it finds things BLIP
  forgets to mention (would help recall a lot).
- Filter out non-object words like "kitchen" and BLIP's "arafed" glitch
  (would help precision).
- Lower/raise the box-confidence cutoff and re-measure the trade-off.

**PHASE 1 IS COMPLETE:** the pipeline now sees objects, locates them,
describes actions, reasons about affordances, AND measures its own
accuracy. Next up: Phase 2 (video — understanding change over time).

---

# Improving the Baseline (Phase 1f) — Measure, Change, Measure Again

Now that I had a baseline (29% F1), I could actually try to BEAT it and
prove whether my changes helped. I made two changes to perceive.py:

1. **Added a built-in list of ~50 everyday objects** (knife, pencil, cup,
   cucumber...) that OWL-ViT ALWAYS searches for, on top of BLIP's words.
   Goal: catch small objects BLIP forgets to mention -> raise RECALL.
2. **Added filters** for scene-words ("kitchen", "counter" — you can't
   pick up a kitchen), action-phrases ("cutting vegetables"), and BLIP's
   "arafed" glitch. Goal: fewer wrong guesses -> raise PRECISION.

**The results (before -> after):**

| Photo | F1 before | F1 after |
|---|---|---|
| test_image1 | 37% | 50% |
| cooking | 34% | 58% |
| legos | 17% | 18% |
| **Average** | **29%** | **42%** |

Average **recall nearly doubled: 22% -> 44%.** F1 jumped 29% -> 42%.
That's a real, measured 13-point improvement.

**What I learned:**

- **The built-in list was the big win.** It found objects the old version
  missed entirely: knife, cucumber, lettuce, pepper, pumpkin (cooking),
  and eraser + pen (test_image1). If you never look for something, you
  can never find it.
- **The precision/recall trade-off showed up AGAIN** (3rd time now!).
  On the cooking photo, casting a wider net also caught junk (tomato,
  fork, pan that weren't there), so precision dipped 56% -> 50%. But
  recall rose so much (25% -> 70%) that F1 still climbed.
- **Some images are just hard.** The legos photo barely improved (17% ->
  18%). Even with "toy blocks" and "chair" in the search list, OWL-ViT
  couldn't confidently find them — blurry white background, scattered
  ambiguous blocks. No prompt trick fixes a model's eyesight. (Its truth
  list also had non-objects like "smile" that can never be detected.)
- **This is THE research loop:** baseline -> change one thing -> measure ->
  keep what helped. I can now say exactly how much my change improved
  things, instead of guessing.

**Ideas to push it further:** raise OWL-ViT's confidence cutoff to trade
some recall back for precision (and measure it!); shrink the built-in
list to the objects that actually helped; or upgrade to a bigger OWL-ViT.

---

# PHASE 2 — Video (Understanding Change Over Time)

The big idea: **a video is just many photos ("frames") played fast.** So
I already know how to understand each frame — my whole Phase 1 pipeline.
Video adds ONE new superpower: comparing frames to see what CHANGED.

**Step 2a (video_perceive.py)** — grab a few frames spread across a video
and run the detector on each. New tool: OpenCV (`cv2`) to read videos.
Big gotcha I learned: OpenCV loads colors as BGR (blue first), but our
models expect RGB (red first) — you must convert, or everything looks
blue and detection fails.

**Step 2b (video_changes.py)** — compare each frame to the one before it:
- object in the new frame but not the old one = APPEARED
- object in the old frame but not the new one = DISAPPEARED
- plus: did the action change?

**The neat realization:** once perception has turned messy pixels into
clean word-lists, understanding "change" is just SET MATH — no AI needed.
`after - before` = what appeared. This is exactly WHY the pipeline saves
structured facts (JSON): reasoning becomes easy when the data is clean.
Perception is the hard part; reasoning on clean facts is simple.

**Tested on a stand-in video** (my 3 photos stitched together): it
correctly saw the desk objects vanish and the kitchen appear, and the
action switch from "cutting vegetables" to "playing with blocks". Because
the 3 scenes are unrelated, everything "came and went" — in a real
continuous clip, most objects stay put and only a few change, and THOSE
are the story ("the onion was whole, now it's chopped").

**Step 2c (video_story.py)** — feed the frame-by-frame facts to the local
language model (Qwen, the same brain as affordances.py) and ask it to
narrate the video in plain English, focusing on what CHANGED. This closes
the video loop: pixels -> per-frame facts -> changes -> a written story.

**The whole Phase 2 pipeline:**
  video -> video_perceive.py (facts per frame)
        -> video_changes.py  (what changed)
        -> video_story.py    (a plain-English story)

**Next:** run it all on a REAL recorded clip (something actually changing,
like chopping an onion) for the genuine payoff.
