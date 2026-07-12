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
