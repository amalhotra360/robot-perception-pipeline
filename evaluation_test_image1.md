# Evaluation: test_image1.jpeg

Scoring my detection pipeline against the TRUE contents of the image
(this is called "ground truth" evaluation).

## Ground truth (what's actually in the picture)

1. potted plant (cactus)
2. telephone (old type)
3. donut on a plate
4. clock
5. lamp
6. glasses
7. eraser
8. pencil
9. push pin
10. red pen

## Scorecard

| Model / version | Got right | Made up / mislabeled | Missed |
|---|---|---|---|
| YOLOv8s alone | clock, donut, potted plant | remote (= the telephone), cup (= cactus tin), scissors (= the glasses), mouse, carrot | 7 / 10 |
| YOLO + BLIP zoom check | clock, potted plant | scissors (= glasses); wrongly threw out the real donut | 8 / 10 |
| YOLO + CLIP check | clock, donut, potted plant | remote (= telephone), cup (= cactus tin) | 7 / 10 |
| BLIP scene sentence | lamp, phone, donuts, cactus | (nothing!) | 6 / 10 |

## What I learned

1. **The vocabulary limit was the real problem, not accuracy.**
   YOLO only knows 80 object types. 7 of my 10 objects (telephone, lamp,
   glasses, eraser, pencil, push pin, red pen) are NOT on that list, so
   YOLO could never detect them no matter how well it works.

2. **YOLO's "mistakes" were near-misses, not hallucinations.**
   It saw the telephone but called it "remote"/"mouse" (similar gadgets it
   knows). It saw the glasses but called them "scissors" (similar shape).
   It picks the closest word in its limited vocabulary.

3. **BLIP's plain sentence was the most precise output** (4 correct,
   0 made up) but it only mentions the most obvious objects and gives
   no locations.

4. **Nobody found the small objects** (eraser, pencil, push pin, pen).
   Tiny objects are hard for all of these models.

5. **Precision vs recall trade-off is real:** every double-checking layer
   I added removed some fake objects (better precision) but also removed
   some real ones (worse recall).

## Next steps for better accuracy

- **Open-vocabulary detection** (e.g. OWL-ViT): you give it ANY words
  ("pencil", "eraser", "telephone") and it finds boxes for them —
  removes the 80-word limit while keeping locations.
- **Large vision-language model** (e.g. Claude with vision): one big
  model that lists objects, describes actions, and reasons about the
  scene — most accurate option, costs ~a cent per image via API.
