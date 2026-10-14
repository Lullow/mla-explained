# Plan: homework 14, explain MLA to a programmer

Written 2026-10-14, before building. The assignment is the teacher's Discord message
of 2026-10-01. No deadline is stated.

## The assignment in one paragraph

> Förklara MLA i dina egna ord, hur det funkar. Förklara för en student som ej
> pluggat ai men som kan programmering generellt.

Plus a link to the video "Multi-Head Latent Attention Explained Visually" (Visual AI),
given as an example of how MLA works.

Read as: explain Multi-Head Latent Attention, the attention variant from DeepSeek-V2
(arXiv 2405.04434), in the student's own words, for a reader who programs but has
never studied AI. The reader is the constraint: nothing may rest on already knowing
transformers.

## Decisions

**A. The form.** Chosen: A2, a written explanation in README.md plus a small NumPy demo.
- A1, text only. Fastest, but an abstract mechanism explained in prose alone is hard
  to check for understanding, for the grader and for the student alike.
- A2, text plus a demo. Code is the language the reader already speaks. The demo runs
  the mechanism twice, the obvious way and the MLA way, and shows that the answers
  agree while the cache shrinks. That is proof, not description.
- A3, an interactive visualisation. Time goes to the front end rather than the
  mechanism, and "own words" becomes thin.

**B. Who writes what.** The explanation is written by the student. Claude teaches the
mechanism (done 2026-10-14, four zoom levels), builds the demo, and reviews the text
for correctness and clarity. Why: the assignment says "in your own words", and the
point of the course is understanding, which does not transfer by delegation.

How the text was written, from 2026-10-14: Claude explains one topic in a few lines
and asks one question. The student's answer becomes the section's text. Claude flags
only factual errors, and adds the tables, numbers and commands, which are facts rather
than wording. Three writing rounds: attention and the cache, the latent and
absorption, position.

**C. What the demo covers.** Chosen: C1, the content path only.
- C1: the latent cache, the two code paths that give the same output (rebuild keys
  and values, or translate the query), and the cache and compute counts. One
  mechanism, shown completely.
- C2: also the decoupled RoPE key, showing that naive absorption breaks under RoPE
  and the separate key repairs it. Stronger, but more code to explain. Deferred as an
  extension if time allows. RoPE is explained in the text and counted in the cache
  numbers.

**D. Language.** The README is in Swedish. Changed on 2026-10-14 from English: own
words come easiest in one's own language, and the assignment is written in Swedish.
Code, comments, tests and this plan stay in English.

## The reader's path through the README

The order is the student's, chosen on 2026-10-14. It explains attention before the
problem, because the reader does not know what a key or a head is.

0. Title and summary, written last.
1. What the model does: tokens, one at a time, hidden states, W * h.
2. Attention and the KV cache, with the example "Lisa ... Hon".
3. MLA: why the cache is large, one shared latent per token, and absorption on the
   key and the value side.
4. Position: RoPE, why it does not survive absorption, and the decoupled key.
5. What it gains and what it costs: the DeepSeek-V2 tables from the demo.
6. Run the demo, with the toy example's output.
7. Sources.

## The demo, `mla_demo.py`

- The worked example from the teaching session: hidden size 4, one head of size 2,
  latent of size 2, integer weights. Prints both paths' scores and outputs.
- DeepSeek-V2 sizes with random weights: 128 heads of 128, latent of 512, hidden
  5120. Asserts that the two paths agree.
- A table of what MHA, GQA, MQA and MLA store per token: per layer, for all layers,
  and for a 128k context.
- A table of multiply-adds per generated token for the two paths as the cache grows.
  This is why the trick is necessary and not only elegant.
- `tests/`: the same claims as pytest, including the paper's figures 576 and 32 768.

## Facts the text must get right (DeepSeek-V2, arXiv 2405.04434)

| Symbol | Meaning | DeepSeek-V2 |
|---|---|---|
| d | hidden size | 5 120 |
| n_h | heads | 128 |
| d_h | per-head size of a key, query or value | 128 |
| d_c | latent (KV) size | 512 = 4 d_h |
| d_h^R | decoupled RoPE key size | 64 = d_h / 2 |
| MLA cache per token per layer | d_c + d_h^R | 576 |
| MHA cache per token per layer | 2 n_h d_h | 32 768 |

- The latent is not a compression of existing keys and values. The model is trained
  with the constraint from the start.
- W_UK is absorbed into W_Q and W_UV into W_O at inference only. Training uses the
  explicit form.
- The query is also compressed (d_c' = 1536) to save activation memory in training.
  It is never cached.
- The RoPE key is one per token, shared by all heads. The RoPE query is per head.
- The paper's 93.3 % cache reduction is measured against DeepSeek 67B (GQA), not
  against MHA.

## Done when

- `python mla_demo.py` runs clean and its output is saved in `results/`.
- `pytest` passes.
- README.md is written by the student, reviewed once by Claude, and a programmer
  without AI background can follow it to the end.
- The repo is on GitHub and the link is posted in Discord.
