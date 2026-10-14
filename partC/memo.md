# Part C — Casual replies in 6 Indic languages

**Recommendation: (c) prompt-engineering for the 3-week launch.** Use the A100 in parallel to prepare (a), a LoRA SFT
that distils the winning prompt. Don't build (b). The bottleneck isn't GPU time. It is **30 reviewer-hours covering
only 2 of 6 languages**, and (c) is the only path that spends those hours on evaluating outputs rather than checking
training data.

## Assumptions
- **A1.** "Casual" is defined as a rubric with the reviewer on day 1: informal second person, everyday rather than
  Sanskritised words, common English loanwords allowed, short sentences. Meaning must not change.
- **A2.** The main model is Part B's 4B model on L4s, with a tokenizer like the Part A proxy: about 2.7 tokens per word
  in Hindi and 14.9 in Kannada.
- **A3.** A blind A/B rating of a short reply takes about 1 minute, so roughly 60 per hour, 600 per week, and about
  1,500 usable over 3 weeks.
- **A4.** Tamil, Telugu, Bengali and Marathi get no reviewer. Unless we can borrow native speakers for spot checks,
  they ship as "no regression", not "casual verified".

## Back-of-envelope arithmetic
| | (a) SFT, synthetic pairs | (b) ≤1B rewriter | (c) prompting |
|---|---|---|---|
| Data | 6 langs × 2k pairs × ~1k tok ≈ **12M tokens** | same, for a model weak in these languages | none |
| GPU | 6 × 4.2e9 FLOP/tok ÷ ~125 TFLOP/s ≈ 5k tok/s → **~40 min/epoch**; generating the data ≈ 1–2 h | similar | none |
| Checking training data | 12k pairs ÷ 60/h = 200 h vs 30 h available → **≤15%**, hi/kn only | same | none needed |
| Serving | no change | second decode: ~11 ms/tok (¼ of Part B's 43 ms) × 1,500 tok (100-word Kannada reply) ≈ **+16 s**; 2 GB weights cut the L4 KV pool by 17% | English instruction ≈ 100 tok; one 60-word Kannada example ≈ **900 tok, 22% of the 4096 context** |
| Undo | retrain | remove the model | edit a string |

So (b) fails on latency for the Dravidian languages, (a) is cheap to train but can't be validated in 3 weeks, and (c)
spends the reviewer's hours on what we would actually ship.

## Success metric (launch gate: blind A/B against current outputs, 150 prompts per language for hi and kn)
- **Casualness win rate ≥ 70%**, and
- **meaning or correctness regressions ≤ 3%**, and
- **wrong-language or wrong-script replies ≤ 1%** (automatic check, all 6 languages).

## Kill criterion
At the **end of day 5**, if the best prompt has a **casual win rate below 55%** or **regressions above 5%** on Hindi
or Kannada (100 pairs each), I stop (c). I would then tell product the same day that 6-language launch is at risk,
re-scope to hi and kn, and start (a) in week 2. That (a) would train on real casual dialogue (e.g. subtitles), with the
model writing the formal side, rather than on the model's own outputs, because a model that can't be *prompted* into
casual speech won't generate good casual training data either.

## Day-1 experiment (about 2 reviewer-hours)
1. Write the rubric with the reviewer (30 min).
2. Take 30 Hindi and 30 Kannada real prompts. Generate: baseline; V1 = English style instruction; V2 = V1 plus one
   short native example. Use temperature 0.
3. Log the extra prompt tokens per variant with `partA/fertility_v1.py`.
4. The reviewer blind-rates 120 pairs.

**Same-evening readout:** if V2 beats V1 by fewer than 10 points of win rate, drop the example, because it costs
~900 tokens per Kannada request. Days 2–4 iterate on the best variant; day 5 is the kill-criterion check.
