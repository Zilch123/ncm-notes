# Next-Concept Modeling: Generating Text Coarse-to-Fine Instead of Word-by-Word

*A blog-style summary of four sources: the NCM paper (Stanford, Oct 2026), the Coconut paper (Meta FAIR), the Hybrid Reasoning Agents paper (Stanford / Together AI), and Google's "Looking back at speculative decoding" post.*

---

## The problem: LLMs write one token at a time

An autoregressive (AR) LLM generates text strictly left to right. A 12-token sentence takes 12 full forward passes, and each pass re-reads the model's weights from memory. Google's speculative-decoding post puts it plainly: modern accelerators can do hundreds of operations per byte read, while a transformer needs only about 10. Decoding is limited by memory bandwidth, not compute, so there is spare compute that one-token-at-a-time generation never uses.

The field has attacked this from several directions. NCM is a new one.

## The big idea behind NCM

**Next-Concept Modeling (NCM)** (Kumbong, Khare et al., Stanford / Together AI, arXiv 2610.14162) replaces "predict the next token" with "predict the next *level of detail*."

Instead of writing a sentence word by word, NCM first fixes a rough outline of the whole thing, then refines it. This is similar to sketching a drawing before adding detail. The paper calls the information in the coarse levels a *concept*.

It works in two stages:

### Stage 1: Learn a multiscale tokenizer
A **multiscale residual VQ-VAE** compresses a text sequence into a hierarchy of discrete codes with K levels:

- Level 1 has very few codes and captures the gist.
- Each later level has more codes and encodes only the *residual*, what the earlier levels missed.
- The final level has as many codes as there are tokens (L), so the original text can be reconstructed almost exactly.

During training, **nested dropout** keeps only the first k levels at random. This forces the early levels to carry real information instead of leaving it to the later ones. Reconstruction is near-perfect: 99.99% exact-token accuracy on OpenWebText, 99.86% on LM1B, and 100% on Sudoku boards.

### Stage 2: Generate level by level
A transformer is trained to generate the levels in order. At each level it:

1. Takes the cumulative reconstruction of all earlier levels.
2. Predicts **all codes of the current level in parallel**, in one forward pass.
3. Adds them to the reconstruction and moves to the next level.

When the last level is done, the tokenizer's decoder turns the reconstruction into text. The number of sequential steps is K, and K is much smaller than the sequence length L.

Two design details matter:

- **Early levels are short.** Level 1 has very few positions, so the first steps are cheap. Diffusion models process the full sequence at every step.
- **Conditioning corruption.** During training, some of the conditioning codes are swapped for random ones from the same codebook. This makes the generator robust to its own sampling errors at inference time.

### How it differs from the alternatives

| | Steps | Cost per step | KV cache | Training cost |
|---|---|---|---|---|
| Autoregressive | L (one per token) | Low | Yes | Moderate |
| Masked diffusion (MDLM, DUO) | Many denoising passes | Full sequence every step | Hard (bidirectional) | Very high (~500B tokens for OWT) |
| **NCM** | **K ≪ L** | **Grows with level; early levels are tiny** | n/a | **~4–5× fewer tokens** |

The idea comes from **next-scale prediction** in image generation, adapted to text. Text has no natural "downsampling" the way images do, so the hierarchy has to be learned.

## What the results show

All main experiments use models of about 100M parameters. The numbers below are from the paper.

**Unconditional generation** (TinyStories, LM1B, OpenWebText)
- Matches or beats strong diffusion baselines with about 4–11× fewer function evaluations (NFE), while keeping sample diversity (entropy) close to the real data.
- On LM1B it uses 11× fewer NFE than the best diffusion baseline (DUO) and runs **21× faster** at better generative perplexity.
- On OpenWebText it is competitive and about 10× faster.

**Conditional generation**
- **XSum summarization:** ROUGE-L 28.0 with 8 NFE, against 26.6 for the strongest diffusion baseline (AURORA-LM) at 128 NFE. That is 16× fewer NFE and about **31× faster**.
- **Test-time scaling:** the saved time can be spent generating several candidates and picking one with MBR decoding. This gives ROUGE-L 29.6 on XSum and about 7% better than the strongest baseline at the same time budget.
- **WMT14 De→En translation:** BLEU 22.6 with 11 NFE. That is behind the best diffusion model (E2D2, 24.8) but ahead of most others, and 5.8× fewer NFE than E2D2. It is below the AR baseline (25.2), but MBR decoding closes the gap (25.2).

**Reasoning (Sudoku)**
Coarse-to-fine generation lets later levels fix mistakes in earlier ones. This suits constraint puzzles, where AR models commit to cells they can never revise.

| Method | Easy | Medium | Hard | NFE |
|---|---|---|---|---|
| Autoregressive (greedy) | 14.6 | 5.1 | 1.0 | 128 |
| DUO (diffusion) | 96.3 | 84.7 | 58.4 | 128 |
| FMLM+ (strongest baseline) | 97.7 | 92.8 | 81.3 | 16 |
| **NCM** | **99.5** | **96.5** | **86.1** | **12** |

**Training budget and scaling**
- On LM1B, NCM sees about 13.4B token positions in total against 66B for the baselines (about 4.9× fewer), and still reaches a lower generative perplexity.
- Larger generators (97M → 317M → 725M parameters) improve quality, which is an early sign it scales.

### Figure 3 of the paper is the best intuition builder
For one German→English sentence, the output after step 1 is mush ("However, it is a important to the the the and the."). By step 6 it has the right shape, and by step 11 it matches the reference. The coarse steps fix the idea and the fine steps fill in and correct the words.

## Honest limitations (from the authors)

- **Fixed schedule.** The depth K and the level sizes are fixed, so even a short continuation traverses the full hierarchy. An input-dependent schedule could save compute.
- **Two-stage training.** The tokenizer is trained separately from the generator. End-to-end training might align them better.
- **Small scale.** Main results are around 100M parameters, with scaling shown only up to 725M on LM1B. It is unproven at LLM scale.
- **Artifacts.** Output is generally fluent but has occasional glitches at coarse-level transitions.
- **Baselines.** The comparisons are mainly against diffusion language models. NCM does not match AR translation quality without MBR decoding.

---

## Related reading: three other ways to escape the token bottleneck

These are not NCM variants, but they attack the same inefficiency from different angles.

### Speculative decoding (Google Research)
*"Looking back at speculative decoding" (Leviathan, Kalman, Matias, Dec 2024)*

Keep AR generation but add parallelism through guess-and-check:

1. **Observation 1:** some tokens are easy. In "The square root of 7 is 2.646", the "7" is just copied, while "2.646" needs real computation. A small model can handle the easy ones.
2. **Observation 2:** decoding is memory-bound, so spare compute is available.
3. A small *draft* model proposes several tokens. The big model checks them all in a **single parallel pass**. **Speculative sampling** accepts or rejects each guess probabilistically, which guarantees the output distribution is **identical** to normal decoding.

Google reports ~2–3× speedups (for example an 11B T5-XXL drafted by a 60M T5-small gives ~3×), and the method is used in products such as AI Overviews in Search.

**The contrast with NCM:** speculative decoding is a lossless speedup of an existing model. NCM changes the generation order and the representation, so it is a different model, not a drop-in. Interestingly, the NCM paper notes that diffusion drafters (DFlash) are now being used for speculative decoding, so the two lines of work meet.

### Coconut: Chain of Continuous Thought (Meta FAIR)
*Hao et al., arXiv 2412.06769*

Chain-of-thought forces reasoning through word tokens, but most of those tokens exist for fluency, not reasoning. Coconut takes the model's **last hidden state** (a "continuous thought") and feeds it straight back as the next input embedding, skipping the decode-to-text step.

- A continuous thought can hold *several* candidate next steps at once, so the model behaves like a **breadth-first search** instead of committing to one path.
- On ProsQA (a planning-heavy logic task the authors introduce): **97.0%** accuracy with ~14 tokens, against 77.5% with ~49 tokens for CoT.
- On GSM8k, CoT is still better in accuracy (42.9% vs 34.1% with Coconut, and 16.5% with no CoT), but Coconut uses far fewer tokens.

**Shared theme with NCM:** both argue that language tokens are the wrong unit for the thinking, and that models should work in a more abstract space and decode to words only when needed.

### Hybrid Reasoning Agents (Stanford / Together AI)
*Zou et al., arXiv 2610.18861*

Latent reasoning (Coconut-style) is fast but opaque, and it cannot call tools. HybridAgent lets one model **think in latent space and act in text**:

- A lightweight **HybridLink** adapter maps hidden states back into embedding space so the model can switch between latent and text modes within a single generation.
- Training uses **HybridSFT** (to coordinate the backbone with the adapter) and then **HybridRL** (to learn *when* to think latently and when to speak).
- Latent thoughts can be rendered as human-readable text, so they remain inspectable.
- Across 11 benchmarks at 4B/8B/14B scale, the authors report +10.4% performance, 35.5–54.7% fewer tokens, 1.8–2.6× faster inference and 2.7–3.7× lower cost against text-only and latent-only agents.

---

## Takeaways

1. **The token is not sacred.** Four different papers question one-token-per-step generation: NCM (coarse-to-fine levels), Coconut and HybridAgent (reasoning in hidden states), and speculative decoding (verify many tokens at once).
2. **NCM's pitch is a better trade-off than diffusion LMs.** It offers parallel generation with far fewer steps, cheaper early steps and less training data, without distillation.
3. **Strengths:** speed (10–31× over diffusion baselines), strong results on planning-style tasks (Sudoku), and a built-in route to test-time scaling.
4. **Caveats:** the models are small (~100M parameters), the tokenizer must be near-lossless, and nothing yet shows it can match large AR LLMs. Treat it as a promising research direction rather than a replacement.
5. **Code:** NCM at <https://github.com/HazyResearch/ncm>, Coconut at <https://github.com/facebookresearch/coconut>, HybridAgent at <https://github.com/togethercomputer/HybridAgent>.

## Sources

- Kumbong, Khare, et al. *Next-Concept Modeling: Fast, Parallel Coarse-to-Fine Text Generation.* arXiv:2610.14162v1, Oct 2026.
- Hao, Sukhbaatar, Su, Li, Hu, Weston, Tian. *Training Large Language Models to Reason in a Continuous Latent Space.* arXiv:2412.06769v4.
- Zou, Qi, Chen, Bianchi, Zhu, Lu, He, Zou. *Hybrid Reasoning Agents.* arXiv:2610.18861v1, Oct 2026.
- Leviathan, Kalman, Matias. *Looking back at speculative decoding.* Google Research blog, Dec 6, 2024.
