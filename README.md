# Undercover Agents vs. Hidden Collusion

Apart Research AI Collusion Research Sprint (23–25 Oct 2026), Track 3.2:
*Can we disrupt collusion through introducing trusted agents?*

**Question:** Can an undercover agent expose a model's willingness to collude, and do the
model's internal activations reveal its true intention better than its words?

All code in this repo was written during the sprint (from 23 Oct 2026). Nothing is preexisting.

## Setup

1. `pip install modal`, then `modal setup` (one-time login; needs a payment method on the
   Modal account to unlock GPUs, sprint credits are spent first)
2. Run any script with `modal run <file>.py`

Models run on Modal GPUs (A10G) with Hugging Face `transformers`, so we can read activations.
Model: `Qwen/Qwen2.5-7B-Instruct`.

## Files

| File | What it does |
|---|---|
| `01_single_offer.py` | Pilot: one explicit offer from Mike; prints Lily's reply and per-layer activation norms |
| `02_activation_similarity.py` | Three offer types; layer-by-layer cosine similarity of Lily's activations |
| `03_rigged_decision.py` | Six offer conditions + a rigged challenge decision, Mike vs. Luke as bluffer |
| `results/` | Saved outputs from runs |
| `TASKS.md` | Open work items; put your name next to one before starting |

## Credits

Liar's Bar rules and the "Secret Alliance Invitation" prompt are adapted from
[poseidon-research/collusion_hackathon](https://github.com/poseidon-research/collusion_hackathon)
(`llm-agent-collusion-2/`, originally [AgentSafety2025/LLM-Agent-collusion](https://github.com/AgentSafety2025/LLM-Agent-collusion);
Zeng et al. 2026, arXiv:2605.27593).
