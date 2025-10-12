# RL Grokking Recipe: How RL Unlocks and Transfers New Algorithms in LLMs

[![arXiv](https://img.shields.io/badge/arXiv-2509.21016-b31b1b.svg)](http://arxiv.org/abs/2509.21016)
[![Blog](https://img.shields.io/badge/Blog-Berkeley%20RDI-blue)](https://rdi.berkeley.edu/blog/rl-grokking-recipe)
[![HuggingFace](https://img.shields.io/badge/🤗-Datasets-yellow)](https://huggingface.co/)

**Yiyou Sun¹, Yuhan Cao, Pohao Huang¹, Haoyue Bai², Hannaneh Hajishirzi³⁴, Nouha Dziri⁴♠, Dawn Song¹♠**  
¹ *University of California, Berkeley* · ² *University of Wisconsin–Madison* · ³ *University of Washington* · ⁴ *AI2* (♠ *indicates equal advising*)


## 🎯 TL;DR

1. **DELTA Benchmark Suite**: A controlled collection of synthetic programming families with fully out-of-distribution splits (Manufactoria) and verifiable rewards. DELTA lets us ask two questions:
   - **Learnability**: Can RL solve families where the base model has pass@K=0?
   - **Transferability**: Do the learned procedures generalize?

2. **Grokking Phase Transition**: On several pass@128=0 families, RL exhibits a grokking-like phase transition—after a long near-zero-reward plateau, accuracy snaps to ~100%. That is **discovery**, not mere sharpening.

3. **Two-Phase Reward Schedule**: A staged reward schedule is key to escaping the "all-zero" region:
   - **Phase 1**: Dense per-test rewards to break out of zero region
   - **Phase 2**: Binary full-pass to consolidate exact solutions
   - Binary-only gets stuck; dense-only hovers at "almost right." The schedule yields the grokking jump.

---

## 📦 What's in This Repository?

This repository contains the **DELTA** benchmark suite—a collection of five distinct problem families designed to rigorously test LLM reasoning and RL learnability in truly out-of-distribution settings:

### 🎮 1. Manufactoria
**A pure OOD learnability testbed** based on a classic 2010 Flash game.

- **What**: Program "robot factories" using a minimal DSL with just two primitives: `PULLER` (read) and `PAINTER` (write)
- **Why OOD**: Brand-new textual DSL never seen on the internet; fresh puzzles requiring finite-state, tape-shuffling strategies
- **Difficulty**: 10+ problem families from basic pattern matching to computational tasks where GPT-5 achieves 0% success
- **Location**: [`manufactoria/`](manufactoria/)
- **Datasets**: [HuggingFace @manufactoria](https://huggingface.co/manufactoria)

📖 [**Detailed README**](manufactoria/README.md)

---

### ⚽ 2. BouncingSim
**Physics simulation for testing compositional and transformative generalization.**

- **What**: Synthesize programs that simulate 2D elastic collisions in polygonal containers with precise trajectories
- **Families**: 6 physics scenarios (rotating objects, rotating boxes, moving boxes, gravity, multiple balls/boxes)
- **Generalization Axes**:
  - **Exploratory** 🧭: Harder scenes (more vertices, higher bounciness)
  - **Compositional** 🧩: Recombine primitives (multi-ball + moving boxes)
  - **Transformative** 🔄: Qualitatively different dynamics (periodic trajectories)
- **Location**: [`bouncingsim/`](bouncingsim/)
- **Datasets**: [HuggingFace @bouncingsim](https://huggingface.co/bouncingsim)

📖 [**Detailed README**](bouncingsim/README.md)

---

### 🗄️ 3. Others (SQL/CompetitionCode/Lean)
**Other problem scopes for learnability test** 

- **Comment**: These problems represent family domains where LLMs have already undergone substantial training. They are included here for readers' interest. Typically, a very small LLM (less than 0.5B parameters) is employed for learnability testing in order to explore scenarios where pass@k=0 is applicable.
- **Location**: [`sql`](sql/) / [`competitioncode`](competitioncode/) / [`lean`](lean/)


## 🎓 The Two-Phase Training Methodology

**Training Code:** Complete RLVR infrastructure available at [open-instruct/merge-code-utils](https://github.com/sunyiyou/open-instruct/tree/merge-code-utils)

We acknowledge that some of the training scripts are tailored for AI2's cluster. However, we provide training scripts for the readers' reference. Readers are encouraged to adapt the training parameters to fit other RLVR frameworks. 

We kept the training parameters unchanged across other training settings in the paper. The only variations were in the datasets used, reference models employed, and the scoring modes applied (either per-test accuracy or full pass rate). 

### Phase 1: Dense Per-Test Rewards

Break out of the all-zero reward region by providing partial credit:

```bash
cd train
# Edit phase1.sh to configure your experiment
bash phase1.sh
```

**Key Settings:**
- `SCORE_MODE=pass_rate` (per-test accuracy)
- Reward = fraction of unit tests passed (0.0 to 1.0)
- Enables gradient flow even when no solution is perfect

### Phase 2: Binary Full-Pass Rewards

From the Phase 1 checkpoint, switch to strict correctness:

```bash
# Edit phase2.sh to point to Phase 1 checkpoint
bash phase2.sh
```

**Key Settings:**
- `SCORE_MODE=full_pass` (binary full-pass rate)
- Reward = 1.0 only if all tests pass, else 0.0
- Triggers grokking phase transition to exact solutions


---

## 📚 Citation

If you find this work useful, please cite our paper:

```bibtex
@misc{sun2025rlgrok,
  title = {RL Grokking Recipe: How Does RL Unlock and Transfer New Algorithms in LLMs?},
  author = {Yiyou Sun and Yuhan Cao and Pohao Huang and Haoyue Bai and Hannaneh Hajishirzi and Nouha Dziri and Dawn Song},
  year = {2025},
  month = {sep},
  eprint = {2509.21016},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
  doi = {10.48550/arXiv.2509.21016},
  url = {https://arxiv.org/abs/2509.21016}
}
```

**Note:** Some components like (SQL) are still under development. Please check individual component READMEs for the latest status and contact authors if you're interested in contributing or using these datasets.
