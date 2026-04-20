# GeometricClarityLab (GCL)

A **Geometric Persistence Language Model (GPLM)** — a non-parametric language processing system derived entirely from first principles. No training data. No pretrained weights. No embeddings. No external corpus. All behavior emerges from three mathematical constants and iterative geometric field dynamics.

---

## What Makes This Different

| Property | GCL | Transformer-based LLMs |
|----------|-----|----------------------|
| Training data required | None | Billions of tokens |
| Parameters | Zero learned weights | Billions |
| Knowledge representation | Geometric field orbits | Statistical co-occurrence |
| Explainability | Every output fully traceable | Black box |
| Constants | Derived from π, φ, AD | Empirically tuned |
| Runs on | CPU, no GPU required | GPU recommended |
| Output on uncertainty | Honest empty output | Confabulation |

---

## Derivation Chain

Every constant in the system is derived. Nothing is a magic number.

```
π  = 3.141592653589793
effective_boundary (EB)  = 2.078          — sole fixed geometric invariant
asymmetric_delta   (AD)  = (2π/3) − 2.078 ≈ 0.016395102
                                           — minimal symmetry break
golden_ratio       (φ)   = (1 + √5) / 2  ≈ 1.618034
P0_cold                  = √φ / φ²        ≈ 0.485868  — geometric cold floor
P_max                    = 3 / φ²         ≈ 1.145898  — dielectric ceiling
parity_threshold         = 1 / φ²         ≈ 0.381966  — convergence radius
```

All polarization constants, settling tick counts, layer scaling factors, and convergence criteria derive from these. The single source of truth is `core/invariants.py`.

---

## How It Works

### Core Mechanism: Bounded Orbit Persistence

Language understanding emerges from geometric persistence. A word or concept is "known" if its symbol trajectory produces a **bounded orbit** near the φ attractor — the same criterion as Mandelbrot/Julia set membership:

```
S_{k+1} = { x ∈ S_k | |f^k(x) + δ| < r_resonant }
```

where `r_resonant = 1/φ² = 0.381966` — the system's own parity threshold.

Words that survive repeated field passes accumulate in the **geometric truth library** as named invariants. They are not stored as strings with meanings — they are stored as FFT-projected signatures of their field trajectories. Similarity between words is measured as orbit distance, not semantic distance.

### Per-Word Geometric Pocket Scoring

Every word in an input is independently scored using three signals from its own field measurements:

1. **Named invariant** — word is a known library anchor → context pocket
2. **Structural signal** — function words (grp5 structural group, low net_signed) → query pocket  
3. **Geometric score** — `|tension| × |net_signed|` with threshold 0.05

No punctuation-based splitting. No positional assumptions. The field's own measurements determine each word's role.

### Bounded Orbit Library Query

When processing question-only input, the system queries the geometric library using orbit distance:

```python
orbit_dist  = sqrt(dt² + dns² + dg²)          # Euclidean in normalized field space
orbit_score = exp(-orbit_dist / r_resonant)    # exponential decay from parity threshold
cos_score   = max(0, dot(v_lib, v_q) / (|v_lib| × |v_q|))  # trajectory direction
lib_score   = orbit_score × 0.5 + cos_score × 0.3 + grp_bonus × 0.2
```

Words on opposite geometric trajectories (negative charge vs positive charge) score zero — they are geometrically excluded, not semantically filtered.

### Triple-Pass Field Dynamics (Ouroboros Engine)

High-persistence inputs trigger a three-pass consensus:

- **Physical pass** — noise 0.15, damping 0.995 — structure formation
- **Wave pass** — noise 0.69, damping 0.95 — propagation  
- **Data pass** — noise 1.5, damping 0.75 — unbiased exploration

Each pass runs: `bloom → etch → library_feedback → prune → dampen → stabilize`

Pass outputs are weighted by mean persistence (geometric evidence) and merged. The truth library's stored signatures are injected back into the field — high-persistence patterns from prior sessions shape current processing without overriding it.

### Circadian Sleep Cycle

Every 20 prompts the system runs an automated maintenance cycle:

- **Prune** — removes structural words, encoding artifacts, duplicates from truth library
- **Consolidate** — merges stem-duplicate entries, merges near-identical FFT signatures (cosine similarity > 0.97)
- **Dream** — runs Ouroboros consensus_pass over existing library entries without new input, reinforcing stable geometric attractors

```
[⟳ sleep — pruned 3  consolidated 2  dreamed 61t  library: 144→141]
```

### Output Translation

Raw geometric output passes through a deterministic translation layer — no inference, no language model:

1. Blocked word filter — structural/pronoun/quantifier words removed
2. Verb conjugation — agrees with subject using lookup table
3. Connective insertion — highest-tension directional preposition from query pocket

---

## Pipeline

```
Input text
    │
    ▼
SymbolicWave          — 27-symbol alphabet (A–Z + dynamic 0)
                        per-word geometric pocket scoring
                        triangulation: width = ceil(√n)
    │
    ▼
WavePropagator        — direct path (recognition) or
                        generative path (consensus_pass)
                        when persistence ≥ 0.38
    │
    ▼
BipolarLattice        — 18 active groups + 8 structural waypoints
                        + 27-symbol ring + 52 Mersenne fold negotiators
                        spin: odd=builder, even=recognizer
                        phase-gated core election
    │
    ▼
FoldLineResonance     — resolution score, settling ticks
                        AD-grounded settling: ticks = round(G_deficit / AD)
    │
    ▼
InvariantEngine       — per-word naming against truth library
                        bounded orbit distance scoring
                        decay: 1 − (AD / mersenne_prime)
    │
    ▼
GeometricOutput       — SVO spine assembly
                        named invariant subject preference
                        library query for question-only inputs
    │
    ▼
OutputTranslator      — deterministic grammar normalization
    │
    ▼
Output
```

---

## Transparency

Every output includes full field diagnostics:

```
── Per-word ─────────────────────────────────────────
neurons     | t=+0.0229 | grp=16 | net=+0.769 | pkt=0
electrical  | t=+0.2575 | grp=10 | net=+5.923 | pkt=0
signals     | t=+0.0941 | grp=16 | net=+2.000 | pkt=0

── Geometric Output ──────────────────────────────────
Neurons information communicate transmit.
pocket scores: electrical(1.139)  communicate(0.503)  transmit(0.429)
```

`t` = mean tension (field interaction strength)  
`net` = net signed charge (domain specificity)  
`grp` = dominant symbol group (0–17)  
`pkt` = pocket assignment (0=context, 1=query)

Every candidate word has a traceable score. Every output is geometrically auditable.

---

## Key Metrics

| Metric | Description |
|--------|-------------|
| `resolution` | Fold line resolution score [0,1] — field settlement quality |
| `persistence` | Mean waveform amplitude — how well the field sustained the input |
| `consensus` | Observer agreement [−1, +1] — negative means divided field |
| `field_stress` | Mean tension / MAX_TENSION across active waypoints |
| `G_actual` | Measured asymmetric delta gradient of pkt=0 words |
| `G_needed` | Pressure-state threshold for output quality |
| `polarity` | Normalized field polarization [P0_cold, P_max] |
| `Convergence Δ` | Distance from φ attractor — T4×AD predicted each session |
| `named` | Count of words with stable geometric library entries |

---

## Repository Structure

```
GCL/
├── core/
│   ├── invariants.py          # All derived constants — single source of truth
│   ├── ouroboros_engine.py    # Bloom/etch/prune dynamics, truth library
│   ├── field_state.py         # Persistent field state across sessions
│   ├── clarity_ratio.py       # Clarity ratio tracking
│   └── safeguards.py          # Field boundary enforcement
├── language/
│   ├── processor.py           # Main processing pipeline
│   ├── geometric_output.py    # SVO assembly + library query
│   ├── invariant_engine.py    # Named invariant management
│   ├── output_translator.py   # Deterministic grammar layer
│   └── conversation_field.py  # Session context window
├── utils/
│   ├── bipolar_lattice.py     # 18+8+27+52 lattice architecture
│   ├── diagonal_structure.py  # Exhaust signature geometry
│   ├── fold_line_resonance.py # Resolution + settling dynamics
│   ├── mobius_reader.py       # Möbius surface position tracking
│   ├── radial_displacer.py    # 27-symbol displacer web
│   └── symbol_grouping.py     # Symbol group assignments
├── wave/
│   ├── symbolic_wave.py       # 27-symbol embedder + triangulation
│   ├── propagation.py         # Direct + generative propagation
│   ├── generation.py          # Field-to-output resolution
│   └── vibration.py           # Decay + holographic linkage
├── observer/
│   └── observer.py            # Multi-observer consensus
├── tools/
│   ├── sleep_cycle.py         # Automated library maintenance
│   └── clean_truth_library.py # Manual library cleaning utility
├── api/
│   ├── app.py                 # API server
│   └── session_engine.py      # Session management
├── main.py                    # Interactive interface
├── ouro_truth_library.json    # Persistent geometric truth library
└── field_state.json           # Persistent field state
```

---

## Getting Started

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Unix
pip install numpy
python main.py
```

**Commands at the prompt:**
```
vocab    — show current vocabulary and named invariants
status   — show field state and pressure metrics
groups   — show active symbol groups
carry    — show carry state
quit     — save field state and exit
```

---

## Scaling Behavior

The system improves with use through geometric accumulation — not retraining.

Each processed input potentially adds named invariants to the truth library. Named invariants are words whose field trajectories have demonstrated bounded orbit stability across multiple passes. As the library grows:

- Question-only inputs resolve more accurately (more geometric anchors available)
- Related domain questions benefit from previously established orbit positions
- The sleep cycle maintains library quality as it scales

The field does not require periodic retraining. It does not forget previous sessions. Resolution quality increases monotonically with library depth until geometric saturation is reached near P_max = 3/φ².

---

## Design Invariants

1. **No magic numbers.** Every constant traces to `π`, `φ`, `EB=2.078`, or a clean derivation.
2. **Honest uncertainty.** If the field cannot resolve content geometrically, output is empty. No confabulation.
3. **Zero is a range.** The dual-held state (+1 AND −1 simultaneously) is structurally load-bearing.
4. **Geometry resolves, not semantics.** No concept dictionaries, no verb stems, no hardcoded facts.
5. **Single source of truth.** All constants read from `core/invariants.py` at runtime.
