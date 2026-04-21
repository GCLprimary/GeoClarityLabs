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
| Constants | Derived from π, φ, Ω_m=0.315 | Empirically tuned |
| Runs on | CPU, no GPU required | GPU recommended |
| Output on uncertainty | Honest sparse output | Confabulation |
| Improves by | Geometric accumulation | Retraining |

---

## Derivation Chain

Every constant in the system is derived. Nothing is a magic number.

```
Ω_m  = 0.315          — Planck 2018 measured matter density
EB   = (1/Ω_m − 1) × π/3  ≈ 2.078   — effective boundary
AD   = 2π/3 − EB      ≈ 0.016395102  — asymmetric delta
φ    = (1 + √5) / 2   ≈ 1.618034     — golden ratio
P0_cold = √φ / φ²     ≈ 0.485868     — geometric cold floor
P_max   = 3 / φ²      ≈ 1.145898     — dielectric ceiling
r_resonant = 1 / φ²   ≈ 0.381966     — convergence radius
```

All thresholds, multipliers, tick counts, and scaling factors derive from these. The single source of truth is `core/invariants.py`.

---

## How It Works

### Bounded Orbit Persistence

A word is "known" if its symbol trajectory produces a bounded orbit near the φ attractor — the same convergence criterion as the Mandelbrot set:

```
S_{k+1} = { x ∈ S_k | |f^k(x) + δ| < r_resonant }
r_resonant = 1/φ² = 0.381966
```

Words that survive repeated field passes accumulate in the **geometric truth library** as named invariants stored as FFT-projected field signatures. The library grows without retraining.

### Four-Arm Role Assignment

The 27-symbol alphabet maps to the Dual-13 signed integer system (−13 to +13). The sign and parity of each symbol's group ID directly encodes its syntactic role:

```
N arm: gid > 0, odd  (+1,+3,+5…+13) — builders   → SUBJECT / primary nouns
S arm: gid < 0, odd  (−1,−3,−5…−13) — inverters  → VERB
E arm: gid > 0, even (+2,+4,+6…+12) — recognizers → OBJECTS / relations
W arm: gid < 0, even (−2,−4,−6…−12) — compressors → CONNECTIVES
```

Output skeleton: **[N-subject] [S-verb] [connective] [N-objects] [E-objects]**

Chain length is resolution-gated: at cold start (res=0.15) outputs are 2 words. At full parity lock (res=0.875) outputs reach 5 words — one per Möbius twist point.

### Accumulation Without Retraining

Each processed prompt potentially adds named invariants to the truth library. As the library grows:

- Question-only inputs resolve more accurately (more geometric anchors)
- Related domain questions benefit from prior orbit positions
- Ask the same question twice — the second answer is richer

The field does not require periodic retraining. Resolution quality increases monotonically with library depth until saturation near P_max = 3/φ².

### Transparency

Every output includes full field diagnostics. Per-word output shows:

```
electrical  | t=+0.2242 | grp=+13 | net=+5.923 | pkt=0
communicate | t=+0.0712 | grp= -1 | net=+2.615 | pkt=0
```

`t` = mean tension · `net` = net signed charge · `grp` = Dual-13 group · `pkt` = pocket (0=context, 1=query)

Every candidate word has a traceable score. Every output is geometrically auditable.

---

## Pipeline

```
Input
  │
  ▼
SymbolicWave        — 27-symbol alphabet, per-word geometric pocket scoring
  │
  ▼
WavePropagator      — direct (recognition) or generative (consensus_pass) path
  │
  ▼
BipolarLattice      — 18 outer + 8 structural + 27-symbol ring + 52 Mersenne strings
                      quad displacer: NS axis (builders/inverters) ↔ EW axis (recognizers/compressors)
  │
  ▼
FoldLineResonance   — resolution score, AD-grounded settling ticks
  │
  ▼
InvariantEngine     — per-word naming, bounded orbit distance scoring
  │
  ▼
GeometricOutput     — four-arm role chain assembly
  │
  ▼
OutputTranslator    — deterministic grammar normalization
  │
  ▼
Output
```

---

## Observed Behavior

These outputs are produced geometrically — the system has no knowledge of biology, physics, or dendrochronology:

```
"How do neurons transmit electrical signals…"
→ "Electrical communicates to transmit."   (first pass, cold library)
→ "Neurons transmit electrical to signals." (second pass, named anchors loaded)

"What slow biological process inside tree trunks adds a new concentric ring…"
→ "Age inside adds concentric records rainfall."   (first pass)
→ "Concentric adds inside records rainfall."       (second pass, 9 new invariants named)

"How do vaccines introduce weakened forms of a pathogen…"
→ "Vaccines introduce to pathogen recognize."  (full library, res=0.875)
```

Outputs are compressed geometric sentences — not prose, not summaries. The field reports what it measured.

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
│   ├── geometric_output.py    # Four-arm role chain assembly + library query
│   ├── invariant_engine.py    # Named invariant management
│   ├── output_translator.py   # Deterministic grammar normalization
│   └── conversation_field.py  # Session context window
├── utils/
│   ├── bipolar_lattice.py     # 18+8+27+52 lattice, quad displacer axis state
│   ├── diagonal_structure.py  # Exhaust signature geometry
│   ├── fold_line_resonance.py # Resolution + settling dynamics
│   ├── mobius_reader.py       # Möbius surface position tracking
│   ├── radial_displacer.py    # 27-symbol displacer web
│   └── symbol_grouping.py     # Dual-13 direct group assignment (27 groups)
├── wave/
│   ├── symbolic_wave.py       # 27-symbol embedder + triangulation
│   ├── propagation.py         # Direct + generative propagation
│   ├── generation.py          # Field-to-output resolution
│   └── vibration.py           # Decay + holographic linkage
├── observer/
│   └── observer.py            # Multi-observer consensus
├── tools/
│   ├── sleep_cycle.py         # Automated library maintenance (every 10 prompts)
│   └── clean_truth_library.py # Manual library cleaning utility
├── api/
│   ├── app.py                 # FastAPI server
│   └── session_engine.py      # Per-user session management
├── main.py                    # Interactive local interface
├── ouro_truth_library.json    # Persistent geometric truth library
└── field_state.json           # Persistent field state
```

---

## Getting Started

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Unix
pip install numpy fastapi uvicorn
python main.py
```

**Commands at the prompt:**
```
vocab    — show vocabulary and named invariants
status   — show field state and pressure metrics
groups   — show active symbol groups
diag     — fold line diagnostics
carry    — show carry state
quit     — save field state and exit
```

**For the web API:**
```bash
uvicorn api.app:app --reload --port 8000
```

---

## Design Invariants

1. **No magic numbers.** Every constant traces to π, φ, EB=2.078 (from Ω_m=0.315), or a clean derivation.
2. **Honest uncertainty.** If the field cannot resolve content geometrically, output is sparse. No confabulation.
3. **Zero is a range.** The dual-held state (+1 AND −1 simultaneously) is structurally load-bearing.
4. **Geometry resolves, not semantics.** No concept dictionaries, no verb stems, no hardcoded facts.
5. **Single source of truth.** All constants read from `core/invariants.py` at runtime.
6. **Accumulation, not retraining.** The library grows through use. Resolution improves monotonically.
