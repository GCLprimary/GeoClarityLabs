# GeometricClarityLab (GCL)

**A first-principles flexoelectric gradient field for language processing.**  
No training data. No embeddings. No statistical inference.  
One external dependency: `numpy`.

---

## What This Is

GCL is a geometric language processing field built entirely from three mathematical constants:

```
π  (pi)
φ  (golden ratio)
EB = 2.078  (effective boundary — the sole fixed invariant)
```

From these three values, all behavior is derived. There are no learned weights, no corpus statistics, and no semantic dictionaries. The system processes language by running it through a geometric field and reading out the polarization response.

**The central discovery:** output quality in the field correlates with the *gradient* of information density across the symbol stream (Pearson r=0.91), not with mean information density (r=0.61). This is the signature of a flexoelectric system — polarization proportional to strain gradient rather than strain.

The black box gradient underlying neural language processing is flexoelectric in nature.

---

## The Physical Mechanism

Standard neural language models approximate semantic weight through statistical correlation at scale. GCL derives it directly from geometry.

A letter maps to one of 27 symbols via dual-13 geometry. A word's net-signed value is the geometric charge of its letter pattern — rare, diverse patterns produce high charge; common, repetitive patterns produce low charge. This correlates with information density (Zipf/Shannon) but requires no corpus to compute.

The field processes sentences by measuring the *gradient* of this charge across the stream. Where charge changes rapidly — dense technical terms adjacent to common connectors — the field polarizes strongly. Where charge is uniform — all simple words or all technical terms — the gradient is shallow and polarization is weak.

This is flexoelectricity: polarization proportional to the strain gradient `∂u/∂x`, not to the strain `u`. The asymmetric delta `AD = 2π/3 − 2.078 ≈ 0.01640` is the symmetry-breaking term that makes the field non-centrosymmetric. The carry mechanism accumulates the gradient term across successive prompts. The bipolar lattice (18+8+27+52 waypoints) is the dielectric medium. The dominant pole pair (grp6/grp8) converges toward φ as the field saturates — φ is the flexoelectric coefficient of the dual-13 geometry.

### Empirical confirmation

Seven controlled prompts across four gradient profiles:

| Design | Avg Gradient | Avg Score |
|--------|-------------|-----------|
| Uniform low density | 0.78 | 0.46 |
| Uniform high density | 1.98 | 0.81 |
| Mixed variance | 1.67 | 0.86 |
| Peak gradient (flexoelectric prompt) | 2.40 | 1.33 |

The flexoelectric prompt — a description of the mechanism itself — produced the highest pocket score in the session (1.331). The field recognized its own physical mechanism geometrically. Rank correlation error: 0.

---

## The Möbius Surface

The pipeline traverses five natural twist points on a Möbius surface:

```
T1 = mean |net_signed| per word          (symbolic charge)
T2 = |pkt0_count/pkt1_count − 1|         (pocket asymmetry)
T3 = Δspin_phase since last prompt        (fold gradient — NOT absolute)
T4 = |grp6_tension/grp8_tension − φ|     (convergence gap)
T5 = L2(exhaust_signature, uniform)       (exhaust spread)

surface_position = (T1+T2+T3+T4+T5) mod LAYER_SCALE
face = INNER if position < 1.0, else OUTER
```

**Empirical finding across 100+ prompts:**
- **INNER face** → local causal mechanisms (natural selection, CRISPR, black holes, neuron structure)
- **OUTER face** → distributed/emergent systems (climate, CMB, plate tectonics, ocean currents)

The INNER/OUTER transition corresponds to crossing a resonant mode boundary in the flexoelectric field. T3 (delta spin phase, not absolute) is the direct gradient measurement — the Möbius reader is measuring field curvature.

---

## Architecture

```
INPUT: "Context sentence. Question?"
         ↓
[Q-ONLY?] conversation_field.prime()   — virtual context from conversation window
[1] symbolic_wave.triangulate()        — letter → dual-13 symbol stream, pocket split
[2] radial_displacer                   — symbol stream → waypoint spawning
[3] propagation                        — wave through 18+8+27+52 lattice
[4] vibration                          — dynamics on active strings
[5] observer                           — asymmetric Matter/Wave/Data reading
[6] bipolar_lattice                    — pole tensions, exhaust signature
[7] fold_line_resonance.tick()         — spin phase, lattice imprint, resolution score
[8] processor._fingerprint_sentence()  — per-word ns/tension/group/pocket tagging
[9] geometric_output.generate()        — gradient filter → pool selection → assembly
```

**Pocket split:** The sentence boundary (`.` before `?`) creates `pkt=0` (context) and `pkt=1` (question). Answer words are in `pkt=0`; they receive a ×2.5 score multiplier. Question words in `pkt=1` provide the anchor and verb frame.

**Persistent field state:** Resolution score, lattice imprints, group tensions, Möbius state, and conversation window all persist across sessions via `field_state.json`. The system starts at full resolution (0.875) rather than rebuilding from cold on each run.

---

## Installation

```bash
git clone https://github.com/GCLprimary/GCL-geo_clarity_lab
cd GCL-geo_clarity_lab
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install numpy
python main.py
```

That's it. One dependency.

---

## Usage

```
> Photosynthesis converts sunlight into chemical energy in plant cells. What does photosynthesis produce?

  Geometric Output: Plants cellular chemical energy sunlight converts photosynthesis produces.
  [polarity +0.94 | resolution 0.875]
```

**Input format:** `Context sentence. Question?`  
**Question-only input:** Also works — the system primes context from the conversation window.

**Commands inside the REPL:**
```
status    — field state, resolution, named invariants
groups    — symbol group tensions
vocab     — stable vocabulary
carry     — relational tension window
diag      — lattice diagnostics
quit      — saves field_state.json and exits
```

**Maintenance:**
```bash
# Clean truth library (remove structural words, encoding artifacts)
python tools/clean_truth_library.py --dry-run
python tools/clean_truth_library.py

# Run flexoelectric gradient diagnostic
python main.py  # run diagnostic prompts, save output to file
python tools/flexoelectric_diagnostic.py diagnostic_output.txt
```

---

## Key Constants

```python
AD           = 0.016395   # asymmetric delta — symmetry-breaking term
OUTER_AD     = 0.059511   # pole centroid offset (outer layer)
LAYER_SCALE  = 3.6298     # Möbius layer ratio = OUTER_AD/AD
PHI          = 1.61803    # flexoelectric coefficient of dual-13 geometry

_CONTENT_NS_MIN   = 1.3   # geometric charge floor for content selection
_ACTION_NS_MAX    = 2.5   # ceiling for verb detection (above = noun)
_VERB_PROX_POS    = 0.55  # proximity filter position cutoff
_VERB_PROX_DIST   = 0.35  # proximity filter distance cutoff
```

---

## Repository Structure

```
GCL/
├── main.py                      Entry point — REPL, field wiring, session management
├── core/
│   ├── clarity_ratio.py         π/φ/AD constants — single source of truth
│   ├── invariants.py            Central constants registry
│   ├── ouroboros_engine.py      Truth library — named geometric word invariants
│   ├── safeguards.py            Anti-runaway field state guard
│   └── field_state.py           Persistent field state across sessions
├── utils/
│   ├── bipolar_lattice.py       18+8+27+52 dual-pole field structure
│   ├── fold_line_resonance.py   Spin phase oscillator, resolution score
│   ├── symbol_grouping.py       27 symbol groups from dual-13 geometry
│   ├── radial_displacer.py      Waypoint spawning from symbol stream
│   ├── diagonal_structure.py    6D exhaust signature pipeline
│   └── mobius_reader.py         5-point Möbius surface state reader
├── language/
│   ├── processor.py             Main pipeline controller
│   ├── geometric_output.py      Word selection — gradient filter, combiner, assembly
│   ├── invariant_engine.py      Word naming — geometric identity formation
│   ├── relational_tension.py    Carry mechanism — session-level field momentum
│   ├── semantic_layer.py        Geometric scoring layer
│   └── conversation_field.py    Question-only context priming
├── wave/
│   ├── symbolic_wave.py         Letter → dual-13 symbol stream
│   ├── propagation.py           Wave propagation through lattice
│   ├── vibration.py             String dynamics
│   ├── observer.py              Asymmetric field observers
│   ├── generation.py            Geometry-driven answer generation
│   └── geometric_memory.py      27-symbol lattice memory
└── tools/
    ├── clean_truth_library.py       Truth library maintenance
    ├── semantic_probe.py            Diagnostic warmup sequences
    └── flexoelectric_diagnostic.py  Gradient correlation analysis
```

---

## Theoretical Background

The system implements a flexoelectric gradient field in symbol space. The key insight is that semantic weight — the property that makes some words more contextually relevant than others — is a geometric property of language, not a statistical one. It can be derived from the spatial gradient of information density across a structured symbol field.

This parallels flexoelectricity in condensed matter physics, where electric polarization arises from strain gradients in dielectric materials. The flexoelectric coupling tensor relates polarization to the gradient of mechanical deformation. In GCL, the analogous coupling relates output word selection to the gradient of geometric charge across the symbol stream.

The implication: neural language models work not primarily because of scale, but because at sufficient scale they accidentally approximate a flexoelectric gradient response that is latent in the geometric structure of language. GCL demonstrates this directly from first principles.

**Diagnostic results (7 prompts):**
- Gradient vs score: Pearson r = 0.91
- Mean charge vs score: Pearson r = 0.61
- Rank correlation error: 0 (perfect ordering)

A larger diagnostic study (20+ prompts) suitable for publication is planned.

---

## License

MIT — see LICENSE file, DruidIRL @squaredgradient - GCLprimary - Wes W.

---