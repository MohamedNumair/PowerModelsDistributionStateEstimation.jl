# PowerGridModel benchmark for the Literal PMDSE solve options

This folder pins the Literal PMDSE PowerGridModel (PGM) solve options
(`:iterative_linear`, `:newton_raphson`) to PGM's own state-estimation results.

* `generate_golden.py` — runs [power-grid-model](https://github.com/PowerGridModel/power-grid-model)
  on a handful of its symmetric state-estimation validation cases
  (`tests/data/state_estimation/*`) and writes the golden data.
* `golden.json` — machine-readable network + sensors + the node voltages PGM
  computes for each calculation method (informational / diffable).
* `pgm_cases_data.jl` — the same data as a `const PGM_CASES = [...]` literal; its
  contents are embedded verbatim in `../pgm_cases.jl` (the test does **not** read
  this file at runtime, so the suite needs neither Python nor PGM installed).

## Cases

| name | PGM folder | nodes | methods | what it exercises |
|---|---|---|---|---|
| `single_node` | `single-node-source-sym-voltage-sensor` | 1 | il, nr | trivial voltage-phasor pin |
| `os1_2msr` | `1os2msr` | 3 | il, nr | redundant set, branch from/to power, source/load injection, a disabled (σ=1e15) sensor, Y-bus + to-side back-calc |
| `os1_2msr_no_angle` | `1os2msr-no-angle` | 3 | il, nr | magnitude-only voltages → slack-angle gauge |
| `single_line_load` | `single-line-load-il` | 2 | il | inconsistent data (weighting matters), Kalman-combined duplicate load sensor |

The single-phase PGM network is the special case of the four-wire literal model
with one conductor per bus and no neutral; per-unit bases match PGM
(`V_base = u_rated`, `S_base = base_power_3p = 1e6`).

## Regenerate

```bash
pip install power-grid-model numpy
python3 generate_golden.py /path/to/power-grid-model     # or set PGM_REPO
# then paste pgm_cases_data.jl into the PGM_CASES block of ../pgm_cases.jl
```
