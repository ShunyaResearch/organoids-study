# organoids-study

Code for "Brain Organoids versus Small Neural Networks on Cart-Pole: A Budget-Matched Comparison of Learning".

## Setup

Tested with Python 3.10.12 on Linux x86_64, CPU only: numpy 2.2.6, pandas 2.3.3, scipy 1.15.3, torch 2.10.0,
snntorch 1.0.0 (pinned in `requirements.txt`).

1. Install [uv](https://docs.astral.sh/uv/), or use nix (`nix-shell` in this folder provides uv).
2. `./setup.sh` installs Python 3.10.12 and the packages into `.venv/`.
3. Download `aggregated_experiment_data.pkl` from the UCSC deposit, https://zenodo.org/records/17684862 (CC-BY 4.0),
   into `data/`. Re-run `./setup.sh` to check its checksum.

The pickle is loaded with a restricted unpickler that only allows three numpy globals.

## Run

Everything in the paper, in one go (all agents below, 10 seeds each, then the comparison with the organoids and the
reference policies):

```
source .venv/bin/activate
bash scripts/paper_runs.sh
```

Or step by step:

```
python -I scripts/organoid_runs.py data/aggregated_experiment_data.pkl   # organoid table -> out/
python -I scripts/train.py AGENT --tune --seeds 9000-9004                # choose lr / episodes per update
python -I scripts/train.py AGENT --seeds 0-9 --lr LR --epu EPU           # 100 episodes (--min-steps 2984: step-matched)
python -I scripts/compare.py "AGENT_lrLR_epuEPU_e100_s*" ...             # vs the organoids, Holm over all given
python -I scripts/reference_policies.py                                  # zero force, random, P, PD, interface
python -I scripts/replay.py data/aggregated_experiment_data.pkl          # open-loop replay of the organoid runs
```

Agents and the settings used in the paper (learning rate, episodes per update):

| Agent | What it is | lr | epu |
|---|---|---|---|
| `mlp` | MLP 2-63-63 on the last two angles | 3e-3 | 4 |
| `gru` | GRU, 36 units, on the current angle | 3e-3 | 1 |
| `snn` | 64 recurrent LIF neurons, 1 update per decision | 3e-4 | 1 |
| `snn5`, `snn10` | 5 or 10 LIF updates per decision | 3e-3 | 4 |
| `snn5x2` | `snn5` on the last two angles | 3e-3 | 4 |
| `res5x2` | spiking reservoir: `snn5x2` with only the readout trained | 3e-3 | 4 |
| `snn_out0` | `snn` with small output initialisation | 1e-3 | 4 |
| `snn5_out0`, `snn10_out0`, `snn5x2_out0`, `res5x2_out0` | the others with small output initialisation | 3e-3 | 4 |
| `if_mlp`, `if_snn5` | MLP / SNN-K5 (small init) through the organoids' interface | 3e-3 | 4 |
| `if_gru` | GRU through the organoids' interface | 1e-2 | 4 |

Runs are deterministic per seed (one CPU thread per run).

## Layout

```
organoids/  data.py  env.py  metrics.py  agents.py  ppo.py  interface.py  stats.py
scripts/    organoid_runs.py  train.py  compare.py  reference_policies.py  replay.py  paper_runs.sh
```
