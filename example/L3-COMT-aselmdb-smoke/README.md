# L3-COMT ASELMDB smoke (datahub / annotate) + Enerzymette AL

Minimal slice of the `L3-COMT-mixed-frag6-1` active-learning campaign, adapted for
Enerzyme PR [#80](https://github.com/Benzoin96485/Enerzyme/pull/80) (`data_format: aselmdb`
and annotate → ASE DB), plus a **tiny** Enerzymette AL iteration that uses the real
`enerzymette enerzyme_active_learning` launcher (not a hand-rolled workflow script).

## What is included

| Path | Role |
|------|------|
| `fixtures/fragments_tiny.sdf` | 3 small extracted fragments (supplier input) |
| `fixtures/fragments_tiny.pkl` | 3 QM-labeled frames from the campaign (legacy pickle) |
| `fixtures/atomic_energy.csv` | B3LYP/6-31Gs PCM atomic energies |
| `fixtures/pcm_radii` | TeraChem PCM radii (vendored; no host path) |
| `fixtures/terachem_template.in` | Annotate template (`pcm_radii_file pcm_radii`) |
| `fixtures/COMT_2ZVJ.pdb` / `cluster.xyz` | Tiny AL topology + initial structure |
| `config/annotate.yaml` | ASE LMDB annotate (`output_file: *.aselmdb`) |
| `config/annotate_pickle.yaml` | Enerzymette-compat pickle (`pickle_name: fragments.pkl`) |
| `config/annotate_legacy_pickle.yaml` | Campaign-style annotate (pre-#80) for comparison |
| `config/train_aselmdb.yaml` | Minimal SpookyNet Datahub pointing at `.aselmdb` |
| `config/train_uma_qs.yaml` | Minimal `uma_qs` + Q/S readout train (1 epoch; checkpoint via env) |
| `config/train_uma_flow_qs.yaml` | Minimal `uma_flow_qs` CFM train (1 epoch; needs `.[flow]` + fairchem) |
| `fixtures/fragments_flow_tiny.pkl` | 3 tiny molecules with synthetic `chrg` / `spin_dens` for flow |
| `scripts/pickle_to_aselmdb.py` | Convert campaign pickles → ASE LMDB |
| `enerzymette_al/` | Min-scale simulate / extract / annotate / train for AL |

No model checkpoints or full training sets are checked in. Configs use **repo-relative**
paths only — run commands from the Enerzyme repository root.

## Enerzymette note

Enerzymette AL still expects `fragments.pkl`. Use `enerzymette_al/annotate.yaml`
(`pickle_name: fragments.pkl`) until Enerzymette switches to LMDB. Default
`config/annotate.yaml` writes ASE LMDB only (no dual write).

Put the latest Enerzymette tree on `PYTHONPATH` yourself (e.g. open PR branch
`feature/pre-simulation-restraints`).

## Quick checks (no TeraChem)

```bash
python example/L3-COMT-aselmdb-smoke/scripts/pickle_to_aselmdb.py \
  -i example/L3-COMT-aselmdb-smoke/fixtures/fragments_tiny.pkl \
  -o /tmp/fragments_tiny.aselmdb

pytest test/test_aselmdb_al_smoke.py test/test_uma_qs_train_smoke.py test/test_uma_flow_qs_train_smoke.py -q
```

## Live uma_flow_qs train smoke (GPU + fairchem + torchdiffeq)

Same scratch rules as `uma_qs` (avoid tiny `/tmp` quotas). Install optional ODE deps with
`pip install -e ".[flow]"`. Committed config uses placeholder `UMA_CHECKPOINT`.

```bash
export UMA_CHECKPOINT=...
export FAIRCHEM_CACHE_DIR=...
export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
export UMA_SMOKE_OUT=...
export TMPDIR="$UMA_SMOKE_OUT/tmp"
mkdir -p "$TMPDIR"

pytest test/test_uma_flow_qs_train_smoke.py -q -k one_epoch --basetemp="$UMA_SMOKE_OUT/pytest"
```

## Live uma_qs train smoke (GPU + fairchem)

From the Enerzyme repo root on a GPU node. Set cache / checkpoint env vars on the
command line only (do not put host paths in the YAML). The committed config uses
the placeholder `UMA_CHECKPOINT`.

```bash
# Point UMA_CHECKPOINT at a local uma-s-*.pt under your fairchem cache
export UMA_CHECKPOINT=...
export FAIRCHEM_CACHE_DIR=...   # if your install expects it
export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled

# On shared GPU nodes, /tmp often has a small per-user quota; put scratch elsewhere
# (UMA saves a large .pth). Example: local node disk under $UMA_SMOKE_OUT.
export UMA_SMOKE_OUT=...        # directory for resolved run artifacts
export TMPDIR="$UMA_SMOKE_OUT/tmp"
mkdir -p "$TMPDIR"

pytest test/test_uma_qs_train_smoke.py -q -k one_epoch --basetemp="$UMA_SMOKE_OUT/pytest"
# or resolve the YAML yourself and run:
#   enerzyme train -c /path/to/train_uma_qs_resolved.yaml -o /path/to/out
```

## Live min AL iteration via Enerzymette (TeraChem)

On a GPU node, from the **Enerzyme repo root**. Load modules, conda, `PYTHONPATH`
(Enerzymette + this repo), and any calculator cache env vars **on the command line**
(do not commit host-specific paths). Then:

```bash
EX=example/L3-COMT-aselmdb-smoke
OUT=/tmp/enerzymette_al_smoke_$$
mkdir -p "$OUT" "$OUT/tmp"

# PRETRAIN: directory that contains FF02-SpookyNet/ (your choice; not committed)
enerzymette enerzyme_active_learning \
  -p "$PRETRAIN" \
  -cp uma \
  -pp sammt \
  -rp "$EX/fixtures/COMT_2ZVJ.pdb" \
  -ix "$EX/fixtures/cluster.xyz" \
  -sc "$EX/enerzymette_al/simulate.yaml" \
  -ec "$EX/enerzymette_al/extract.yaml" \
  -ac "$EX/enerzymette_al/annotate.yaml" \
  -tc "$EX/enerzymette_al/train.yaml" \
  -o "$OUT" \
  -t "$OUT/tmp" \
  -n 1 \
  -np 0 \
  -r 0.5 \
  -rm hard
```

Scale knobs already minimized in `enerzymette_al/`: MD `n_step: 4`, PLUMED
`dump_interval: 1`, extract 1 fragment/frame @ 3 Å, annotate `n_processes: 1`, train
`max_epochs: 1`. Success looks like `FF02-SpookyNet-0_fragments/fragments.pkl`,
`FF02-SpookyNet-1_training/training_completed`, and `FF02-SpookyNet-1` under `$OUT`.
