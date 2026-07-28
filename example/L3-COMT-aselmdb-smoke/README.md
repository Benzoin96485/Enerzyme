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

pytest test/test_aselmdb_al_smoke.py -q
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
