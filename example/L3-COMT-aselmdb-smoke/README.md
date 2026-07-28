# L3-COMT ASELMDB smoke (datahub / annotate)

Minimal slice of the `L3-COMT-mixed-frag6-1` active-learning campaign, adapted for
Enerzyme PR [#80](https://github.com/Benzoin96485/Enerzyme/pull/80) (`data_format: aselmdb`
and annotate → ASE DB).

## What is included

| Path | Role |
|------|------|
| `fixtures/fragments_tiny.sdf` | 3 small extracted fragments (supplier input) |
| `fixtures/fragments_tiny.pkl` | 3 QM-labeled frames from the campaign (legacy pickle) |
| `fixtures/atomic_energy.csv` | B3LYP/6-31Gs PCM atomic energies |
| `fixtures/terachem_template.in` | Annotate template (basis/xc/solvent) |
| `config/annotate.yaml` | ASE LMDB annotate (`output_file: *.aselmdb`) |
| `config/annotate_pickle.yaml` | Enerzymette-compat pickle (`pickle_name: fragments.pkl`) |
| `config/annotate_legacy_pickle.yaml` | Campaign-style annotate (pre-#80) for comparison |
| `config/train_aselmdb.yaml` | Minimal SpookyNet Datahub pointing at `.aselmdb` |
| `scripts/pickle_to_aselmdb.py` | Convert campaign pickles → ASE LMDB |

No model checkpoints or full training sets are checked in.

## Enerzymette note

Enerzymette AL still expects `fragments.pkl`. Point campaigns at `annotate_pickle.yaml`
(`pickle_name: fragments.pkl`) until Enerzymette switches to LMDB. Default `annotate.yaml`
writes ASE LMDB only (no dual write).

## Quick checks

```bash
# Convert tiny pickle → aselmdb
python example/L3-COMT-aselmdb-smoke/scripts/pickle_to_aselmdb.py \
  -i example/L3-COMT-aselmdb-smoke/fixtures/fragments_tiny.pkl \
  -o /tmp/fragments_tiny.aselmdb

# Unit / integration tests (no TeraChem required)
pytest test/test_aselmdb_al_smoke.py -q
```

Optional real annotate on a GPU node with TeraChem on `PATH` (e.g. `d-7-5-1` after `module load terachem`):

```bash
enerzyme annotate -c example/L3-COMT-aselmdb-smoke/config/annotate.yaml \
  -o /tmp/aselmdb_annot_out -t /tmp/aselmdb_annot_tmp -s 0 -e 1
```
