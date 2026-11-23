# avogadro-sevennet

Energy and gradient models for Avogadro 2 using the
[SevenNet](https://github.com/MDIL-SNU/SevenNet) machine-learning potentials
(PyPI package `sevenn`).

## Models

| Name | Checkpoint | Task (modal) | Elements | Periodic | Charge / spin |
| --- | --- | --- | --- | --- | --- |
| SevenNet-Omni | `7net-omni` | `mpa` (PBE(+U), MPtrj + sAlex) | 1-118 | yes | neutral closed-shell only |
| SevenNet-MF-ompa | `7net-mf-ompa` | `mpa` (PBE(+U), MPtrj + sAlex) | 1-118 | yes | neutral closed-shell only |
| SevenNet-Omni (OMol25) | `7net-omni` | `omol25_low` (wB97M-V, OMol25) | 1-83 | no | neutral closed-shell only |

Both are conservative (energy-derived forces) equivariant graph networks with a
6 Å cutoff. They are multi-task / multi-fidelity: one set of weights carries
several "modals", each reproducing a different level of theory. The upstream
documentation recommends `mpa` as the default for most systems, so both entries
use it. SevenNet-Omni also offers other modals (for example `omol25_low`,
`omat24`, `matpes_r2scan`) and SevenNet-MF-ompa offers `omat24`; they are not
exposed here.

The models take no charge or spin input, so Avogadro only offers them for
neutral, closed-shell systems. The `mpa` task is trained on bulk inorganic
crystals, so treat isolated-molecule energies with care.

SevenNet-Omni (OMol25) uses the same Omni checkpoint (no extra download) with
the `omol25_low` task, the representative wB97M-V molecular task upstream lists.
The sibling `omol25_high` task is for high-spin configurations only and is not
exposed. OMol25 contains charged and open-shell systems, but the model has no
charge or spin input, so it cannot tell them apart; only neutral closed-shell
molecules are offered. It is molecular, so periodic systems are not offered.
Its elements (1-83, matching OMol25's 83 elements) are those with nonzero
per-element reference energies for that task in the checkpoint.

Element coverage for the other two entries comes from the checkpoints' type maps (H through Og); it says
nothing about how well each element is sampled in training.

The models run in float64 (the upstream default is float32), so energy differences
near a minimum are not lost to round-off.

Energies are reported in kJ/mol, gradients in kJ/mol/Å. The device is CUDA when
available, otherwise CPU.

## Setup

The checkpoints (about 100 MB each) are downloaded from the SevenNet GitHub
releases on first use and cached inside the Python environment. Pre-download
them so the first energy call does not stall:

```bash
pixi install
pixi run setup
```

Loading a model still takes roughly 10-15 seconds per process on CPU.

## Platforms

Upstream lists only Linux. `linux-64` and `osx-arm64` are locked; `osx-arm64`
was tested (CPU). `linux-64` and CUDA were not tested. Windows is not listed.

## Testing

```bash
pixi run test
```

## Licenses

- SevenNet code: MIT.
- Pretrained weights: distributed as GitHub release assets with no separate
  license stated by upstream; the repository license is MIT. The weights are
  trained on public datasets (MPtrj, sAlex, OMat24, and for Omni also OMol25,
  OC20/OC22, ODAC23, SPICE, QCML, MatPES and others), whose own terms may
  apply. Check upstream before commercial use.
- This plugin: BSD-3-Clause.

## Citation

If you use these models, please cite the SevenNet papers:

- Park et al., J. Chem. Theory Comput. 20, 4857 (2024),
  doi:10.1021/acs.jctc.4c00190 (SevenNet code).
- Kim et al., J. Am. Chem. Soc. 147, 1042 (2024),
  doi:10.1021/jacs.4c14455 (SevenNet-MF-ompa, multi-fidelity learning).
- Kim et al., Nat. Commun. 17, 3432 (2026),
  doi:10.1038/s41467-026-70195-8 (SevenNet-Omni).

## Suggestions

Please file suggestions and bugs at
<https://github.com/ghutchis/avogadro-sevennet/issues>.
