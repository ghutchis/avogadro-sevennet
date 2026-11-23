#  This source file is part of the Avogadro project
#  This source code is released under the BSD-3-Clause License, (see "LICENSE").
#  https://github.com/ghutchis/avogadro-sevennet/

"""Shared binary-protocol energy server for the SevenNet models.

Avogadro works in kJ/mol and Angstrom, and wants the gradient dE/dx, not the
force. SevenNet is evaluated through its ASE calculator, in eV and eV/Angstrom.
"""

import json
import sys

import numpy as np
from ase import Atoms
from ase.cell import Cell

from avogadro.energy import EnergyServer

# 1 eV in kJ/mol
_EV_TO_KJ_MOL = 96.4853321

# identifier -> (SevenNet pretrained name, modal)
#
# SevenNet-Omni and SevenNet-MF-ompa are multi-task / multi-fidelity: one set
# of weights, several "modals" that each reproduce a different level of theory.
# The upstream documentation recommends modal "mpa" (PBE(+U), MPtrj + sAlex)
# as the default for most systems, so that is what both models use here.
MODELS = {
    "SevenNet-Omni": ("7net-omni", "mpa"),
    "SevenNet-MF-ompa": ("7net-mf-ompa", "mpa"),
    # omol25_low: wB97M-V molecular task (OMol25), the representative
    # molecular task upstream lists. omol25_high is high-spin-only.
    "SevenNet-Omni-OMol25": ("7net-omni", "omol25_low"),
}


def _device():
    # CUDA if present, else CPU. Never MPS: no float64 there.
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def make_calculator(identifier):
    """Build the SevenNetCalculator for one of the models above."""
    name, modal = MODELS[identifier]

    # stdout (fd 1) has already been redirected to stderr by main(), so the
    # download notice and library chatter cannot reach the protocol stream.
    from sevenn.calculator import SevenNetCalculator

    return SevenNetCalculator(model=name, modal=modal, device=_device())


def _prepare(identifier, mol_cjson):
    numbers = np.array(mol_cjson["atoms"]["elements"]["number"])
    coords = np.array(
        mol_cjson["atoms"]["coords"]["3d"], dtype=float
    ).reshape(-1, 3)

    # num_atoms must be the full count Avogadro sent: it is the atom count in
    # every binary frame, images included.
    num_atoms = len(numbers)

    # Avogadro includes periodic image atoms at fractional coordinate ~1.0.
    # They duplicate atoms at 0.0, so keep a mask and drop them for the model.
    keep = np.ones(num_atoms, dtype=bool)

    if "unitCell" in mol_cjson:
        fractional = mol_cjson["atoms"]["coords"].get("3dFractional", [])
        if fractional:
            frac = np.array(fractional, dtype=float).reshape(-1, 3)
            keep = ~np.any(np.abs(frac - 1.0) < 1e-4, axis=1)

        lattice = np.array(
            mol_cjson["unitCell"]["cellVectors"], dtype=float
        ).reshape(3, 3)
        atoms = Atoms(numbers[keep], coords[keep], cell=Cell(lattice), pbc=True)
    else:
        atoms = Atoms(numbers[keep], coords[keep], pbc=False)

    atoms.calc = make_calculator(identifier)
    return {"atoms": atoms, "keep": keep, "num_atoms": num_atoms}


def _energy(model, coords):
    model["atoms"].set_positions(coords[model["keep"]])
    return float(model["atoms"].get_potential_energy()) * _EV_TO_KJ_MOL


def _energy_and_gradient(model, coords):
    atoms = model["atoms"]
    atoms.set_positions(coords[model["keep"]])

    energy = float(atoms.get_potential_energy()) * _EV_TO_KJ_MOL

    # force = -gradient; image rows come back as zeros.
    gradient = np.zeros((model["num_atoms"], 3), dtype=np.float64)
    gradient[model["keep"]] = atoms.get_forces() * -_EV_TO_KJ_MOL
    return energy, gradient


def _respond(model, request):
    if request.wants_energy_and_gradient:
        energy, gradient = _energy_and_gradient(model, request.coords)
        request.send_energy_and_gradient(energy, gradient)
    elif request.wants_gradient:
        request.send_gradient(_energy_and_gradient(model, request.coords)[1])
    else:
        request.send_energy(_energy(model, request.coords))


def run(identifier, proto_out):
    """Serve requests; proto_out is the saved binary stream (original fd 1)."""
    # One compact JSON line, then binary frames. Read the line from
    # sys.stdin.buffer: the text wrapper would swallow the first frame.
    bootstrap = json.loads(sys.stdin.buffer.readline())

    # SevenNet takes no charge or spin input, so bootstrap["charge"] and
    # bootstrap["spin"] are deliberately unused (support.ions/radicals are
    # false, so Avogadro only offers these models for neutral closed-shell
    # systems).
    model = _prepare(identifier, bootstrap["cjson"])

    with EnergyServer(
        sys.stdin.buffer, proto_out, model["num_atoms"]
    ) as server:
        for request in server.requests():
            try:
                _respond(model, request)
            except Exception as error:  # noqa: BLE001
                request.send_error(str(error))
