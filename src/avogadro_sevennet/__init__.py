#  This source file is part of the Avogadro project
#  This source code is released under the BSD-3-Clause License, (see "LICENSE").
#  https://github.com/ghutchis/avogadro-sevennet/

"""Entry point for the avogadro-sevennet plugin.

Avogadro calls this as:
    avogadro-sevennet <identifier> --protocol binary-v1 [--lang <locale>] [--debug]

with the molecule bootstrap JSON on stdin as one compact line, followed by
binary coordinate frames for as long as the model stays loaded.
"""

import argparse
import os


def setup():
    """Download and cache every exposed model so the first energy is not slow.

    Avogadro runs this once after installing the plugin. The checkpoints are
    fetched from the SevenNet GitHub releases (about 100 MB each).
    """
    from ase.build import molecule

    from ._server import MODELS, make_calculator

    atoms = molecule("H2O")
    for identifier in MODELS:
        print(f"Preparing {identifier}...", flush=True)
        atoms.calc = make_calculator(identifier)
        atoms.get_potential_energy()
    print("SevenNet setup complete.")


def main():
    parser = argparse.ArgumentParser("avogadro-sevennet")
    # Avogadro always passes the feature identifier as the first argument.
    parser.add_argument("feature")
    parser.add_argument("--lang", nargs="?", default="en")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--protocol", nargs="?", default="binary-v1")
    args = parser.parse_args()

    # Protect the binary protocol at the file-descriptor level. ML libraries
    # (and their C/C++ code) print to fd 1 at import time and while building
    # calculators, and Avogadro never resyncs after a stray byte. Keep a private
    # copy of the real stdout for the protocol and point fd 1 at stderr.
    proto_out = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)

    match args.feature:
        case "SevenNet-Omni" | "SevenNet-MF-ompa" | "SevenNet-Omni-OMol25":
            from ._server import run

            run(args.feature, proto_out)
