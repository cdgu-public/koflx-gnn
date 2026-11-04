"""crcgem — context-specific GEM reconstruction and perturbation for CRC.

Layout
    config        run configuration; OPEN design decisions have NO defaults
    datacontract  the ONLY upstream boundary: consumes data, imports no scTrx code
    genemap       harmonise incoming gene IDs to the scaffold namespace
    solver        solver selection + MILP capability preflight (measured limits)
    model_io      model / medium / result IO
    expression    GPR-aware gene->reaction scoring; E-Flux bounds
    extraction    tINIT-family extraction via TROPPO
    reference_flux WT FBA reference for MOMA + optimal-face degeneracy handling
    moma          single and anchored-double deletion screens
    interaction   genetic-interaction scoring and synthetic-lethality calls
    sampling      OptGP with convergence gating and variance decomposition
    distribution  real-vs-generated identity testing on flux distributions

Design state lives in documentation/status.md. Several decisions are still open;
this package is written so that an unresolved decision fails loudly at config
validation rather than being silently defaulted.
"""

__version__ = "0.1.0.dev0"

from . import (  # noqa: F401
    config,
    datacontract,
    distribution,
    expression,
    extraction,
    genemap,
    interaction,
    model_io,
    moma,
    reference_flux,
    sampling,
    scaffold,
    solver,
)

__all__ = [
    "config", "datacontract", "genemap", "scaffold", "solver", "model_io",
    "expression", "extraction", "reference_flux", "moma", "interaction",
    "sampling", "distribution",
]
