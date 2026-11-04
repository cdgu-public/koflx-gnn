"""Model loading, medium handling, and result persistence."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Mapping, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def load_model(path: str | Path, *, solver_info=None, threads: Optional[int] = None):
    """Load an SBML/JSON/YAML model and attach a solver."""
    import cobra

    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in (".xml", ".sbml"):
        model = cobra.io.read_sbml_model(str(p))
    elif suffix == ".json":
        model = cobra.io.load_json_model(str(p))
    elif suffix in (".yml", ".yaml"):
        model = cobra.io.load_yaml_model(str(p))
    elif suffix == ".mat":
        model = cobra.io.load_matlab_model(str(p))
    else:
        raise ValueError(f"unsupported model format {suffix!r}")

    logger.info("loaded %s: %d reactions, %d metabolites, %d genes",
                p.name, len(model.reactions), len(model.metabolites), len(model.genes))
    if solver_info is not None:
        from . import solver as solver_mod

        solver_mod.apply(model, solver_info, threads=threads)
    return model


def save_model(model, path: str | Path):
    import cobra

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() in (".xml", ".sbml"):
        cobra.io.write_sbml_model(model, str(p))
    elif p.suffix.lower() == ".json":
        cobra.io.save_json_model(model, str(p))
    else:
        raise ValueError(f"unsupported output format {p.suffix!r}")
    return p


def apply_medium(model, medium: Mapping[str, float], *, strict: bool = True):
    """Set exchange bounds from a medium definition.

    Medium is a first-class, pre-registered design variable in this project, not
    a default: colonocytes derive much of their ATP from luminal butyrate while
    tumour cells shift toward glucose/glutamine, so one generic medium cannot
    represent both ends of the trajectory and the choice materially determines
    the flux shift that gets reported.

    Args:
        medium: {exchange_reaction_id: max_uptake (positive number)}
        strict: raise if an id is absent, rather than silently ignoring it.
    """
    unknown = [k for k in medium if k not in model.reactions]
    if unknown:
        msg = f"medium references {len(unknown)} unknown reactions: {unknown[:5]}"
        if strict:
            raise KeyError(msg)
        logger.warning(msg)

    model.medium = {k: v for k, v in medium.items() if k in model.reactions}
    return model


def load_medium(path: str | Path) -> dict[str, float]:
    """Load a medium definition from JSON or a two-column TSV."""
    p = Path(path)
    if p.suffix.lower() == ".json":
        return json.loads(p.read_text())
    df = pd.read_csv(p, sep="\t", header=0, names=["reaction", "uptake"])
    return dict(zip(df["reaction"], df["uptake"].astype(float)))


def write_table(df: pd.DataFrame, path: str | Path, *, index: bool = True) -> Path:
    """Persist a result table (parquet for wide matrices, tsv otherwise)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() == ".parquet":
        df.to_parquet(p, index=index)
    else:
        df.to_csv(p, sep="\t", index=index)
    logger.info("wrote %s (%d x %d)", p, *df.shape)
    return p
