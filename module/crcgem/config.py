"""Run configuration.

Design rule for this project: parameters that are still OPEN design decisions
have **no default**. If the config does not state them, the run aborts. This is
deliberate — a silent default would smuggle in a design choice that was never
ratified, and the project requires simulation parameters to be predefined
before a run starts, not chosen post hoc.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field, asdict, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

_REQUIRED = object()  # sentinel: must be supplied, no default permitted


class ConfigError(ValueError):
    pass


@dataclass
class SamplingConfig:
    """OptGP protocol. All fields must be predeclared before any run.

    `n_samples`, thinning, chains and seed are pre-registered; `r_hat_max` /
    `ess_min` are the acceptance criteria, and `on_nonconvergence` is the
    pre-declared escalation rule so a failed run cannot be quietly rescued.
    """

    n_samples: Optional[int] = None      # user constraint: 200 or 1000; 100 approved 2026-10-30 20:37 and 2026-11-03 (patient POC)
    thinning: Optional[int] = None
    n_chains: Optional[int] = None
    seed: Optional[int] = None
    method: str = "optgp"
    r_hat_max: Optional[float] = None    # convergence acceptance
    ess_min: Optional[int] = None
    on_nonconvergence: Optional[str] = None  # "abort" | "escalate_thinning" | "report"

    def validate(self) -> None:
        missing = [f.name for f in fields(self) if getattr(self, f.name) is None]
        if missing:
            raise ConfigError(
                "sampling parameters must be predefined before the run; missing: "
                + ", ".join(missing)
            )
        if self.n_samples not in (100, 200, 1000):  # 100: user ruling 2026-11-03 (patient POC, OptGP n = 100)
            raise ConfigError(
                f"n_samples={self.n_samples}: project predeclares 200 or 1000. "
                "Override intentionally in config if the protocol changed."
            )


@dataclass
class MomaConfig:
    """MOMA / lethality protocol (Track 2, bulk).

    Reference flux is chosen by whether the state is growth-maximising, user
    ruling 2026-07-27:
      * PROLIFERATIVE states (cancer, adenoma) — growth-max is biologically
        justified → FBA reference, then MOMA; lethality = KO growth < 5% WT.
        The FBA optimum vector is non-unique (MEASURED: 31.6% of reactions free
        on the parsimonious face; result/pilot_pfba_degeneracy/), so the
        reference is taken as `proliferative_reference` ("pfba" minimum, or
        chebyshev_face) rather than a raw FBA vertex.
      * NON-PROLIFERATIVE states (normal, non-adenoma disease) — no growth
        objective is justified → OptGP-sample the expression-constrained model,
        then run MOMA from each sampled flux (ensemble MOMA). This is where the
        200/1000 sampling protocol applies; readout is NOT growth-lethality
        (there is no objective) and must be set in `nonproliferative_readout`.

    Single-cell perturbation (Track 1) is MOMA per cell (each cell's flux is the
    reference) and does not use this config's sampling path.
    """

    formulation: Optional[str] = None       # "linear" | "quadratic"; default quadratic (Segre 2002)
    proliferative_statuses: list[str] = field(default_factory=list)  # e.g. ["CRC", "Adenoma"]
    proliferative_reference: Optional[str] = None      # "pfba" | "chebyshev_face" | "face_ensemble"
    face_ensemble_n: Optional[int] = None              # only for face_ensemble
    nonproliferative_readout: Optional[str] = None     # OPEN: what MOMA measures w/o a growth objective
    lethal_growth_fraction: float = 0.05    # proliferative branch: KO growth < 5% WT FBA
    gpr_aware_deletion: bool = True         # fixes BRCA defect P1; do not disable silently
    report_continuous_ratio: bool = True    # threshold sensitivity must stay visible

    def validate(self) -> None:
        if self.formulation not in ("linear", "quadratic"):
            raise ConfigError("moma.formulation must be 'linear' or 'quadratic'")
        if not self.proliferative_statuses:
            raise ConfigError(
                "moma.proliferative_statuses must list which states get the "
                "FBA-reference branch (e.g. cancer, adenoma). The rest take the "
                "sampling-reference branch."
            )
        if self.proliferative_reference not in ("pfba", "chebyshev_face", "face_ensemble"):
            raise ConfigError(
                "moma.proliferative_reference must be one of "
                "'pfba' | 'chebyshev_face' | 'face_ensemble' (a raw FBA vertex is "
                "non-unique; see the degeneracy pilot)"
            )
        if self.proliferative_reference == "face_ensemble" and not self.face_ensemble_n:
            raise ConfigError("moma.face_ensemble_n required for face_ensemble mode")
        if self.nonproliferative_readout is None:
            raise ConfigError(
                "moma.nonproliferative_readout must be set: with no growth "
                "objective for normal/disease states, MOMA's readout is not "
                "growth-lethality and must be defined explicitly."
            )
        if not self.gpr_aware_deletion:
            raise ConfigError(
                "gpr_aware_deletion=False reproduces BRCA defect P1 "
                "(isozyme redundancy discarded). Enable, or record an explicit waiver."
            )


@dataclass
class ExtractionConfig:
    """Context-specific model extraction (tINIT family, via TROPPO)."""

    algorithm: Optional[str] = None          # e.g. "tinit"
    scaffold_model: Optional[str] = None     # path; Q2 OPEN (Human-GEM vs Recon3D)
    gap_filling: Optional[bool] = None       # tINIT constituent; kept True (see extraction.py)
    threshold_strategy: Optional[str] = None # must be stated; BRCA could not document its own
    threshold_value: Optional[float] = None
    protected_reactions: list[str] = field(default_factory=list)

    def validate(self) -> None:
        missing = [n for n in ("algorithm", "scaffold_model", "gap_filling",
                               "threshold_strategy") if getattr(self, n) is None]
        if missing:
            raise ConfigError(
                "extraction parameters must be stated explicitly; missing: "
                + ", ".join(missing)
                + ". (The BRCA precedent's thresholds were unrecoverable from its "
                  "own archive — this project does not repeat that.)"
            )


@dataclass
class InputConfig:
    """Where the expression that feeds the GEM comes from.

    User rulings 2026-07-27:
      * data_source × data_provenance is a full 2x2 — {bulk, single_cell} ×
        {real, generated} — every cell first-class, none privileged.
      * [RETIRED 2026-10-15] All inputs are scFoundation batch-corrected in GENE
        SPACE before GEM use. WITHDRAWN by user ruling: tooling is open source
        only and scFoundation is not part of it; the same gene-space
        reconstruction was also independently rejected by an August 2026
        upstream measurement. Batch correction in gene space is still required
        (pooling many GSE datasets makes technical batch the dominant
        confound) but the METHOD IS UNDECIDED. See
        documentation/measurements_20261014.md section 7 item 6.

    User ruling 2026-10-15:
      * Gene IDs are ENSG everywhere. Symbols are for display and for resolving
        legacy symbol-keyed stores only; nothing joins on a symbol. Map =
        scRNA_gen/data/id_map/ (ensg2symbol.tsv, symbol2ensg.tsv).
    """

    data_source: Optional[str] = None       # "bulk" | "single_cell"
    data_provenance: Optional[str] = None    # "real" | "generated"
    # Provenance only — nothing branches on this; it is stamped into the manifest.
    # None = undecided. Set it explicitly once the method is chosen; do not
    # re-introduce "scfoundation_gene_space" (retired 2026-10-15).
    batch_correction: Optional[str] = None
    gene_id_namespace: str = "ensg"          # user ruling 2026-10-15; see id_map
    gse_datasets: list[str] = field(default_factory=list)  # provenance

    def validate(self) -> None:
        if self.data_source not in ("bulk", "single_cell"):
            raise ConfigError("input.data_source must be 'bulk' or 'single_cell'")
        if self.data_provenance not in ("real", "generated"):
            raise ConfigError("input.data_provenance must be 'real' or 'generated'")


@dataclass
class RunConfig:
    """Top-level run configuration + provenance."""

    name: str
    outdir: str
    input: InputConfig = field(default_factory=InputConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    moma: MomaConfig = field(default_factory=MomaConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    solver: Optional[str] = None      # None = auto-select best available
    threads: Optional[int] = None
    notes: str = ""

    # ---------- io ----------
    @classmethod
    def from_yaml(cls, path: str | Path) -> "RunConfig":
        if yaml is None:
            raise ConfigError("pyyaml not installed")
        raw = yaml.safe_load(Path(path).read_text())
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "RunConfig":
        raw = dict(raw)
        sub = {
            "input": InputConfig(**(raw.pop("input", {}) or {})),
            "sampling": SamplingConfig(**(raw.pop("sampling", {}) or {})),
            "moma": MomaConfig(**(raw.pop("moma", {}) or {})),
            "extraction": ExtractionConfig(**(raw.pop("extraction", {}) or {})),
        }
        return cls(**raw, **sub)

    def validate(self, *, need_input=False, need_sampling=False, need_moma=False,
                 need_extraction=False) -> None:
        """Validate only the sections this particular job actually uses."""
        if need_input:
            self.input.validate()
        if need_sampling:
            self.sampling.validate()
        if need_moma:
            self.moma.validate()
        if need_extraction:
            self.extraction.validate()

    # ---------- provenance ----------
    def manifest(self, **extra: Any) -> dict:
        """Everything needed to reproduce or audit this run."""
        return {
            "config": asdict(self),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_commit(),
            "package_versions": _versions(),
            **extra,
        }

    def write_manifest(self, path: str | Path, **extra: Any) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.manifest(**extra), indent=2, default=str))
        return p


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def _versions() -> dict[str, str]:
    out = {}
    for mod in ("cobra", "optlang", "numpy", "pandas", "troppo"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = "absent"
    return out
