"""GI-tract set 2: falsifiers G1 and G2 of design result/external_plan_20261103/design_external_gi_normal_20261103.md section 1.3,
counted over the 6 GI-tract cancer cell lines only (85b prints its F1/F2 count over every labeled cell line of the dataset, which
here includes the normal cell lines Hs 729 and BPH-1). Reads the 85b outputs (paired.tsv, per_line.tsv); no new metric.
G1: supported if GNN - MOMA AUROC (set all) > 0 with bootstrap interval excluding 0 in >= 5 of 6; falsified if the point difference
    <= 0 in >= 3 of 6; otherwise inconclusive.
G2: F1 = set all, GNN > base rate with interval excluding 0; F2 = set discordant, GNN AUROC > 0.5 with interval excluding 0.5.
    Counted as in 85b (F1 count >= 5 or F2 count >= 5 of 6 -> falsified); the per-line union count (F1 or F2) is also written.
Usage: python 140b_eval_gi_falsifiers.py <85b eval dir> <out.tsv>
"""
import os, sys, time
import pandas as pd

GI = ["ACH-001654", "ACH-000216", "ACH-000466", "ACH-000427", "ACH-000307", "ACH-000976"]  # design 1.1 pick
ev, out = sys.argv[1], sys.argv[2]
assert not os.path.exists(out), f"{out} exists (no overwrite)"
pp = pd.read_csv(f"{ev}/paired.tsv", sep="\t", comment="#"); pl = pd.read_csv(f"{ev}/per_line.tsv", sep="\t", comment="#")
rows = []
for (g, t), x in pl.groupby(["graph", "tokens"]):
    q = pp[(pp.graph == g) & (pp.tokens == t) & (pp.set == "all") & pp.line.isin(GI)]
    m = q[q.comparator == "moma"].set_index("line"); b = q[q.comparator == "base_rate"].set_index("line")
    assert len(m) == 6 and len(b) == 6, f"{g}/{t}: {len(m)} MOMA rows, {len(b)} base-rate rows for the 6 GI cell lines"
    sup = int(((m["diff"] > 0) & m.excludes_0).sum()); le0 = int((m["diff"] <= 0).sum())
    G1 = "supported" if sup >= 5 else ("falsified" if le0 >= 3 else "inconclusive")
    f1 = (b["diff"] > 0) & b.excludes_0
    dsc = x[(x.set == "discordant") & x.line.isin(GI)].set_index("line")
    f2 = ((dsc.AUROC_gnn > 0.5) & (dsc.AUROC_gnn_lo > 0.5) & (dsc.scored == True)).reindex(GI, fill_value=False)
    n1, n2, nu = int(f1.sum()), int(f2.sum()), int((f1.reindex(GI, fill_value=False) | f2).sum())
    rows.append(dict(graph=g, tokens=t, n_gi_lines=6, G1_diff_gt0_excl0=sup, G1_diff_le0=le0, G1=G1,
                     G2_F1_count=n1, G2_F2_count=n2, G2_F2_discordant_sets_scored=int((dsc.scored == True).sum()),
                     G2_F1_or_F2_per_line=nu, G2_falsified_85b_rule=bool(n1 >= 5 or n2 >= 5)))
r = pd.DataFrame(rows)
with open(out, "w") as fh:
    fh.write(f"# source: 85b eval {ev} (paired.tsv, per_line.tsv); 6 GI-tract cancer cell lines {GI}; design section 1.3\n"
             f"# script_python/140b_eval_gi_falsifiers.py; {time.ctime()}; n rows {len(r)}\n")
    r.to_csv(fh, sep="\t", index=False)
print(r.to_string(index=False))
