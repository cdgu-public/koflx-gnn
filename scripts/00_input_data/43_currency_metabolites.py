"""Currency metabolite list for graph encoding A' (user ruling 2026-10-26).

Source: Table A1, "List of all potential currency metabolites (CM)", in Stelzer M,
Sun J, Kamphans T, Fekete SP, Zeng A-P. Integr Biol 2011;3(11):1071-1086
[ref: doi: 10.1039/c1ib00008j], the upgrade of Ma & Zeng 2003
[ref: doi: 10.1093/bioinformatics/19.2.270]. Preprint kept at
data/graph/Stelzer2011_IntegrBiol_preprint.pdf (Table A1, pp. 24-27).

Simplification, stated in the methods: the source rules judge currency per
reaction ("CMs cannot be defined per se"); here the Table A1 list is applied to
every reaction and every compartment. In A' a currency metabolite is moved from
edges to reaction node features, so nothing is discarded.

Mapping to Human-GEM metabolite names: 'exact' = identical name; 'synonym' =
same chemical entity under Human-GEM's name (each listed with its reason);
'unmapped' = absent from Human-GEM or a generic placeholder.
"""
import csv
import cobra

ROOT = "/data/workspace/cdgu/Winery/seeding/CRC_GEM"
NA = None
# (Table A1 compound, frequency in former database, frequency in upgraded database), table order
TABLE_A1 = [
    ("H2O", 1255, 2236), ("H+", 359, 1269), ("O2", 379, 858), ("NADP+", 387, 725),
    ("NADPH", 386, 722), ("NAD+", 392, 666), ("NADH", 387, 657), ("ATP", 311, 467),
    ("CO2", 250, 430), ("Pi", 253, 395), ("CoA", 196, 371), ("ADP", 234, 333),
    ("NH3", 212, 297), ("PPi", 158, 288), ("S-Adenosyl-L-homocysteine", 91, 236),
    ("UDP", 95, 224), ("Acceptor", 73, 189), ("Reduced acceptor", 73, 187),
    ("H2O2", 98, 162), ("AMP", 89, 160), ("Glutathione", 27, 65), ("CMP", 48, 64),
    ("dTDP", 6, 48), ("HCl", 8, 45), ("3'-Phosphoadenylyl sulfate", 21, 45),
    ("Adenosine 3',5'-bisphosphate", 22, 44), ("GDP", 27, 40), ("FAD", 29, 40),
    ("FADH2", 29, 38), ("CTP", 27, 31), ("GTP", 23, 31), ("Oxidized ferredoxin", 17, 31),
    ("Reduced ferredoxin", 15, 31), ("H2SO4", 20, 29), ("UMP", 23, 28), ("H2SO3", 19, 26),
    ("UTP", 23, 26), ("Oxidized glutathione", 12, 26), ("HNO2", 17, 25),
    ("Acyl-carrier protein", 22, 23), ("H2S", 10, 20), ("ITP", 20, 20),
    ("Oxidized thioredoxin", 15, 20), ("Reduced thioredoxin", 15, 20),
    ("Cl- (Chloride)", 2, 19), ("IDP", 19, 19), ("Ferrocytochrome c", 2, 19), ("H2", 14, 18),
    ("Ferricytochrome c", 13, 18), ("GMP", 14, 16), ("CO", 2, 13), ("NDP", 2, 13),
    ("CDP", 11, 13), ("dATP", 11, 13), ("e-", 4, 12), ("IMP", 12, 12), ("dADP", 12, 12),
    ("PQQ", 11, 12), ("HNO3", 8, 11), ("HS2O3", 8, 11), ("FMN", 5, 11), ("OH-", 1, 10),
    ("NO", 8, 10), ("PQQH2", 10, 10), ("Ubiquinone", 9, 10), ("Cl- (Chloride ion)", 1, 9),
    ("dCMP", 8, 9), ("dGTP", 7, 9), ("Oxidized rubredoxin", 4, 9), ("Reduced rubredoxin", 4, 9),
    ("PPPi", 5, 8), ("dCTP", 6, 8), ("dTMP", 5, 8), ("dUMP", 6, 8), ("Ubiquinol", 7, 8),
    ("S", 5, 7), ("H2Se", 7, 7), ("SeO3", 6, 7), ("HBr", NA, 7), ("dTTP", 6, 7), ("dUTP", 7, 7),
    ("Fe2+", NA, 6), ("dCDP", 6, 6), ("dGDP", 6, 6), ("dUDP", 6, 6), ("Reduced FMN", 0, 6),
    ("HSO3-", 1, 5), ("Ferricytochrome b5", 1, 5), ("Ferrocytochrome b5", 1, 5),
    ("Electron-transferring flavoprotein", 2, 5), ("Red. elec.-transf. flavo.", 2, 5),
    ("O2.-", 2, 4), ("Fe3+", NA, 4), ("dGMP", 4, 4), ("Oxidized adrenal ferredoxin", 5, 4),
    ("Reduced adrenal ferredoxin", 5, 4), ("Oxidized flavoprotein", 2, 4),
    ("Reduced flavoprotein", 2, 4), ("I2", 3, 3), ("I-", 2, 3), ("H2SeO4", 2, 3),
    ("Alkylphosphonate", 1, 3), ("dAMP", 3, 3), ("Ferredoxin", 3, 3),
    ("Ferricytochrome c2", 2, 3), ("Ferrocytochrome c2", 2, 3), ("Quinone", NA, 3),
    ("Hydroquinone", NA, 3), ("Coenzyme F420", 3, 3), ("Reduced coenzyme F420", 3, 3),
    ("Oxidized dithiothreitol", 0, 3), ("Dithiothreitol", 0, 3), ("Amino group donor", 0, 3),
    ("RING", 3, 3), ("Hg2+", 1, 2), ("HI", 2, 2), ("H3PSeO3", 1, 2), ("Mg2+", 2, 2), ("Fe", 6, 2),
    ("Oxidized azurin", 1, 2), ("Reduced azurin", 1, 2), ("Cytochrome c", 2, 2),
    ("Ferrocytochrome c3", 1, 2), ("Ferricytochrome c-553", 1, 2), ("Ferrocytochrome c-553", 1, 2),
    ("Ferricytochrome b1", 2, 2), ("Ferrocytochrome b1", 2, 2), ("Apocytochrome c", 2, 2),
    ("Donor", 0, 2), ("Oxidized donor", 0, 2), ("H2S2O3", 1, 1), ("SO2", 1, 1), ("Mn", 0, 1),
    ("Br-", 0, 1), ("NMP", NA, 1), ("Cytochrome c3", 1, 1), ("Ferricytochrome c3", 1, 1),
    ("Ferricytochrome b-561", 0, 1), ("Ferrocytochrome b-561", 0, 1),
    ("Flavodoxin semiquinone", 1, 1), ("Dihydroflavodoxin", 1, 1), ("Oxidized flavodoxin", NA, 1),
    ("Reduced flavodoxin", 0, 1), ("Decylubiquinone", NA, 1), ("Decylubiquinol", NA, 1),
    ("Oxidized plastocyanin", 1, 1), ("Reduced plastocyanin", 1, 1),
    ("Oxidized putidaredoxin", 0, 1), ("Putidaredoxin", 0, 1),
    ("NTP", NA, NA), ("dNTP", NA, NA), ("dNDP", NA, NA), ("dNMP", NA, NA),
]
# Table A1 name -> (Human-GEM name, reason). Only same-entity renamings.
SYNONYM = {
    "S-Adenosyl-L-homocysteine": ("SAH", "abbreviation"),
    "Glutathione": ("GSH", "reduced glutathione"),
    "Oxidized glutathione": ("GSSG", "glutathione disulfide"),
    "3'-Phosphoadenylyl sulfate": ("PAPS", "abbreviation"),
    "Adenosine 3',5'-bisphosphate": ("PAP", "abbreviation"),
    "H2SO4": ("sulfate", "ionized form"),
    "H2SO3": ("sulfite", "ionized form"),
    "HSO3-": ("sulfite", "ionized form"),
    "HNO2": ("nitrite", "ionized form"),
    "Reduced thioredoxin": ("thioredoxin", "Human-GEM names the reduced form 'thioredoxin'"),
    "Cl- (Chloride)": ("chloride", "same ion"),
    "Cl- (Chloride ion)": ("chloride", "same ion"),
    "Ferrocytochrome c": ("ferrocytochrome C", "case"),
    "Ferricytochrome c": ("ferricytochrome C", "case"),
    "Ferrocytochrome b5": ("ferrocytochrome B5", "case"),
    "Ferricytochrome b5": ("ferricytochrome B5", "case"),
    "Cytochrome c": ("cytochrome-C", "punctuation"),
    "Apocytochrome c": ("apocytochrome-C", "punctuation"),
    "Oxidized ferredoxin": ("oxidized ferredoxin", "case"),
    "Reduced ferredoxin": ("reduced ferredoxin", "case"),
    "Oxidized adrenal ferredoxin": ("oxidized adrenal ferredoxin", "case"),
    "Reduced adrenal ferredoxin": ("reduced adrenal ferredoxin", "case"),
    "Ubiquinone": ("ubiquinone", "case"),
    "Ubiquinol": ("ubiquinol", "case"),
    "Hydroquinone": ("hydroquinone", "case"),
    "Oxidized thioredoxin": ("oxidized thioredoxin", "case"),
    "Oxidized dithiothreitol": ("oxidized dithiothreitol", "case"),
    "Dithiothreitol": ("dithiothreitol", "case"),
    "HS2O3": ("thiosulfate", "ionized form"),
    "OH-": ("hydroxide", "same ion"),
    "PPPi": ("triphosphate", "inorganic triphosphate"),
    "H2Se": ("selenide", "ionized form"),
    "SeO3": ("selenite", "same anion"),
    "H2SeO4": ("selenate", "ionized form"),
    "H3PSeO3": ("selenophosphate", "same entity"),
    "Reduced FMN": ("FMNH2", "same entity"),
    "O2.-": ("O2-", "superoxide"),
    "I2": ("iodine", "same entity"),
    "I-": ("iodide", "same ion"),
}


def main():
    cobra.Configuration().solver = "glpk"
    m = cobra.io.read_sbml_model(f"{ROOT}/data/scaffold/Human-GEM.xml")
    by_name = {}
    for x in m.metabolites:
        by_name.setdefault(x.name, []).append(x.id)
    rows, ids = [], set()
    for name, f_old, f_new in TABLE_A1:
        if name in by_name:
            hg, how, note = name, "exact", ""
        elif name in SYNONYM:
            hg, note = SYNONYM[name]; how = "synonym"
            assert hg in by_name, (name, hg)
        else:
            hg, how, note = "", "unmapped", "not in Human-GEM, or a generic placeholder"
        mids = sorted(by_name.get(hg, []))
        ids.update(mids)
        rows.append(dict(tableA1_compound=name, freq_former=f_old, freq_upgraded=f_new,
                         humangem_name=hg, mapping=how, note=note,
                         n_compartments=len(mids), humangem_met_ids=";".join(mids)))
    out = f"{ROOT}/data/graph/currency_metabolites_StelzerA1.tsv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t"); w.writeheader(); w.writerows(rows)
    with open(f"{ROOT}/data/graph/currency_metabolite_ids.txt", "w") as f:
        f.write("\n".join(sorted(ids)) + "\n")
    from collections import Counter
    c = Counter(r["mapping"] for r in rows)
    print(f"Table A1 entries {len(rows)}: {dict(c)}; distinct Human-GEM names "
          f"{len({r['humangem_name'] for r in rows if r['humangem_name']})}; metabolite ids (all compartments) {len(ids)}")


if __name__ == "__main__":
    main()
