"""Build citable culture-medium definitions mapped onto Human-GEM v2.0.0 exchanges.

Reads ONLY archived sources under data/medium/sources/ (manufacturer formulation
pages/sheets, DepMap 24Q4 condition tables, GDSC, KCLB) plus the scaffold, and
writes data/medium/*.tsv. No FBS composition is encoded anywhere (no defined
formulation exists; see data/medium/README.md). Single process, no LP solved.

Concentration fidelity rules
- Gibco rows: mg/L, MW and mM copied verbatim from the Gibco formulation table
  (parsed from raw HTML; every row re-checked mg/L / MW == mM).
- ATCC rows: g/L copied verbatim from the ATCC sheet (x1000 -> mg/L); mM is
  computed from the molar mass of the printed salt/hydrate form (IUPAC 2021
  standard atomic weights), stated per row.
- species_mM = component mM x stoichiometry of the mapped species in the printed
  form. Counter-ions of ORGANIC components (HCl, Na, Ca of pantothenate, Cl of
  choline/thiamine) are not tallied; ions of INORGANIC salts are.
"""
import csv, json, re, sys, gzip, hashlib
from collections import defaultdict, OrderedDict
from pathlib import Path

ROOT = Path("/data/workspace/cdgu/Winery/seeding/CRC_GEM")
SRC = ROOT / "data/medium/sources"
OUT = ROOT / "data/medium"
RETRIEVED = "2026-10-26"

# ----------------------------------------------------------------------------
# molar masses (IUPAC 2021 standard atomic weights, abridged)
AW = dict(H=1.008, C=12.011, N=14.007, O=15.999, Na=22.990, Mg=24.305, P=30.974,
          S=32.06, Cl=35.45, K=39.098, Ca=40.078, Fe=55.845, Cu=63.546, Zn=65.38,
          Se=78.971, Co=58.933)


def mw(formula):
    """Molar mass of a formula like 'C6H14N4O2*HCl*H2O' or 'Ca(NO3)2.4H2O'."""
    total = 0.0
    for part in formula.split("*"):
        m = re.match(r"^(\d*\.?\d*)(.*)$", part)
        mult = float(m.group(1)) if m.group(1) else 1.0
        total += mult * _mw_simple(m.group(2))
    return total


def _mw_simple(f):
    stack = [0.0]
    i = 0
    while i < len(f):
        c = f[i]
        if c == "(":
            stack.append(0.0); i += 1
        elif c == ")":
            i += 1; n = re.match(r"\d*", f[i:]).group(0); i += len(n)
            v = stack.pop() * (int(n) if n else 1); stack[-1] += v
        else:
            m = re.match(r"([A-Z][a-z]?)(\d*)", f[i:])
            el, n = m.group(1), m.group(2)
            stack[-1] += AW[el] * (int(n) if n else 1); i += len(m.group(0))
    return stack[0]


# ----------------------------------------------------------------------------
# Component dictionary: printed name -> (class, species list, ATCC formula, note)
# species: list of (Human-GEM extracellular metabolite NAME or None, stoich,
#                   label for unmapped species)
U = None  # unmapped marker
C = {}


def comp(names, cls, species, formula=None, note=""):
    for n in names:
        C[n] = dict(cls=cls, species=species, formula=formula, note=note)


ORG_CI = "organic counter-ion not tallied"
# amino acids
comp(["Glycine"], "amino acid", [("glycine", 1)], "C2H5NO2")
comp(["L-Alanine"], "amino acid", [("alanine", 1)], "C3H7NO2")
comp(["L-Arginine", "L-Arginine (free base)"], "amino acid", [("arginine", 1)], "C6H14N4O2")
comp(["L-Arginine hydrochloride", "L-Arginine·HCl"], "amino acid", [("arginine", 1)], "C6H14N4O2*HCl", ORG_CI)
comp(["L-Asparagine", "L-Asparagine (freebase)"], "amino acid", [("asparagine", 1)], "C4H8N2O3",
     "Gibco lists 'L-Asparagine' with its own MW; copied as printed")
comp(["L-Asparagine-H2O", "L-Asparagine·H2O"], "amino acid", [("asparagine", 1)], "C4H8N2O3*H2O")
comp(["L-Aspartic acid", "L-Aspartic Acid"], "amino acid", [("aspartate", 1)], "C4H7NO4")
comp(["L-Cystine 2HCl", "L-Cystine·2HCl"], "amino acid", [("cystine", 1)], "C6H12N2O4S2*2HCl", ORG_CI)
comp(["L-Cysteine", "L-Cysteine (free base)"], "amino acid", [("cysteine", 1)], "C3H7NO2S")
comp(["L-Cysteine hydrochloride-H2O", "L-Cysteine·HCl·H2O"], "amino acid", [("cysteine", 1)], "C3H7NO2S*HCl*H2O", ORG_CI)
comp(["L-Cystine·HCl·H2O"], "amino acid", [("cysteine", 1)], "C3H7NO2S*HCl*H2O",
     "SOURCE TYPO: ATCC 30-2006 prints 'L-Cystine·HCl·H2O'; the value 0.01756 g/L equals half of "
     "F-12's L-cysteine·HCl·H2O (35.12 mg/L) and there is no cystine·HCl·H2O salt, so it is "
     "read as L-cysteine·HCl·H2O (AMBIGUOUS - user ruling requested)")
comp(["L-Glutamic Acid", "L-Glutamic acid"], "amino acid", [("glutamate", 1)], "C5H9NO4")
comp(["L-Glutamine"], "amino acid", [("glutamine", 1)], "C5H10N2O3")
comp(["L-Histidine", "L-Histidine (free base)"], "amino acid", [("histidine", 1)], "C6H9N3O2")
comp(["L-Histidine hydrochloride-H2O", "L-Histidine·HCl·H2O"], "amino acid", [("histidine", 1)], "C6H9N3O2*HCl*H2O", ORG_CI)
comp(["L-Hydroxyproline", "Hydroxy-L-Proline"], "amino acid", [("trans-4-hydroxy-L-proline", 1)], "C5H9NO3",
     "medium hydroxyproline is trans-4-hydroxy-L-proline (CHEBI:18095 in Human-GEM)")
comp(["L-Isoleucine"], "amino acid", [("isoleucine", 1)], "C6H13NO2")
comp(["L-Leucine"], "amino acid", [("leucine", 1)], "C6H13NO2")
comp(["L-Lysine"], "amino acid", [("lysine", 1)], "C6H14N2O2")
comp(["L-Lysine hydrochloride", "L-Lysine·HCl"], "amino acid", [("lysine", 1)], "C6H14N2O2*HCl", ORG_CI)
comp(["L-Methionine"], "amino acid", [("methionine", 1)], "C5H11NO2S")
comp(["L-Phenylalanine"], "amino acid", [("phenylalanine", 1)], "C9H11NO2")
comp(["L-Proline"], "amino acid", [("proline", 1)], "C5H9NO2")
comp(["L-Serine"], "amino acid", [("serine", 1)], "C3H7NO3")
comp(["L-Threonine"], "amino acid", [("threonine", 1)], "C4H9NO3")
comp(["L-Tryptophan"], "amino acid", [("tryptophan", 1)], "C11H12N2O2")
comp(["L-Tyrosine", "L-Tyrosine (free base)"], "amino acid", [("tyrosine", 1)], "C9H11NO3")
comp(["L-Tyrosine disodium salt dihydrate", "L-Tyrosine·2Na·2H2O", "L-Tyrosine disodium salt"],
     "amino acid", [("tyrosine", 1)], "C9H9NO3Na2*2H2O", ORG_CI)
comp(["L-Valine"], "amino acid", [("valine", 1)], "C5H11NO2")
# vitamins / cofactors
comp(["Biotin", "D-Biotin"], "vitamin", [("biotin", 1)], "C10H16N2O3S")
comp(["Choline chloride", "Choline Chloride"], "vitamin", [("choline", 1)], "C5H14NOCl", ORG_CI)
comp(["D-Calcium pantothenate"], "vitamin", [("pantothenate", 2)], "C18H32N2O10Ca",
     "calcium salt Ca(pantothenate)2: 2 pantothenate per formula unit; " + ORG_CI)
comp(["D-Pantothenic Acid (hemicalcium)"], "vitamin", [("pantothenate", 1)], "C9H16NO5*0.5Ca",
     "hemicalcium salt: 1 pantothenate per formula unit; " + ORG_CI)
comp(["Folic Acid"], "vitamin", [("folate", 1)], "C19H19N7O6")
comp(["Niacinamide", "Nicotinamide"], "vitamin", [("nicotinamide", 1)], "C6H6N2O")
comp(["Nicotinic acid (Niacin)", "Nicotinic Acid"], "vitamin", [("nicotinate", 1)], "C6H5NO2")
comp(["Para-Aminobenzoic Acid", "p-Amino Benzoic Acid"], "vitamin", [(U, 1, "4-aminobenzoate")], "C7H7NO2",
     "UNMAPPED: no 4-aminobenzoate metabolite/exchange in Human-GEM v2.0.0 (human cells do not use PABA)")
comp(["Pyridoxine hydrochloride", "Pyridoxine·HCl"], "vitamin", [("pyridoxine", 1)], "C8H11NO3*HCl", ORG_CI)
comp(["Pyridoxal hydrochloride"], "vitamin", [("pyridoxal", 1)], "C8H9NO3*HCl", ORG_CI)
comp(["Riboflavin"], "vitamin", [("riboflavin", 1)], "C17H20N4O6")
comp(["Riboflavin 5'-phosphate Na", "Riboflavin-5-PO4·Na·2H2O"], "vitamin", [("FMN", 1)], "C17H20N4NaO9P*2H2O",
     "riboflavin 5'-phosphate = FMN (CHEBI:17621); " + ORG_CI)
comp(["Thiamine hydrochloride", "Thiamine·HCl"], "vitamin", [("thiamin", 1)], "C12H17N4OSCl*HCl", ORG_CI)
comp(["Thiamine monophosphate", "Thiamine·PO4·Cl·2H2O"], "vitamin", [("thiamin-P", 1)], "C12H18N4O4PSCl*2H2O",
     "thiamine monophosphate -> thiamin-P (CHEBI:37574); " + ORG_CI)
comp(["Vitamin B12", "Vitamin B-12"], "vitamin", [("aquacob(III)alamin", 1)], "C63H88CoN14O14P",
     "AMBIGUOUS: media contain cyanocobalamin; Human-GEM has no cyanocobalamin exchange. Mapped to "
     "aquacob(III)alamin following Human-GEM's own Ham's growth task, which encodes vitamin B12 this way")
comp(["i-Inositol", "myo-Inositol"], "vitamin", [("inositol", 1)], "C6H12O6",
     "Human-GEM 'inositol' = myo-inositol (CHEBI:17268)")
comp(["Ascorbic Acid"], "vitamin", [("ascorbate", 1)], "C6H8O6")
comp(["Lipoic Acid", "DL-Thioctic Acid"], "vitamin", [("lipoic acid", 1)], "C8H14O2S2",
     "media lipoic acid is DL (racemic) thioctic acid; mapped to Human-GEM 'lipoic acid'")
# inorganic salts
comp(["Calcium Chloride (CaCl2) (anhyd.)", "CaCl2 (anhydrous)"], "inorganic salt", [("Ca2+", 1), ("chloride", 2)], "CaCl2")
comp(["CaCl2·2H2O"], "inorganic salt", [("Ca2+", 1), ("chloride", 2)], "CaCl2*2H2O")
comp(["Calcium nitrate (Ca(NO3)2 4H2O)", "Ca(NO3)2·4H2O"], "inorganic salt", [("Ca2+", 1), (U, 2, "nitrate")], "Ca(NO3)2*4H2O",
     "nitrate UNMAPPED: no nitrate exchange in Human-GEM v2.0.0")
comp(["Magnesium Sulfate (MgSO4) (anhyd.)", "MgSO4 (anhydrous)"], "inorganic salt", [("Mg2+", 1), ("sulfate", 1)], "MgSO4")
comp(["Magnesium Sulfate (MgSO4-7H2O)"], "inorganic salt", [("Mg2+", 1), ("sulfate", 1)], "MgSO4*7H2O")
comp(["Magnesium Chloride (anhydrous)"], "inorganic salt", [("Mg2+", 1), ("chloride", 2)], "MgCl2")
comp(["MgCl2·6H2O"], "inorganic salt", [("Mg2+", 1), ("chloride", 2)], "MgCl2*6H2O")
comp(["Potassium Chloride (KCl)", "KCl"], "inorganic salt", [("K+", 1), ("chloride", 1)], "KCl")
comp(["Potassium Nitrate (KNO3)"], "inorganic salt", [("K+", 1), (U, 1, "nitrate")], "KNO3",
     "nitrate UNMAPPED: no nitrate exchange in Human-GEM v2.0.0")
comp(["Potassium Phosphate monobasic (KH2PO4)", "KH2PO4 (anhydrous)"], "inorganic salt", [("K+", 1), ("Pi", 1)], "KH2PO4")
comp(["Sodium Bicarbonate (NaHCO3)", "NaHCO3"], "inorganic salt", [("Na+", 1), ("HCO3-", 1)], "NaHCO3")
comp(["Sodium Chloride (NaCl)", "NaCl"], "inorganic salt", [("Na+", 1), ("chloride", 1)], "NaCl")
comp(["Sodium Phosphate dibasic (Na2HPO4) anhydrous", "Na2HPO4 (anhydrous)"], "inorganic salt", [("Na+", 2), ("Pi", 1)], "Na2HPO4")
comp(["Sodium Phosphate monobasic (NaH2PO4) anhydrous"], "inorganic salt", [("Na+", 1), ("Pi", 1)], "NaH2PO4")
comp(["Sodium Phosphate monobasic (NaH2PO4-H2O)", "NaH2PO4·H2O", "NaH2PO4 ·H2O"], "inorganic salt", [("Na+", 1), ("Pi", 1)], "NaH2PO4*H2O")
comp(['Ferric Nitrate (Fe(NO3)3"9H2O)', "Fe(NO3)3·9H2O"], "inorganic salt", [("Fe3+", 1), (U, 3, "nitrate")], "Fe(NO3)3*9H2O",
     "nitrate UNMAPPED: no nitrate exchange in Human-GEM v2.0.0")
comp(["Ferrous sulfate (FeSO4-7H2O)", "Ferrous Sulfate (FeSO4 7H2O)", "FeSO4·7H2O"], "inorganic salt", [("Fe2+", 1), ("sulfate", 1)], "FeSO4*7H2O")
comp(["Cupric sulfate (CuSO4-5H2O)", "CuSO4·5H2O"], "inorganic salt", [("Cu2+", 1), ("sulfate", 1)], "CuSO4*5H2O")
comp(["CuSO4 (anhydrous)"], "inorganic salt", [("Cu2+", 1), ("sulfate", 1)], "CuSO4")
comp(["Zinc sulfate (ZnSO4-7H2O)", "ZnSO4·7H2O"], "inorganic salt", [("zinc", 1), ("sulfate", 1)], "ZnSO4*7H2O")
comp(["Sodium Selenite (Na2SeO3-5H20)"], "inorganic salt", [("Na+", 2), (U, 1, "selenite")], "Na2SeO3*5H2O",
     "selenite UNMAPPED: Human-GEM v2.0.0 has selenate (MAR09169) but no selenite exchange; not substituted")
# energy / other
comp(["D-Glucose (Dextrose)", "D-Glucose"], "carbon source", [("glucose", 1)], "C6H12O6")
comp(["D+ Galactose", "D-Galactose"], "carbon source", [("galactose", 1)], "C6H12O6")
comp(["Sodium Pyruvate", "Pyruvic Acid·Na"], "carbon source", [("pyruvate", 1)], "C3H3NaO3", ORG_CI)
comp(["Glutathione (reduced)"], "other", [("GSH", 1)], "C10H17N3O6S")
comp(["Hypoxanthine Na"], "other", [("hypoxanthine", 1)], "C5H3N4ONa", ORG_CI)
comp(["Hypoxanthine"], "other", [("hypoxanthine", 1)], "C5H4N4O",
     "ATCC prints 'Hypoxanthine' (form not stated); MW of free base used")
comp(["Linoleic Acid"], "lipid", [("linoleate", 1)], "C18H32O2")
comp(["Putrescine 2HCl", "Putrescine·2HCl"], "other", [("putrescine", 1)], "C4H12N2*2HCl", ORG_CI)
comp(["Thymidine"], "other", [("thymidine", 1)], "C10H14N2O5")
comp(["HEPES"], "buffer", [(U, 1, "HEPES")], "C8H18N2O4S",
     "UNMAPPED (intentional): non-metabolic zwitterionic buffer; no Human-GEM metabolite")
comp(["Phenol Red", "Phenol Red, Sodium Salt"], "pH indicator", [(U, 1, "phenol red")], "C19H13NaO5S",
     "UNMAPPED (intentional): pH indicator dye; no Human-GEM metabolite")
comp(["Bacto-Peptone"], "undefined", [(U, 1, "Bacto-peptone")], None,
     "UNMAPPED: undefined enzymatic protein digest (no MW, no defined composition); not encoded")
comp(["Transferrin (human)"], "protein", [(U, 1, "transferrin")], None,
     "UNMAPPED: protein; Human-GEM has no transferrin. Apo vs holo form not stated by DepMap/ATCC, "
     "so any iron delivered is unknown and is not encoded")


# ----------------------------------------------------------------------------
def load_model_index():
    import cobra
    cobra.Configuration().solver = "glpk"   # BEFORE reading: no Gurobi licence session
    m = cobra.io.read_sbml_model(str(ROOT / "data/scaffold/Human-GEM.xml"))
    ex = {}
    for r in m.exchanges:
        mt = list(r.metabolites)[0]
        ex.setdefault(mt.name, []).append((mt.id, r.id, mt.compartment, mt.annotation))
    return m, ex


def gibco_rows(tf, cat):
    d = tf[cat]
    out = []
    for sec, name, mwp, mg, mm in d["rows"]:
        out.append(dict(section=sec, name=name, mg=float(mg),
                        mw=(float(mwp) if mwp else None),
                        mM=(float(mm) if mm not in ("", "Infinity") else None),
                        mM_basis="verbatim (Gibco table)" if mm not in ("", "Infinity") else "no MW (undefined mixture)"))
    src = (f"Thermo Fisher Scientific, Gibco cat. {cat} ('{d['title']}') media formulation table, "
           f"{d['url']} (retrieved {RETRIEVED}; raw page archived, sha256 {d['sha256'][:12]}...)")
    return out, src


def atcc_rows(cat):
    rows = [r for r in csv.DictReader(open(SRC / "atcc_media/atcc_formulations_transcribed.tsv"), delimiter="\t")
            if r["atcc_catalog"] == cat]
    out = []
    for r in rows:
        name = r["component_as_printed"]
        info = C[name]
        mg = float(r["g_per_L"]) * 1000.0
        if info["formula"]:
            M = mw(info["formula"]); mM = mg / M
            basis = f"computed: mg/L / MW({info['formula']}={M:.2f})"
        else:
            M = None; mM = None; basis = "no MW"
        out.append(dict(section=r["section"], name=name, mg=mg, mw=M, mM=mM, mM_basis=basis,
                        row_note=ATCC_ROW_NOTE.get((cat, name), "")))
    url = [l.split("\t") for l in open(SRC / "manifest.tsv") if l.startswith(f"atcc_media/ATCC_{cat}_")][0][1]
    src = f"ATCC-formulated medium ATCC {cat}, formulation sheet {url} (retrieved {RETRIEVED}; PDF archived)"
    return out, src


DEPMAP_REF = "[ref: doi: 10.25452/figshare.plus.27993248.v1]"

# Row-level source ambiguities on ATCC sheets (salt form printed vs. value printed)
ATCC_ROW_NOTE = {
    ("30-2004", "L-Arginine (free base)"):
        "AMBIGUOUS FORM: ATCC prints 'free base' 0.42140 g/L; as free base = 2.419 mM, as Arg-HCl = 2.000 mM "
        "(Gibco F-12K 21127 lists L-arginine-HCl 422 mg/L = 2.0 mM). mM here follows the printed form",
    ("30-2006", "Hypoxanthine"):
        "AMBIGUOUS FORM: ATCC prints 'Hypoxanthine' 0.00239 g/L without form; as free base = 0.01756 mM, as Na salt = "
        "0.01503 mM (Gibco DMEM/F-12 11320/11330 lists hypoxanthine-Na 2.39 mg/L = 0.015 mM). mM here uses free base",
}


def supplement_rows():
    """Supplements exactly as stated in DepMap FormulationID strings."""
    tf = json.load(open(SRC / "thermofisher_gibco/parsed_tables.json"))
    S = OrderedDict()
    neaa, neaa_src = gibco_rows(tf, "11140")
    rows = []
    for r in neaa:  # Gibco 11140 is 100X; 1X = 100 uM each
        rows.append(dict(section="Amino Acids", name=r["name"], mg=r["mg"] / 100.0, mw=r["mw"], mM=r["mM"] / 100.0,
                         mM_basis="Gibco 100X table / 100 (1X = 0.1 mM each)"))
    S["SUPP_NEAA_1X_Gibco11140"] = (rows, neaa_src + "; used at 1X, matching DepMap '100uM NEAA' " + DEPMAP_REF,
                                    "supplement", "MEM Non-Essential Amino Acids (1X = 100 uM each)")
    g = C["L-Glutamine"]; M = mw(g["formula"])
    S["SUPP_L-glutamine_2mM"] = ([dict(section="Amino Acids", name="L-Glutamine", mg=2.0 * M, mw=M, mM=2.0,
                                      mM_basis=f"as stated (2 mM); mg/L computed with MW {M:.2f}")],
                                "DepMap 24Q4 FormulationID text '+ 2mM Glutamine' " + DEPMAP_REF,
                                "supplement", "L-glutamine 2 mM as stated by DepMap")
    h = mw(C["HEPES"]["formula"])
    S["SUPP_HEPES_25mM"] = ([dict(section="Other Components", name="HEPES", mg=25.0 * h, mw=h, mM=25.0,
                                 mM_basis=f"as stated (25 mM); mg/L computed with MW {h:.2f}")],
                           "DepMap 24Q4 FormulationID text '+ 25mM HEPES' " + DEPMAP_REF +
                           "; KCLB SNU-C4 datasheet https://cellbank.snu.ac.kr/cellline/search?mode=view&cell_id=498",
                           "supplement", "HEPES 25 mM as stated by DepMap/KCLB")
    b = mw(C["NaHCO3"]["formula"])
    S["SUPP_NaHCO3_25mM"] = ([dict(section="Inorganic Salts", name="NaHCO3", mg=25.0 * b, mw=b, mM=25.0,
                                  mM_basis=f"as stated (25 mM); mg/L computed with MW {b:.2f}")],
                            "DepMap 24Q4 FormulationID text '+ 25mM Sodium bicarbonate' " + DEPMAP_REF +
                            "; KCLB SNU-C4 datasheet https://cellbank.snu.ac.kr/cellline/search?mode=view&cell_id=498",
                            "supplement", "sodium bicarbonate 25 mM as stated by DepMap/KCLB")
    S["SUPP_transferrin_0.01mg_per_mL"] = ([dict(section="Other Components", name="Transferrin (human)", mg=10.0, mw=None,
                                                mM=None, mM_basis="mg/L as stated (0.01 mg/mL); MW not stated by source, mM not computed")],
                                          "DepMap 24Q4 FormulationID text '+ 0.01 mg/ml Transferrin' " + DEPMAP_REF +
                                          "; ATCC CRL-2102 (C2BBe1) page 'human transferrin 0.01 mg/ml' https://www.atcc.org/products/crl-2102",
                                          "supplement", "human transferrin 0.01 mg/mL (apo/holo not stated)")
    return S


def build_media():
    tf = json.load(open(SRC / "thermofisher_gibco/parsed_tables.json"))
    M = OrderedDict()
    spec = [  # key, family, source, catalog, variant description
        ("RPMI1640_Gibco11875", "RPMI-1640", "gibco", "11875", "classical RPMI-1640: 2.0 g/L glucose, 300 mg/L L-Gln, 2.0 g/L NaHCO3, no HEPES, no pyruvate"),
        ("RPMI1640_HEPES_Gibco22400", "RPMI-1640", "gibco", "22400", "RPMI-1640 with 25 mM HEPES, 300 mg/L L-Gln, 2.0 g/L NaHCO3"),
        ("RPMI1640_ATCC30-2001", "RPMI-1640", "atcc", "30-2001", "ATCC-modified RPMI-1640: 4.5 g/L glucose, 10 mM HEPES, 1 mM pyruvate, 1.5 g/L NaHCO3"),
        ("IMDM_Gibco12440", "IMDM", "gibco", "12440", "IMDM: 4.5 g/L glucose, 4 mM L-Gln, 25 mM HEPES, 1 mM pyruvate, selenite"),
        ("DMEM_highglc_Gibco11965", "DMEM", "gibco", "11965", "DMEM high glucose (4.5 g/L), 4 mM L-Gln, no pyruvate"),
        ("DMEM_highglc_pyr_Gibco11995", "DMEM", "gibco", "11995", "DMEM high glucose (4.5 g/L), 4 mM L-Gln, 1 mM pyruvate"),
        ("DMEM_lowglc_pyr_Gibco11885", "DMEM", "gibco", "11885", "DMEM low glucose (1.0 g/L), 4 mM L-Gln, 1 mM pyruvate"),
        ("DMEM_ATCC30-2002", "DMEM", "atcc", "30-2002", "ATCC DMEM: 4.5 g/L glucose, 4 mM L-Gln, 1 mM pyruvate, 1.5 g/L NaHCO3"),
        ("MEM_Earle_Gibco11095", "MEM", "gibco", "11095", "MEM (Eagle 1959), Earle's salts, 2 mM L-Gln, no NEAA, no pyruvate"),
        ("EMEM_ATCC30-2003", "EMEM", "atcc", "30-2003", "ATCC EMEM: Earle's BSS + NEAA + 1 mM pyruvate + 2 mM L-Gln, 1.5 g/L NaHCO3"),
        ("L15_Gibco11415", "Leibovitz L-15", "gibco", "11415", "Leibovitz L-15: galactose (no glucose), pyruvate, no NaHCO3"),
        ("L15_ATCC30-2008", "Leibovitz L-15", "atcc", "30-2008", "ATCC L-15"),
        ("McCoy5A_Gibco16600", "McCoy's 5A", "gibco", "16600", "McCoy's 5A (modified), 3.0 g/L glucose, Bacto-peptone"),
        ("McCoy5A_ATCC30-2007", "McCoy's 5A", "atcc", "30-2007", "ATCC McCoy's 5A (modified)"),
        ("DMEMF12_Gibco11320", "DMEM/F-12 (1:1)", "gibco", "11320", "DMEM/F-12 1:1, no HEPES, 2.5 mM L-Gln, 0.5 mM pyruvate"),
        ("DMEMF12_HEPES_Gibco11330", "DMEM/F-12 (1:1)", "gibco", "11330", "DMEM/F-12 1:1 with 15 mM HEPES"),
        ("DMEMF12_ATCC30-2006", "DMEM/F-12 (1:1)", "atcc", "30-2006", "ATCC DMEM:F-12 (15 mM HEPES, 0.5 mM pyruvate, 1.2 g/L NaHCO3)"),
        ("F12_Gibco11765", "Ham's F-12", "gibco", "11765", "Ham's F-12 Nutrient Mixture (Ham 1965)"),
        ("F12K_Gibco21127", "Ham's F-12K (Kaighn's)", "gibco", "21127", "Ham's F-12K (Kaighn's modification)"),
        ("F12K_ATCC30-2004", "Ham's F-12K (Kaighn's)", "atcc", "30-2004", "ATCC F-12K"),
    ]
    for key, fam, kind, cat, var in spec:
        rows, src = gibco_rows(tf, cat) if kind == "gibco" else atcc_rows(cat)
        M[key] = (rows, src, fam, var)
    for k, v in supplement_rows().items():
        M[k] = v
    return M


# ----------------------------------------------------------------------------
def main():
    model, ex = load_model_index()
    OUT.mkdir(parents=True, exist_ok=True)
    media = build_media()

    def resolve(name):
        hits = ex.get(name)
        if not hits:
            raise SystemExit(f"Human-GEM metabolite '{name}' has no exchange - dictionary error")
        hits = [h for h in hits if h[2] == "e"]
        assert len(hits) == 1, (name, hits)
        return hits[0]

    comp_rows = []
    unknown = set()
    for key, (rows, src, fam, var) in media.items():
        for r in rows:
            info = C.get(r["name"])
            if info is None:
                unknown.add(r["name"]); continue
            for sp in info["species"]:
                spname, stoich = sp[0], sp[1]
                base = dict(medium=key, medium_family=fam, medium_variant=var, component_name=r["name"],
                            component_class=info["cls"],
                            concentration_mg_per_L=f"{r['mg']:.6g}",
                            molar_mass_g_per_mol=(f"{r['mw']:.2f}" if r["mw"] else ""),
                            concentration_mM=(f"{r['mM']:.6g}" if r["mM"] is not None else ""),
                            mM_basis=r["mM_basis"], species_stoichiometry=stoich,
                            species_mM=(f"{r['mM'] * stoich:.6g}" if r["mM"] is not None else ""))
                if spname is None:
                    base.update(humangem_met_id="", humangem_met_name="", exchange_rxn_id="",
                                species=sp[2], mapping_status="unmapped",
                                mapping_note=info["note"] or "unmapped")
                else:
                    mid, rid, _, ann = resolve(spname)
                    chebi = ann.get("chebi", ""); mnx = ann.get("metanetx.chemical", "")
                    if isinstance(chebi, list): chebi = ";".join(chebi)
                    if isinstance(mnx, list): mnx = ";".join(mnx)
                    rn = r.get("row_note", "")
                    status = "ambiguous" if ("AMBIGUOUS" in info["note"] or "AMBIGUOUS" in rn) else "mapped"
                    note = "; ".join(x for x in [f"name match '{spname}' (ChEBI {chebi or 'NA'}, MetaNetX {mnx or 'NA'})", info["note"], rn] if x)
                    base.update(humangem_met_id=mid, humangem_met_name=spname, exchange_rxn_id=rid,
                                species=spname, mapping_status=status, mapping_note=note)
                base["source"] = src
                comp_rows.append(base)
    if unknown:
        raise SystemExit(f"components without dictionary entry: {sorted(unknown)}")

    cols = ["medium", "medium_family", "medium_variant", "component_name", "component_class",
            "concentration_mg_per_L", "molar_mass_g_per_mol", "concentration_mM", "mM_basis",
            "species", "species_stoichiometry", "species_mM", "humangem_met_id", "humangem_met_name",
            "exchange_rxn_id", "mapping_status", "mapping_note", "source"]
    with open(OUT / "media_components.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t"); w.writeheader(); w.writerows(comp_rows)

    # per-medium exchange totals (species summed across components)
    agg = defaultdict(lambda: dict(mM=0.0, n=0, comps=[]))
    for r in comp_rows:
        if r["exchange_rxn_id"] and r["species_mM"]:
            a = agg[(r["medium"], r["exchange_rxn_id"], r["humangem_met_id"], r["humangem_met_name"])]
            a["mM"] += float(r["species_mM"]); a["n"] += 1; a["comps"].append(r["component_name"])
    with open(OUT / "medium_exchange_mM.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["medium", "exchange_rxn_id", "humangem_met_id", "humangem_met_name", "total_species_mM",
                    "n_components", "components"])
        for (med, rid, mid, nm), a in agg.items():
            w.writerow([med, rid, mid, nm, f"{a['mM']:.6g}", a["n"], " | ".join(a["comps"])])

    # coverage summary
    cov = OrderedDict()
    for r in comp_rows:
        c = cov.setdefault(r["medium"], dict(components=set(), mapped=set(), ambiguous=set(), unmapped=set(), exch=set()))
        c["components"].add(r["component_name"])
        c[r["mapping_status"]].add(r["component_name"])
        if r["exchange_rxn_id"]: c["exch"].add(r["exchange_rxn_id"])
    with open(OUT / "medium_coverage.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["medium", "n_components", "n_components_fully_mapped", "n_components_with_ambiguous_mapping",
                    "n_components_with_unmapped_species", "n_distinct_exchange_rxns", "unmapped_components"])
        for k, c in cov.items():
            full = {x for x in c["components"] if x not in c["unmapped"] and x not in c["ambiguous"]}
            w.writerow([k, len(c["components"]), len(full), len(c["ambiguous"]), len(c["unmapped"]), len(c["exch"]),
                        "; ".join(sorted(c["unmapped"]))])

    # Human-GEM Ham's task vs Gibco Ham's F-12
    ham = ham_task_inputs()
    f12 = {r["humangem_met_name"] for r in comp_rows if r["medium"] == "F12_Gibco11765" and r["humangem_met_name"]}
    f12_unm = sorted({r["species"] for r in comp_rows if r["medium"] == "F12_Gibco11765" and not r["humangem_met_name"]})
    with open(OUT / "hams_task_vs_F12.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["humangem_met_name", "exchange_rxn_id", "in_HumanGEM_Hams_task", "in_Gibco_11765_F12", "category"])
        for nm in sorted(ham | f12):
            rid = resolve(nm)[1] if nm in ex else ""
            cat = ("both" if (nm in ham and nm in f12) else
                   "task only (not in F-12 formulation)" if nm in ham else "F-12 only (absent from task)")
            w.writerow([nm, rid, int(nm in ham), int(nm in f12), cat])
        for s in f12_unm:
            w.writerow([s, "", 0, 1, "F-12 component without Human-GEM exchange"])

    # cross-check: Gibco A10491 (RPMI 'ATCC modification') vs ATCC 30-2001 sheet
    a = {r["name"]: r for r in gibco_rows(json.load(open(SRC / "thermofisher_gibco/parsed_tables.json")), "A10491")[0]}
    b = {r["name"]: r for r in atcc_rows("30-2001")[0]}
    def sp_mM(rows):
        d = defaultdict(float)
        for n, r in rows.items():
            for s in C[n]["species"]:
                if r["mM"] is not None: d[s[0] or s[2]] += r["mM"] * s[1]
        return d
    da, db = sp_mM(a), sp_mM(b)
    with open(OUT / "crosscheck_RPMI_A10491_vs_ATCC30-2001.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t"); w.writerow(["species", "Gibco_A10491_mM", "ATCC_30-2001_mM", "rel_diff"])
        for s in sorted(set(da) | set(db)):
            x, y = da.get(s, 0.0), db.get(s, 0.0)
            w.writerow([s, f"{x:.6g}", f"{y:.6g}", f"{abs(x - y) / max(x, y, 1e-12):.3f}"])

    lm = build_line_map(media)
    assemble(comp_rows, lm)
    print("components rows:", len(comp_rows), "media:", len(media))


def assemble(comp_rows, lm):
    """Base medium + stated supplements -> exchange-level mM, under two rules.

    final_concentration (DEFAULT, pending user ruling): a supplement replaces every
      base component that supplies the same species set (e.g. 25 mM NaHCO3 replaces
      the base NaHCO3 row; 2 mM L-Gln replaces the base L-Gln row; NEAA 0.1 mM
      replaces each base Gly/Ala/Asn/Asp/Glu/Pro/Ser row).
    additive: supplement rows are added on top of the base.
    Serum is NOT included (no defined FBS composition; README section 'FBS').
    """
    by_med = defaultdict(list)
    for r in comp_rows:
        by_med[r["medium"]].append(r)

    def comp_key(rows):  # species set supplied by one component (within a medium)
        d = defaultdict(set)
        for r in rows:
            d[r["component_name"]].add(r["species"])
        return d

    out = []
    for aid, (base, supps) in sorted({(row["assembly_id"]): (row["default_medium"], row["supplements"])
                                      for row in lm}.items()):
        supps = [s for s in supps.split(" + ") if s]
        for rule in ("final_concentration", "additive"):
            rows = list(by_med[base])
            for s in supps:
                srows = by_med[s]
                if rule == "final_concentration":
                    s_species = [frozenset(v) for v in comp_key(srows).values()]
                    bk = comp_key(rows)
                    drop = {c for c, sp in bk.items() if frozenset(sp) in s_species}
                    rows = [r for r in rows if r["component_name"] not in drop]
                rows = rows + srows
            tot = defaultdict(lambda: [0.0, []])
            for r in rows:
                if r["exchange_rxn_id"] and r["species_mM"]:
                    k = (r["exchange_rxn_id"], r["humangem_met_id"], r["humangem_met_name"])
                    tot[k][0] += float(r["species_mM"]); tot[k][1].append(f"{r['medium']}:{r['component_name']}")
            for (rid, mid, nm), (v, cs) in sorted(tot.items(), key=lambda x: x[0][2]):
                out.append([aid, rule, base, " + ".join(supps), rid, mid, nm, f"{v:.6g}", " | ".join(cs)])
    with open(OUT / "assembled_media_exchange_mM.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["assembly_id", "supplement_rule", "base_medium_key", "supplements", "exchange_rxn_id",
                    "humangem_met_id", "humangem_met_name", "total_species_mM", "contributing_components"])
        w.writerows(out)


def ham_task_inputs():
    names = []; started = False
    for line in open(ROOT / "data/scaffold/metabolicTasks_Essential.txt"):
        p = line.rstrip("\n").split("\t")
        if not p or p[0].startswith("#"): continue
        p += [""] * 10
        if p[1] == "GR": started = True
        elif started and p[1]: break
        if started and p[4].strip().endswith("[e]"):
            names.append(p[4].strip()[:-3])
    assert len(names) == 44, len(names)
    return set(names)


# ----------------------------------------------------------------------------
# FormulationID -> base medium family, supplements as stated, serum %, default
# medium key and alternatives. Defaults are pending user ruling (README D1-D9).
FORM = {
    "RPMI + 10% FBS": ("RPMI-1640", [], 10, "RPMI1640_Gibco11875", ["RPMI1640_ATCC30-2001"]),
    "RPMI + 20% FBS": ("RPMI-1640", [], 20, "RPMI1640_Gibco11875", ["RPMI1640_ATCC30-2001"]),
    "RPMI + 10% FBS + 2mM Glutamine + 25mM HEPES + 25mM Sodium bicarbonate":
        ("RPMI-1640", ["SUPP_L-glutamine_2mM", "SUPP_HEPES_25mM", "SUPP_NaHCO3_25mM"], 10,
         "RPMI1640_HEPES_Gibco22400", ["RPMI1640_Gibco11875"]),
    "IMDM + 10% FBS + 2mM Glutamine": ("IMDM", ["SUPP_L-glutamine_2mM"], 10, "IMDM_Gibco12440", []),
    "DMEM + 10% FBS": ("DMEM", [], 10, "DMEM_highglc_Gibco11965", ["DMEM_highglc_pyr_Gibco11995", "DMEM_lowglc_pyr_Gibco11885", "DMEM_ATCC30-2002"]),
    "DMEM + 10% FBS + 2mM Glutamine": ("DMEM", ["SUPP_L-glutamine_2mM"], 10, "DMEM_highglc_Gibco11965", ["DMEM_highglc_pyr_Gibco11995", "DMEM_lowglc_pyr_Gibco11885"]),
    "DMEM + 10% FBS + 0.01 mg/ml Transferrin": ("DMEM", ["SUPP_transferrin_0.01mg_per_mL"], 10, "DMEM_highglc_Gibco11965", ["DMEM_ATCC30-2002", "DMEM_highglc_pyr_Gibco11995"]),
    "MEM + 10% FBS": ("MEM", [], 10, "MEM_Earle_Gibco11095", ["EMEM_ATCC30-2003"]),
    "MEM + 20% FBS + 2mM Glutamine + 100uM NEAA": ("MEM", ["SUPP_L-glutamine_2mM", "SUPP_NEAA_1X_Gibco11140"], 20, "MEM_Earle_Gibco11095", ["EMEM_ATCC30-2003"]),
    "EMEM + 10% FBS": ("EMEM", [], 10, "EMEM_ATCC30-2003", ["MEM_Earle_Gibco11095"]),
    "EMEM + 10% FBS + 2mM Glutamine + 100uM NEAA": ("EMEM", ["SUPP_L-glutamine_2mM", "SUPP_NEAA_1X_Gibco11140"], 10, "EMEM_ATCC30-2003", ["MEM_Earle_Gibco11095"]),
    "EMEM + 5% FBS + 2mM Glutamine": ("EMEM", ["SUPP_L-glutamine_2mM"], 5, "EMEM_ATCC30-2003", ["MEM_Earle_Gibco11095"]),
    "L-15 + 10% FBS": ("Leibovitz L-15", [], 10, "L15_Gibco11415", ["L15_ATCC30-2008"]),
    "McCoy's 5A + 10% FBS": ("McCoy's 5A", [], 10, "McCoy5A_Gibco16600", ["McCoy5A_ATCC30-2007"]),
    "DMEM:F12 + 20% FBS": ("DMEM/F-12 (1:1)", [], 20, "DMEMF12_Gibco11320", ["DMEMF12_HEPES_Gibco11330", "DMEMF12_ATCC30-2006"]),
    "DMEM:F12 + 5% FBS": ("DMEM/F-12 (1:1)", [], 5, "DMEMF12_Gibco11320", ["DMEMF12_ATCC30-2006", "DMEMF12_HEPES_Gibco11330"]),
    "F12 + 10% FBS": ("Ham's F-12", [], 10, "F12_Gibco11765", ["F12K_ATCC30-2004", "F12K_Gibco21127"]),
}
GDSC = {"R": ("RPMI-1640", "RPMI1640_Gibco11875", ["RPMI1640_ATCC30-2001"]),
        "D/F12": ("DMEM/F-12 (1:1)", "DMEMF12_Gibco11320", ["DMEMF12_HEPES_Gibco11330"])}


def build_line_map(media):
    import pandas as pd
    t = pd.read_csv(ROOT / "data/depmap/crc_trainset/crc_trainset_models.tsv", sep="\t")
    model = pd.read_csv(ROOT / "data/depmap/Model.csv", low_memory=False).set_index("ModelID")
    mc = pd.read_csv(SRC / "depmap_24Q4/ModelCondition.csv").set_index("ModelConditionID")
    sm = pd.read_csv(SRC / "depmap_24Q4/ScreenSequenceMap.csv")
    gd = pd.read_excel(SRC / "gdsc/GDSC_Cell_Lines_Details.xlsx", sheet_name=0) if _has_openpyxl() else None
    gdsc = {}
    if gd is not None:
        gd.columns = [str(c).replace("\n", " ") for c in gd.columns]
        gdsc = dict(zip(gd["COSMIC identifier"], gd["Screen Medium"]))
    else:
        gdsc = {float(k): v for k, v in csv.reader(open(SRC / "gdsc/GDSC_screen_medium_all_lines.tsv"), delimiter="\t")}
    s = sm[sm.ModelID.isin(t.ModelID)]
    rows = []
    for _, L in t.iterrows():
        mid = L.ModelID; M = model.loc[mid]
        ss = s[s.ModelID == mid]
        def screen_summary(lib_prefix):
            q = ss[ss.ScreenID.str.contains(lib_prefix)]
            if q.empty: return None
            sid = q.ScreenID.iloc[0]; cond = q.ModelConditionID.iloc[0]
            used = bool(((~q.ExcludeFromCRISPRCombined.astype(bool)) & q.PassesQC.astype(bool)).any())
            c = mc.loc[cond]
            return dict(sid=sid, cond=cond, used=used, src=c.DataSource,
                        mf=(c.GrowthMedia if isinstance(c.GrowthMedia, str) else ""),
                        form=(c.FormulationID if isinstance(c.FormulationID, str) else ""))
        broad = screen_summary(r"\.(?:AV|CD)")
        sanger = screen_summary(r"\.KY")
        gmed = gdsc.get(float(M.COSMICID)) if pd.notna(M.COSMICID) else None
        notes = []
        if broad and broad["used"] and broad["form"]:
            basis = "Broad screen condition (DepMap ModelCondition)"
            fam, sup, serum, dflt, alts = FORM[broad["form"]]
            sup_text = broad["form"]
            if broad["form"] != (M.FormulationID if isinstance(M.FormulationID, str) else ""):
                notes.append(f"screen-condition medium differs from onboarding medium ('{M.FormulationID}')")
        elif sanger and sanger["used"] and isinstance(gmed, str):
            basis = "Sanger screen only (GDSC 'Screen Medium'; Sanger serum % not verified)"
            fam, dflt, alts = GDSC[gmed]; sup = []; serum = ""; sup_text = f"GDSC Screen Medium '{gmed}'"
            if broad and not broad["used"]:
                notes.append(f"Broad screen {broad['sid']} failed QC (no passing sequences); gene effect from Sanger screen only")
        else:
            raise SystemExit(f"no medium basis for {mid}")
        if broad and broad["used"] and sanger and sanger["used"]:
            notes.append(f"gene effect integrates Broad and Sanger screens; Sanger screen medium (GDSC) = '{gmed}'")
        sup_note = ""
        if any(x in sup for x in ["SUPP_L-glutamine_2mM"]):
            sup_note = ("default rule: stated supplement concentration = FINAL concentration in complete medium "
                        "(replaces the base value), pending user ruling D3")
        rows.append(OrderedDict(
            ModelID=mid, CellLineName=L.CellLineName,
            OnboardedMedia=(M.OnboardedMedia if isinstance(M.OnboardedMedia, str) else ""),
            FormulationID=(M.FormulationID if isinstance(M.FormulationID, str) else ""),
            base_medium=fam, supplements=" + ".join(sup), serum_percent=serum,
            medium_basis=basis, medium_basis_text=sup_text,
            broad_screen=(broad["sid"] if broad else ""), broad_screen_used=(broad["used"] if broad else ""),
            broad_screen_MF=(broad["mf"] if broad else ""), broad_screen_FormulationID=(broad["form"] if broad else ""),
            sanger_screen=(sanger["sid"] if sanger else ""), sanger_screen_used=(sanger["used"] if sanger else ""),
            gdsc_screen_medium=(gmed or ""),
            default_medium=dflt, alternative_media=" | ".join(alts),
            assembly_id=dflt + "".join("+" + x for x in sup),
            supplement_rule=sup_note, notes=" ; ".join(notes)))
    pd.DataFrame(rows).to_csv(OUT / "line_medium_map.tsv", sep="\t", index=False)
    return rows


def _has_openpyxl():
    try:
        import openpyxl  # noqa
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    main()
