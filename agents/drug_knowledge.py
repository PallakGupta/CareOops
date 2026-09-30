"""
Small, hand-curated drug knowledge base used by the Prescription Safety Agent.

This is deliberately tiny and readable. It covers the drugs in the demo
scenarios plus common primary-care drugs. It is:
  1. the fallback when the RxNorm interaction lookup is unavailable, and
  2. the source of drug-class and renal-dosing rules, which RxNorm itself
     does not provide.

Note: NLM retired the public RxNav Drug Interaction API in January 2024, so the
interaction call in prescription_safety.py now always lands here. The RxNorm
name-to-RxCUI lookup still works and is used live.

Future work: replace with a licensed interaction database (e.g. DrugBank,
First Databank, Lexicomp) and full renal/hepatic dosing tables.
"""

# Brand names and abbreviations mapped to one canonical generic name.
DRUG_ALIASES = {
    "advil": "ibuprofen", "motrin": "ibuprofen", "brufen": "ibuprofen",
    "aleve": "naproxen",
    "tylenol": "acetaminophen", "paracetamol": "acetaminophen", "crocin": "acetaminophen",
    "bactrim": "trimethoprim-sulfamethoxazole", "septra": "trimethoprim-sulfamethoxazole",
    "tmp-smx": "trimethoprim-sulfamethoxazole", "co-trimoxazole": "trimethoprim-sulfamethoxazole",
    "cotrimoxazole": "trimethoprim-sulfamethoxazole", "sulfamethoxazole-trimethoprim": "trimethoprim-sulfamethoxazole",
    "macrobid": "nitrofurantoin", "macrodantin": "nitrofurantoin",
    "augmentin": "amoxicillin-clavulanate", "amoxicillin/clavulanate": "amoxicillin-clavulanate",
    "coumadin": "warfarin", "zestril": "lisinopril", "glucophage": "metformin",
    "cipro": "ciprofloxacin", "zithromax": "azithromycin", "keflex": "cephalexin",
}

# Generic name -> drug classes. Classes drive the allergy and interaction checks.
DRUG_CLASSES = {
    "ibuprofen": ["nsaid"],
    "naproxen": ["nsaid"],
    "diclofenac": ["nsaid"],
    "aspirin": ["nsaid", "antiplatelet"],
    "acetaminophen": ["analgesic"],
    "codeine": ["opioid"],
    "tramadol": ["opioid"],
    "amoxicillin": ["penicillin", "beta_lactam"],
    "amoxicillin-clavulanate": ["penicillin", "beta_lactam"],
    "cephalexin": ["cephalosporin", "beta_lactam"],
    "azithromycin": ["macrolide"],
    "ciprofloxacin": ["fluoroquinolone"],
    "trimethoprim-sulfamethoxazole": ["sulfonamide", "trimethoprim"],
    "nitrofurantoin": ["nitrofuran"],
    "lisinopril": ["ace_inhibitor"],
    "enalapril": ["ace_inhibitor"],
    "ramipril": ["ace_inhibitor"],
    "losartan": ["arb"],
    "spironolactone": ["potassium_sparing_diuretic"],
    "furosemide": ["loop_diuretic"],
    "metformin": ["biguanide"],
    "warfarin": ["anticoagulant"],
    "apixaban": ["anticoagulant"],
    "metoprolol": ["beta_blocker"],
    "atorvastatin": ["statin"],
    "cetirizine": ["antihistamine"],
    "omeprazole": ["ppi"],
}

# Allergy class -> other classes with known cross-reactivity (reported as caution).
CROSS_REACTIVITY = {
    "penicillin": ["cephalosporin"],
}

# Renal (eGFR) rules: flag when the patient's latest eGFR is below the threshold.
RENAL_RULES = [
    {"applies_to": "nsaid", "egfr_below": 60, "severity": "high",
     "message": "NSAIDs can cause acute kidney injury in CKD. Avoid when eGFR < 60."},
    {"applies_to": "metformin", "egfr_below": 30, "severity": "high",
     "message": "Metformin is contraindicated when eGFR < 30 (lactic acidosis risk)."},
    {"applies_to": "nitrofurantoin", "egfr_below": 30, "severity": "high",
     "message": "Nitrofurantoin is ineffective and toxic when eGFR < 30."},
    {"applies_to": "trimethoprim-sulfamethoxazole", "egfr_below": 30, "severity": "medium",
     "message": "TMP-SMX needs dose reduction when eGFR < 30 (hyperkalemia risk)."},
]

# Offline interaction table. Keys are pairs of drug names OR drug classes.
LOCAL_INTERACTIONS = {
    frozenset({"nsaid", "ace_inhibitor"}): (
        "high",
        "NSAID + ACE inhibitor: lowers the blood-pressure effect and raises the risk of "
        "acute kidney injury and hyperkalemia, especially in CKD."),
    frozenset({"nsaid", "arb"}): (
        "high",
        "NSAID + ARB: raises the risk of acute kidney injury and hyperkalemia."),
    frozenset({"nsaid", "anticoagulant"}): (
        "high",
        "NSAID + anticoagulant: significantly raises the risk of serious bleeding, including GI bleeding."),
    frozenset({"sulfonamide", "anticoagulant"}): (
        "high",
        "Sulfamethoxazole inhibits warfarin metabolism (CYP2C9). This markedly raises the INR and bleeding risk."),
    frozenset({"fluoroquinolone", "anticoagulant"}): (
        "medium",
        "Fluoroquinolone + warfarin: may raise the INR. Monitor closely."),
    frozenset({"trimethoprim", "ace_inhibitor"}): (
        "medium",
        "Trimethoprim + ACE inhibitor: additive hyperkalemia risk."),
    frozenset({"ace_inhibitor", "potassium_sparing_diuretic"}): (
        "high",
        "ACE inhibitor + potassium-sparing diuretic: risk of severe hyperkalemia."),
    frozenset({"nsaid", "loop_diuretic"}): (
        "medium",
        "NSAID + loop diuretic: reduces the diuretic effect and raises renal risk."),
}


def normalize_drug(name: str) -> str:
    """Lowercase, trim and map brand names/abbreviations to the canonical generic name."""
    key = (name or "").strip().lower()
    return DRUG_ALIASES.get(key, key)


def classes_for(drug: str) -> list[str]:
    """Return the drug classes for a canonical drug name (empty list if unknown)."""
    return DRUG_CLASSES.get(drug, [])


def local_interactions(drug_a: str, drug_b: str) -> list[dict]:
    """Look up interactions between two drugs in the offline table.

    Matches on the drug names themselves and on every pair of their classes.
    That means "ibuprofen + lisinopril" is caught by the (nsaid, ace_inhibitor) rule.
    """
    terms_a = {drug_a, *classes_for(drug_a)}
    terms_b = {drug_b, *classes_for(drug_b)}
    found = []
    for a in terms_a:
        for b in terms_b:
            hit = LOCAL_INTERACTIONS.get(frozenset({a, b}))
            if hit and a != b:
                found.append({"severity": hit[0], "description": hit[1]})
    unique = {f["description"]: f for f in found}
    return list(unique.values())
