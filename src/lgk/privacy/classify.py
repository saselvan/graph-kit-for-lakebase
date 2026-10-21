"""Column classification for the privacy gate.

Every column is one of three classes: care subject, public professional, or none. A column is
judged by its NAME and by its sampled VALUES. Values never leave the profiler: they are
reduced to pattern verdicts (the share of values matching each pattern), and the
classifier reads only those verdicts.

Doubt plays safe: a care-subject pattern that matches only part of the values ("unsure")
still makes the column care subject, and care-subject evidence always beats professional.
"""

import re
from typing import Dict, Iterable, List, Optional, Tuple

CARE_SUBJECT = "care_subject"
PUBLIC_PROFESSIONAL = "public_professional"
NONE = "none"
CLASSES = (CARE_SUBJECT, PUBLIC_PROFESSIONAL, NONE)

# A pattern matching at least SURE of the values is certain; at least UNSURE is a doubt,
# and a doubt about a care-subject pattern is treated as certain.
SURE = 0.8
UNSURE = 0.2

# Care-subject words. The spec has three classes only, so direct identifiers (MRN, SSN, person
# and encounter ids, names) and contact / quasi-identifiers (email, phone, address, zip, birth
# date) are all care subject: doubt plays safe. Multi-word identifiers are written joined
# ("medicalrecord"); a name matches when a run of its tokens joins to a care word.
#
# Four ways a word is a care word, each a separate set so each can be proven on its own:
# 1a. a care SUBJECT word, alone or glued to an id affix ("memberid", "patientname", "ptkey"),
#     but only where the column is about one such person, not in an attribute shape (below);
_CARE_SUBJECT_TOKENS = {
    # the care subject and the people tied to them
    "patient", "patients", "pt", "pat", "member", "mbr", "beneficiary", "bene", "enrollee",
    "subject", "guarantor", "insured", "claimant", "subscriber", "guardian", "nok",
    "nextofkin", "emergencycontact",
}
# 1b. a care IDENTIFIER token, alone or glued to an id affix, in any shape ("mrn_count" too);
_CARE_NAME_TOKENS = {
    # direct identifiers
    "mrn", "medicalrecord", "ssn", "socialsecurity", "mbi", "hicn", "empi",
    # names
    "fname", "lname",
    # birth date
    "dob", "dateofbirth",
    # contact and quasi-identifiers
    "mobile", "telephone", "fax", "street", "zip", "postal",
}
_ID_AFFIXES = {
    "id", "ids", "no", "num", "nbr", "number", "key", "name", "nm", "code", "cd", "ref", "dt",
    "date",
}
# 2. an id stem, only with an id suffix: "visit_id", "person_key", "account_number", but not
#    "visit_type" or "plan_type" (these words alone are not about one person);
_CARE_ID_STEMS = {
    "person", "encounter", "visit", "dependent", "account", "plan", "medicare", "medicaid",
}
_ID_SUFFIXES = {"id", "ids", "key", "nbr", "num", "no", "number", "name", "nm"}
# 3. a care prefix, glued to anything: "emailaddr", "birthplace", "deceased_dt", "phonenumber";
_CARE_PREFIXES = {"email", "addr", "phone", "birth", "death", "deceased"}
# 4. a name part glued to a name suffix: "surname", "given_name", "first_nm".
_NAME_PARTS = {"first", "last", "middle", "full", "given", "sur", "maiden", "family"}
_NAME_SUFFIXES = {"name", "nm"}
# Attribute shapes (A3): a subject word describes a population or a figure, not one person.
# - the last token is a figure or a flag (e.g., "patient_count", "patients_seen_count", "new_patient_ok");
# - a PLURAL subject word ends the name after another word (e.g., "accepting_new_patients").
# A singular at the end ("referring_patient") or a qualifier after it ("patient_city") still
# names one person, so it stays care subject: doubt plays safe.
_ATTRIBUTE_SUFFIXES = {
    "count", "cnt", "total", "sum", "avg", "mean", "pct", "percent", "rate", "ratio", "size",
    "ok", "flag", "flg", "ind", "yn", "allowed",
}
_PROFESSIONAL_NAME_TOKENS = {
    "npi", "provider", "physician", "doctor", "clinician", "practitioner", "prescriber",
    "nurse", "surgeon",
}

_MRN = re.compile(r"(?i)^(?:mrn|mr|pt|pat|med)[-_ ]?\d{5,12}$")
_SSN = re.compile(r"^\d{3}-\d{2}-\d{4}$")
_NPI = re.compile(r"^\d{10}$")
_EMAIL = re.compile(r"^[A-Za-z0-9._%+'-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}$")
# A phone needs a separator or brackets: a bare 10-digit string is an NPI's shape, not a phone.
_PHONE = re.compile(r"^(?:\+?1[-. ]?)?(?:\(\d{3}\)[-. ]?|\d{3}[-. ])\d{3}[-. ]\d{4}$")
_ZIP = re.compile(r"^\d{5}(?:-\d{4})?$")
_DATE = re.compile(
    r"^(?:\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?"
    r"|\d{1,2}/\d{1,2}/(?:\d{2}|\d{4}))$"
)


def _npi_valid(value: str) -> bool:
    """CMS NPI check digit: Luhn over the 9 base digits with the 80840 prefix."""
    if not _NPI.match(value):
        return False
    digits = [int(d) for d in "80840" + value[:9]]
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 0:
            d *= 2
            d = d - 9 if d > 9 else d
        total += d
    return (10 - total % 10) % 10 == int(value[9])


# pattern name -> (matcher, class it is evidence for)
PATTERNS = {
    "mrn": (lambda v: bool(_MRN.match(v)), CARE_SUBJECT),
    "ssn": (lambda v: bool(_SSN.match(v)), CARE_SUBJECT),
    "email": (lambda v: bool(_EMAIL.match(v)), CARE_SUBJECT),
    "phone": (lambda v: bool(_PHONE.match(v)), CARE_SUBJECT),
    "zip": (lambda v: bool(_ZIP.match(v)), CARE_SUBJECT),
    "date": (lambda v: bool(_DATE.match(v)), CARE_SUBJECT),
    "npi": (_npi_valid, PUBLIC_PROFESSIONAL),
}


def pattern_verdicts(values: Iterable[object]) -> Dict[str, float]:
    """Share of the non-null sampled values matching each pattern. Holds rates, never values.

    NULLs and blank strings are not values: a mostly-NULL MRN column is judged on the values
    it has, so the NULLs cannot dilute its match rate below the doubt threshold.
    """
    sample = [s for s in (str(v).strip() for v in values if v is not None) if s]
    if not sample:
        return {name: 0.0 for name in PATTERNS}
    return {
        name: float(sum(1 for v in sample if match(v))) / len(sample)
        for name, (match, _cls) in PATTERNS.items()
    }


def name_tokens(name: str) -> List[str]:
    """Lower-case tokens of a column name: split on separators, camelCase and digits.

    "PatientID" -> [patient, id]; "mbrId" -> [mbr, id]; "zip5" -> [zip, 5].
    """
    out: List[str] = []
    for part in re.split(r"[^A-Za-z0-9]+", name):
        out += [t.lower() for t in re.findall(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+", part)]
    return out


def _token_or_affixed(word: str, tokens: set) -> bool:
    """`word` is one of `tokens`, alone or glued to an id affix ("patientid", "idpatient")."""
    if word in tokens:
        return True
    return any(
        (word.endswith(a) and word[: -len(a)] in tokens)
        or (word.startswith(a) and word[len(a):] in tokens)
        for a in _ID_AFFIXES
    )


def _glued(word: str, heads: set, tails: set) -> bool:
    """`word` is one of `heads` followed by one of `tails`."""
    return any(word.endswith(t) and word[: -len(t)] in heads for t in tails)


def _is_identifier_word(word: str) -> bool:
    """A care word in any shape (rules 1b-4)."""
    return (
        _token_or_affixed(word, _CARE_NAME_TOKENS)
        or _glued(word, _CARE_ID_STEMS, _ID_SUFFIXES)
        or any(word.startswith(p) for p in _CARE_PREFIXES)
        or _glued(word, _NAME_PARTS, _NAME_SUFFIXES)
    )


def _attribute_shaped(tokens: List[str], i: int, j: int) -> bool:
    """The subject word at tokens[i:j] describes a population or a figure, not one person."""
    if j < len(tokens) and tokens[-1] in _ATTRIBUTE_SUFFIXES:
        return True
    word = "".join(tokens[i:j])
    return j == len(tokens) and i > 0 and word in _CARE_SUBJECT_TOKENS and word.endswith("s")


def _has_care_name(tokens: List[str]) -> bool:
    """A contiguous run of tokens, joined, is a care word (alone or glued to an id affix);
    a care subject word counts only outside an attribute shape."""
    for i in range(len(tokens)):
        for j in range(i + 1, len(tokens) + 1):
            word = "".join(tokens[i:j])
            if _is_identifier_word(word):
                return True
            if _token_or_affixed(word, _CARE_SUBJECT_TOKENS) and not _attribute_shaped(tokens, i, j):
                return True
    return False


def _name_class(name: str) -> Optional[str]:
    tokens = name_tokens(name)
    if _has_care_name(tokens):
        return CARE_SUBJECT
    if set(tokens) & _PROFESSIONAL_NAME_TOKENS:
        return PUBLIC_PROFESSIONAL
    return None


def _values_class(patterns: Dict[str, float]) -> Optional[str]:
    care = [rate for name, rate in patterns.items() if PATTERNS.get(name, (0, None))[1] == CARE_SUBJECT]
    if any(rate >= UNSURE for rate in care):
        return CARE_SUBJECT
    professional = [r for n, r in patterns.items() if PATTERNS.get(n, (0, None))[1] == PUBLIC_PROFESSIONAL]
    if any(rate >= SURE for rate in professional):
        return PUBLIC_PROFESSIONAL
    return None


def classify_column(name: str, patterns: Dict[str, float]) -> Tuple[str, Optional[str]]:
    """(class, detected_by) for one column; detected_by is "name", "values", "both" or None."""
    by_name, by_values = _name_class(name), _values_class(patterns or {})
    for cls in (CARE_SUBJECT, PUBLIC_PROFESSIONAL):  # care-subject evidence wins
        hits = [src for src, c in (("name", by_name), ("values", by_values)) if c == cls]
        if hits:
            return cls, "both" if len(hits) == 2 else hits[0]
    return NONE, None
