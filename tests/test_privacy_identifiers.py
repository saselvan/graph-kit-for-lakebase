"""Direct identifiers and contact data classify as care subject.

By NAME: every listed name is care subject; each member of each care word set has a
hostile name that ONLY that member catches, and removing the member (monkeypatched out of
its set) turns that name back to `none` - so every stem is proven to be load-bearing.
By VALUES (rates only): email, phone, zip5/zip9 and date shapes are care-subject
evidence, so a renamed column ("contact") is still caught by what it holds.
"""

import pytest

from lgk.privacy import classify

ISSUE_NAMES = [
    "surname", "given_name", "first_nm", "telephone", "fax", "emailaddr", "birthplace_country",
    "encounter_key", "visit_id", "person_key", "claimant_id", "subscriber_id", "dependent_id",
    "guardian_name", "nok_name", "emergency_contact", "medicare_id", "medicaid_id", "mbi",
    "hicn", "empi_id", "account_number", "health_plan_id", "deceased_dt", "death_date",
]

# (set name in classify, member) -> a real-world name only that member catches
HOSTILE = {
    # care subject tokens (alone or with an id affix, outside an attribute shape)
    ("_CARE_SUBJECT_TOKENS", "patient"): "PatientID", ("_CARE_SUBJECT_TOKENS", "patients"): "patients",
    ("_CARE_SUBJECT_TOKENS", "pt"): "pt_key", ("_CARE_SUBJECT_TOKENS", "pat"): "PatId",
    ("_CARE_SUBJECT_TOKENS", "member"): "memberid", ("_CARE_SUBJECT_TOKENS", "mbr"): "mbrId",
    ("_CARE_SUBJECT_TOKENS", "beneficiary"): "beneficiary_no", ("_CARE_SUBJECT_TOKENS", "bene"): "bene_id",
    ("_CARE_SUBJECT_TOKENS", "enrollee"): "enrolleeNum", ("_CARE_SUBJECT_TOKENS", "subject"): "subject_id",
    ("_CARE_SUBJECT_TOKENS", "guarantor"): "guarantor_name", ("_CARE_SUBJECT_TOKENS", "insured"): "insured_id",
    ("_CARE_SUBJECT_TOKENS", "claimant"): "claimant_id", ("_CARE_SUBJECT_TOKENS", "subscriber"): "subscriber_id",
    ("_CARE_SUBJECT_TOKENS", "guardian"): "guardian_name", ("_CARE_SUBJECT_TOKENS", "nok"): "nok_name",
    ("_CARE_SUBJECT_TOKENS", "nextofkin"): "next_of_kin",
    ("_CARE_SUBJECT_TOKENS", "emergencycontact"): "emergency_contact",
    ("_CARE_NAME_TOKENS", "mrn"): "MRN", ("_CARE_NAME_TOKENS", "medicalrecord"): "medical_record_number",
    ("_CARE_NAME_TOKENS", "ssn"): "ssn", ("_CARE_NAME_TOKENS", "socialsecurity"): "SocialSecurityNo",
    ("_CARE_NAME_TOKENS", "mbi"): "mbi", ("_CARE_NAME_TOKENS", "hicn"): "hicn",
    ("_CARE_NAME_TOKENS", "empi"): "empi_id", ("_CARE_NAME_TOKENS", "fname"): "fname",
    ("_CARE_NAME_TOKENS", "lname"): "lname", ("_CARE_NAME_TOKENS", "dob"): "dob",
    ("_CARE_NAME_TOKENS", "dateofbirth"): "DATEOFBIRTH", ("_CARE_NAME_TOKENS", "mobile"): "mobile_no",
    ("_CARE_NAME_TOKENS", "telephone"): "telephone", ("_CARE_NAME_TOKENS", "fax"): "fax",
    ("_CARE_NAME_TOKENS", "street"): "street_line", ("_CARE_NAME_TOKENS", "zip"): "zip5",
    ("_CARE_NAME_TOKENS", "postal"): "postal_code",
    # id affixes glued to a care token (one glued word, so only the affix can split it)
    **{("_ID_AFFIXES", a): f"member{a}" for a in (
        "id", "ids", "no", "num", "nbr", "number", "key", "name", "nm", "code", "cd", "ref",
        "dt", "date")},
    # id stems: care only with an id suffix
    ("_CARE_ID_STEMS", "person"): "person_key", ("_CARE_ID_STEMS", "encounter"): "encounter_key",
    ("_CARE_ID_STEMS", "visit"): "visit_id", ("_CARE_ID_STEMS", "dependent"): "dependent_id",
    ("_CARE_ID_STEMS", "account"): "account_number", ("_CARE_ID_STEMS", "plan"): "health_plan_id",
    ("_CARE_ID_STEMS", "medicare"): "medicare_id", ("_CARE_ID_STEMS", "medicaid"): "medicaid_id",
    **{("_ID_SUFFIXES", a): f"visit{a}" for a in (
        "id", "ids", "key", "nbr", "num", "no", "number", "name", "nm")},
    # prefixes, glued to anything
    ("_CARE_PREFIXES", "email"): "emailaddr", ("_CARE_PREFIXES", "addr"): "addr1",
    ("_CARE_PREFIXES", "phone"): "phonenumber", ("_CARE_PREFIXES", "birth"): "birthplace_country",
    ("_CARE_PREFIXES", "death"): "death_date", ("_CARE_PREFIXES", "deceased"): "deceased_dt",
    # name parts + name suffixes
    ("_NAME_PARTS", "first"): "first_nm", ("_NAME_PARTS", "last"): "last_name",
    ("_NAME_PARTS", "middle"): "middleName", ("_NAME_PARTS", "full"): "full_name",
    ("_NAME_PARTS", "given"): "given_name", ("_NAME_PARTS", "sur"): "surname",
    ("_NAME_PARTS", "maiden"): "maiden_name", ("_NAME_PARTS", "family"): "family_name",
    ("_NAME_SUFFIXES", "name"): "givenname", ("_NAME_SUFFIXES", "nm"): "givennm",
}
SETS = ("_CARE_SUBJECT_TOKENS", "_CARE_NAME_TOKENS", "_ID_AFFIXES", "_CARE_ID_STEMS", "_ID_SUFFIXES", "_CARE_PREFIXES",
        "_NAME_PARTS", "_NAME_SUFFIXES")


class TestIssueNames:
    @pytest.mark.parametrize("name", ISSUE_NAMES)
    def test_listed_identifier_is_care_subject(self, name):
        assert classify.classify_column(name, {}) == ("care_subject", "name")

    @pytest.mark.parametrize("name", [
        "zipper_count",  # hostile: `zip` inside a longer word
        "contact",  # a bare "contact" is not an identifier by name (values catch emails)
        "visit_type", "plan_type", "account_type",  # id stems without an id suffix
        "dose_given", "visits", "hospital_id", "pathway_id", "start_date",
    ])
    def test_near_misses_stay_none(self, name):
        assert classify.classify_column(name, {})[0] == "none"


class TestEveryStemIsLoadBearing:
    def test_every_member_of_every_set_has_a_hostile_name(self):
        for set_name in SETS:
            members = {m for s, m in HOSTILE if s == set_name}
            assert members == getattr(classify, set_name), set_name

    @pytest.mark.parametrize("set_name, member", sorted(HOSTILE))
    def test_hostile_name_is_care_subject(self, set_name, member):
        assert classify.classify_column(HOSTILE[(set_name, member)], {})[0] == "care_subject"

    @pytest.mark.parametrize("set_name, member", sorted(HOSTILE))
    def test_removing_the_member_turns_its_name_to_none(self, monkeypatch, set_name, member):
        monkeypatch.setattr(classify, set_name, getattr(classify, set_name) - {member})
        assert classify.classify_column(HOSTILE[(set_name, member)], {})[0] == "none"


# ---------------------------------------------------------------- value patterns (rates only)
def _npi(base: str) -> str:
    """A checksum-valid NPI built at run time: the 9 base digits plus the CMS check digit (Luhn, 80840 prefix).

    Fixtures never hold a 10-digit literal, so none can be a real provider's number (checked against the public NPI
    registry, 2026-10-14: none of the values below is registered).
    """
    total = 0
    for i, d in enumerate(reversed([int(c) for c in "80840" + base + "0"])):
        if i % 2:
            d = d * 2 - 9 if d * 2 > 9 else d * 2
        total += d
    return base + str((10 - total % 10) % 10)


NPI_VALUES = [_npi("1234" + "56789"), _npi("1000" + "00000"), _npi("1999" + "99999")]
HOSTILE_VALUES = {
    # hostile: mixed case, plus-tags, subdomains
    "email": ["Jane.Doe@Example.org", "a+b" + "@mail.x.co", "z_9" + "@hosp.health"],
    # hostile: three US formats, none a bare 10-digit run (that is an NPI's shape)
    "phone": ["(312) 555-0100", "312-555-0101", "+1 312.555.0102"],
    "zip": ["60611", "60611-1234", "02139"],
    "date": ["2024-01-01", "1961-07-04T08:30:00Z", "7/4/61"],
}


class TestValuePatterns:
    @pytest.mark.parametrize("pattern", sorted(HOSTILE_VALUES))
    def test_renamed_column_is_caught_by_its_values(self, pattern):
        # hostile: the name says nothing ("contact" is none by name); only values can catch it
        rates = classify.pattern_verdicts(HOSTILE_VALUES[pattern] + [None, ""])
        assert rates[pattern] == 1.0
        assert classify.classify_column("contact", rates) == ("care_subject", "values")

    @pytest.mark.parametrize("pattern", sorted(HOSTILE_VALUES))
    def test_removing_the_pattern_lets_the_column_through(self, monkeypatch, pattern):
        monkeypatch.setattr(classify, "PATTERNS",
                            {k: v for k, v in classify.PATTERNS.items() if k != pattern})
        rates = classify.pattern_verdicts(HOSTILE_VALUES[pattern])
        assert classify.classify_column("contact", rates)[0] == "none"

    def test_npi_values_are_not_phones_and_stay_professional(self):
        # hostile: a valid 10-digit NPI must not be mistaken for a phone (care would win)
        rates = classify.pattern_verdicts(NPI_VALUES)
        assert rates["phone"] == 0.0 and rates["zip"] == 0.0
        assert classify.classify_column("npi", rates) == ("public_professional", "both")

    def test_int_cells_are_matched_as_text(self):
        # hostile: real ints, not strings (a zip held as an INT-typed cell in a sample)
        assert classify.pattern_verdicts([60611, 2139_0, 10001])["zip"] == 1.0

    @pytest.mark.parametrize("value", [
        "not an email@", "@x.org", "a@b", "555-0100", "312" + "5550100", "6061", "606111", "123456789", "2024-1-1x",
        "Dr Smith", "prov_005",
    ])
    def test_near_miss_values_match_no_care_pattern(self, value):
        rates = classify.pattern_verdicts([value])
        assert all(rates[p] == 0.0 for p in HOSTILE_VALUES), (value, rates)

    def test_values_are_never_kept(self):
        rates = classify.pattern_verdicts(HOSTILE_VALUES["email"])
        assert set(rates) == set(classify.PATTERNS)
        assert all(isinstance(r, float) for r in rates.values())
