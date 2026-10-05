"""Tests for the real-data sources, against a committed fixture snapshot. No network.

tests/fixtures/snapshot-2026-10-05/ is trimmed from the first live harvest: five
real VA rows for "foley catheter" (two edited, marked in the file), the detail
fields harvested for them, and the FDA registry records the harvester kept.
Two defects that live harvest exposed are pinned here: a false registry join on
a generic part number, and SKU false-positives on catalog numbers with spaces.
"""
from datetime import date
from pathlib import Path

import pytest

from src.guardrails import SKU_PATTERN
from src.ingest import load_all
from src.sources import harvest, openfda, va_nac
from src.sources.va_nac import PageFormatError, VaNacSource

SNAP = Path(__file__).parent / "fixtures" / "snapshot-2026-10-05"
LIST_PAGE = (SNAP / "va_nac" / "list" / "foley-catheter_p1.html").read_text(encoding="utf-8")


# --- VA list page ---------------------------------------------------------

def test_list_page_parses_rows_and_total():
    page = va_nac.parse_list_page(LIST_PAGE)
    assert len(page.rows) == 5 and page.total == 22 and not page.has_next
    first = page.rows[0]
    assert (first["catalog_number"], first["contract_number"], first["lognumber"]) == \
        ("150", "36F79726D0183", "9274850")
    assert first["contractor"] and first["price"] == "$284.22"


def test_renamed_column_fails_loudly():
    with pytest.raises(PageFormatError, match="headers changed"):
        va_nac.parse_list_page(LIST_PAGE.replace(">Contractor Name<", ">Vendor<"))


def test_dropped_cell_fails_loudly():
    broken = LIST_PAGE.replace('<td class="list_content">$284.22</td>', "", 1)
    with pytest.raises(PageFormatError, match="cells"):
        va_nac.parse_list_page(broken)


# --- VA detail page -------------------------------------------------------

DETAIL = """<html><body>
<h2>Item Details:5501</h2>
<dt>Price:</dt><dd>$115.58 / CA</dd>
<dt>Date&nbsp;Effective:</dt><dd>4/15/2025</dd>
<dt>Expiration&nbsp;Date:</dt><dd>4/14/2030</dd>
<dt>Contract Point of Contact:</dt><dt>Name:</dt><dd>A Person</dd>
<dt>EMail:</dt><dd>person@example.com</dd>
<dt>Contract Dates:</dt><dt>Awarded:</dt><dd>4/1/2025</dd>
<dt>Effective:</dt><dd>4/15/2025</dd><dt>Expiration:</dt><dd>4/14/2030</dd>
</body></html>"""


def test_detail_page_yields_unit_and_dates_only():
    fields = va_nac.parse_detail_page(DETAIL)
    assert fields == {"uom": "CA", "effective_start": date(2025, 4, 15),
                      "effective_end": date(2030, 4, 14),
                      "contract_start": date(2025, 4, 15),
                      "contract_end": date(2030, 4, 14)}   # no contact fields, by design


def test_detail_falls_back_to_contract_dates():
    no_item_dates = DETAIL.replace("Date&nbsp;Effective", "X").replace("Expiration&nbsp;Date", "Y")
    fields = va_nac.parse_detail_page(no_item_dates)
    assert fields["effective_start"] == date(2025, 4, 15)


@pytest.mark.parametrize("text,expected", [
    ("$1,234.50", 1234.5), ("$3.79", 3.79), ("", None), ("Call", None), ("$0.00", None)])
def test_parse_price(text, expected):
    assert va_nac.parse_price(text) == expected


@pytest.mark.parametrize("desc,uom,expected", [
    ("Hold-n-Place Foley Catheter 10/BX", "box", 10),
    ("Hold-n-Place Foley Catheter 10/BX", "case", None),   # a box count says nothing about a case
    ("Sponge, case of 50", "case", 50),
    ("Foley catheter 16Fr", "box", None),
    ("Foley catheter 10/BX", None, None),
])
def test_pack_from_description_only_for_the_priced_unit(desc, uom, expected):
    assert va_nac.pack_from_description(desc, uom) == expected


# --- FDA registry -----------------------------------------------------------

GLOVE = {
    "company_name": "M.C. JOHNSON CO., INC.", "device_count_in_base_package": 1,
    "identifiers": [
        {"id": "A", "type": "Primary"},
        {"id": "B", "type": "Package", "unit_of_use_id": "A", "quantity_per_package": "100",
         "package_type": "INNER CARTON"},
        {"id": "C", "type": "Package", "unit_of_use_id": "B", "quantity_per_package": "10",
         "package_type": "MASTER SHIP CASE"},
    ],
}


@pytest.mark.parametrize("unit,expected", [("case", 1000), ("carton", 100), ("box", None)])
def test_units_walk_the_package_chain(unit, expected):
    assert openfda.units_per_package(GLOVE, unit) == expected


def test_trim_drops_contacts():
    rec = {**GLOVE, "customer_contacts": [{"email": "x@example.com", "phone": "1"}],
           "gmdn_terms": [{"name": "Nitrile glove"}]}
    trimmed = openfda.trim(rec)
    assert "customer_contacts" not in trimmed and trimmed["gmdn_terms"] == ["Nitrile glove"]


# --- The whole source, from the fixture snapshot ----------------------------

@pytest.fixture(scope="module")
def loaded():
    source = VaNacSource(SNAP)
    rows = source.load()
    return source, {(r.sku, r.contract_id): r for r in rows}


def test_blank_price_is_rejected_and_counted(loaded):
    source, rows = loaded
    assert source.report()["rejected"] == {"unparseable_or_zero_price": 1}
    assert not any(sku == "316" for sku, _ in rows)


def test_bpa_price_is_its_own_contract_row(loaded):
    _, rows = loaded
    assert rows[("57165814", "36F79720D0038")].contracted_unit_price == 3.79
    assert rows[("57165814", "36C10X26A0001")].contracted_unit_price == 3.41


def test_unit_and_dates_come_from_detail(loaded):
    _, rows = loaded
    row = rows[("SILQ 21400101003", "36F79718D0321")]
    assert row.uom == "box" and row.units_per_pack == 10          # "10/BX" in the text
    assert row.effective_start == date(2017, 12, 15)


def test_item_without_detail_has_unknown_unit(loaded):
    _, rows = loaded
    assert rows[("57165814", "36F79720D0038")].uom is None


def test_contract_dates_reach_rows_without_detail():
    # 316 has no detail page in the fixture, but 150 on the same contract does.
    raw = va_nac.parse_list_page(LIST_PAGE).rows
    details = {"9274850": {"uom": "BX", "contract_start": "2026-09-01",
                           "contract_end": "2031-08-31"}}
    for r in raw:
        if r["catalog_number"] == "316":
            r["price"] = "$73.97"
    rows = {r.sku: r for r in va_nac.rows_to_prices(raw, details, "t").rows}
    assert rows["316"].uom is None
    assert rows["316"].effective_start == date(2026, 9, 1)


def test_generic_part_number_is_not_joined(loaded):
    # Regression: the first live harvest joined this Foley catheter to another
    # company's sclerotherapy catheter, because both use part number "150".
    source, rows = loaded
    assert rows[("150", "36F79726D0183")].manufacturer is None
    assert source.report()["registry_join"]["refused"]["part_number_too_generic"] >= 1


def test_distinctive_part_joins_with_maker_group_and_pack(loaded):
    _, rows = loaded
    row = rows[("81-080416", "36F79723D0026")]
    assert row.manufacturer == "DEROYAL INDUSTRIES, INC."
    assert "catheter" in row.product_group.lower()
    assert row.uom == "case" and row.units_per_pack == 50


def test_real_rows_pass_vetting_unflagged():
    # Regression: catalog numbers with spaces ("SILQ 21400101003") used to fail
    # the SKU check and flag 3 of 22 real lines.
    rows, report = load_all([VaNacSource(SNAP)])
    assert report.flagged == 0 and report.total == len(rows) == 5


@pytest.mark.parametrize("sku,ok", [
    ("SILQ 21400101003", True), ("81-080416EU", True), ("GLV-N100", True),
    ("A  B", False), ("X | vendor=Y", False), ("ABC ", False), ('A"B', False)])
def test_sku_pattern(sku, ok):
    assert bool(SKU_PATTERN.fullmatch(sku)) is ok


# --- Harvester: refuses to overwrite a snapshot -------------------------------

def test_harvest_refuses_existing_snapshot(tmp_path):
    (tmp_path / "x").write_text("already here")
    assert harvest.main(["--out", str(tmp_path)]) == 1     # returns before any request


def test_corpus_version_sees_units_and_dates():
    from src.corpus import corpus_version
    rows, _ = load_all([VaNacSource(SNAP)])
    changed = [r.model_copy(update={"units_per_pack": 999}) if r.units_per_pack else r
               for r in rows]
    assert corpus_version(rows) != corpus_version(changed)
