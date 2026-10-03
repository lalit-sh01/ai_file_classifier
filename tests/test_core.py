import json
import tomllib
import zipfile
from pathlib import Path

import pytest

from fclass.backends import choice_schema, parse_choice
from fclass.config import DEFAULT_CONFIG_TOML, parse_config
from fclass.extract import preview_file, preview_folder
from fclass.plan import Item, Plan, apply, undo


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))


def cfg(**organize):
    data = tomllib.loads(DEFAULT_CONFIG_TOML)
    data["organize"].update(organize)
    return parse_config(data)


# ── config ────────────────────────────────────────────────────────────────

def test_default_config_parses():
    c = cfg()
    assert "Keep/Important" in c.category_paths
    assert c.rules == []  # no file type is skipped by default
    assert c.when_unsure == "ask" and c.model.vision == "auto" and not c.model.allow_remote
    assert c.is_ignored(".DS_Store") and c.is_ignored("~$report.docx")
    assert {"Finance", "Study", "Recreation", "Keep", "_Review"} <= c.top_level_names


def test_rule_requires_category_or_skip():
    data = tomllib.loads(DEFAULT_CONFIG_TOML)
    data["rule"] = [{"match": "*.x"}]
    with pytest.raises(ValueError):
        parse_config(data)


def test_fingerprint_changes_with_taxonomy():
    a = cfg()
    b = cfg()
    b.categories[0].description += " and receipts"
    assert a.fingerprint() != b.fingerprint()


# ── extraction (stdlib office formats) ────────────────────────────────────

W = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'


def make_docx(path: Path, paragraphs: list[str]):
    body = "".join(f"<w:p><w:r><w:t>{p}</w:t></w:r></w:p>" for p in paragraphs)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("word/document.xml", f"<w:document {W}><w:body>{body}</w:body></w:document>")


def make_xlsx(path: Path):
    ns = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("xl/workbook.xml", f"<workbook {ns}/>")
        z.writestr("xl/sharedStrings.xml", f"<sst {ns}><si><t>Ticker</t></si><si><t>VTI</t></si></sst>")
        z.writestr("xl/worksheets/sheet1.xml",
                   f'<worksheet {ns}><sheetData><row><c t="s"><v>0</v></c><c><v>Shares</v></c></row>'
                   f'<row><c t="s"><v>1</v></c><c><v>120</v></c></row></sheetData></worksheet>')


def test_docx_and_xlsx_extract_without_dependencies(tmp_path):
    make_docx(tmp_path / "a.docx", ["RESIDENTIAL LEASE AGREEMENT", "Monthly rent $2,400"])
    make_xlsx(tmp_path / "b.xlsx")
    doc = preview_file(tmp_path / "a.docx")
    assert "LEASE AGREEMENT" in doc.text and doc.kind == "Word document"
    sheet = preview_file(tmp_path / "b.xlsx")
    assert "Ticker, Shares\nVTI, 120" in sheet.text and sheet.kind == "Excel spreadsheet"


def test_binary_is_unreadable_and_images_optional(tmp_path):
    (tmp_path / "blob.bin").write_bytes(b"\x00\x01\x02" * 100)
    (tmp_path / "pic.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 50)
    assert not preview_file(tmp_path / "blob.bin").readable
    assert preview_file(tmp_path / "pic.png").image is None
    assert preview_file(tmp_path / "pic.png", want_image=True).image.startswith(b"\x89PNG")


def test_folder_preview_summarises(tmp_path):
    d = tmp_path / "proj"
    d.mkdir()
    (d / "README.md").write_text("Arduino weather station")
    (d / "main.ino").write_text("void setup(){}")
    p = preview_folder(d)
    assert "2 files" in p.text and "Arduino weather station" in p.text


# ── schema parsing ────────────────────────────────────────────────────────

def test_parse_choice_normalises_and_rejects():
    choices = ["Finance/Taxes", "Keep/Important"]
    assert choice_schema(choices)["properties"]["category"]["enum"] == choices
    assert parse_choice('{"category": "finance / taxes", "reason": "W-2"}', choices)["category"] == "Finance/Taxes"
    assert parse_choice('noise {"category": "Keep/Important"} noise', choices)["category"] == "Keep/Important"
    assert parse_choice('{"category": "Made/Up"}', choices)["category"] is None


# ── plan / apply / undo ───────────────────────────────────────────────────

def test_apply_never_overwrites_and_undo_restores(tmp_path):
    src, dest = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    (src / "w2.pdf").write_text("new")
    (src / "odd.txt").write_text("?")
    (dest / "Finance/Taxes").mkdir(parents=True)
    (dest / "Finance/Taxes/w2.pdf").write_text("existing")

    plan = Plan(str(src), str(dest), "_Review", "test", [
        Item(str(src / "w2.pdf"), "file", "Finance/Taxes", 0.9),
        Item(str(src / "odd.txt"), "file", "Keep/Archives", 0.2, review=True),
        Item(str(src / "skip.dmg"), "file", None, via="skip"),
    ], when_unsure="review_folder")
    journal, moved, errors = apply(plan)
    assert moved == 2 and not errors
    assert (dest / "Finance/Taxes/w2.pdf").read_text() == "existing"
    assert (dest / "Finance/Taxes/w2 (2).pdf").read_text() == "new"
    assert (dest / "_Review/odd.txt").exists()

    restored, problems = undo(journal)
    assert restored == 2 and not problems
    assert (src / "w2.pdf").read_text() == "new" and (src / "odd.txt").exists()
    assert (dest / "Finance/Taxes/w2.pdf").read_text() == "existing"


def test_plan_roundtrip(tmp_path):
    plan = Plan("a", "b", "_Review", "m", [Item("a/x.pdf", "file", "Keep/Manuals", 0.7, alternatives=["Keep/Archives"])])
    again = Plan.from_json(plan.to_json())
    assert again.items[0].alternatives == ["Keep/Archives"]
    assert json.loads(plan.to_json())["items"][0]["category"] == "Keep/Manuals"


def test_unsure_items_stay_put_when_asking(tmp_path):
    src, dest = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    (src / "odd.txt").write_text("?")
    plan = Plan(str(src), str(dest), "_Review", "test", [
        Item(str(src / "odd.txt"), "file", "Keep/Archives", 0.2, review=True)])
    _, moved, _ = apply(plan)
    assert moved == 0 and (src / "odd.txt").exists()


def test_undo_last_n_only(tmp_path):
    src, dest = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    for n in "abc":
        (src / f"{n}.txt").write_text(n)
    plan = Plan(str(src), str(dest), "_Review", "t",
                [Item(str(src / f"{n}.txt"), "file", "Keep/Archives", 0.9) for n in "abc"])
    journal, moved, _ = apply(plan)
    restored, _ = undo(journal, last=1)
    assert restored == 1 and (src / "c.txt").exists() and not (src / "a.txt").exists()
    restored, _ = undo(journal)
    assert restored == 2 and (src / "a.txt").exists() and (src / "b.txt").exists()
