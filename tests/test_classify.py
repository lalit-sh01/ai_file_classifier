"""Pipeline logic with a deterministic fake backend (no model needed)."""

import tomllib

import pytest

from fclass.backends import Backend
from fclass.classify import Classifier, add_example
from fclass.config import DEFAULT_CONFIG_TOML, parse_config
from fclass.extract import Preview
from fclass.planner import build_plan

KEYWORDS = {
    "Finance/Statements": "bank statement balance invoice bill",
    "Finance/Taxes": "tax w-2 1099 return deduction",
    "Keep/Important": "passport lease contract insurance policy will",
    "Recreation/Travel": "flight hotel itinerary booking visa",
}
VOCAB = sorted({w for v in KEYWORDS.values() for w in v.split()})


class FakeBackend(Backend):
    """Bag-of-words embeddings; the 'LLM' always picks the last choice it is offered."""

    def __init__(self):
        self.embed_calls = 0
        self.choose_calls = []

    def embed(self, texts):
        self.embed_calls += len(texts)
        return [[t.lower().count(w) + 0.01 for w in VOCAB] for t in texts]

    def choose(self, system, user, choices, image=None):
        self.choose_calls.append(choices)
        return {"category": choices[-1], "reason": "fake"}

    def status(self):
        return {"ok": True, "models": [], "detail": "fake"}


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))


def make_cfg(mode="hybrid", margin=0.05, tmp_path=None):
    data = tomllib.loads(DEFAULT_CONFIG_TOML)
    data["category"] = [{"path": p, "description": d} for p, d in KEYWORDS.items()]
    data["strategy"].update(mode=mode, margin=margin)
    if tmp_path:
        data["organize"]["destination"] = str(tmp_path / "home")
    return parse_config(data)


def test_clear_case_decided_by_embeddings_without_llm():
    fb = FakeBackend()
    clf = Classifier(make_cfg(), fb)
    v = clf.classify("x.pdf", "file", Preview("Passport contract lease passport"))
    assert v.category == "Keep/Important" and v.via == "embed" and v.confidence >= 0.75
    assert fb.choose_calls == []


def test_close_call_goes_to_llm_with_all_categories_ranked():
    fb = FakeBackend()
    clf = Classifier(make_cfg(margin=0.9), fb)  # nothing is ever "clear"
    v = clf.classify("x.pdf", "file", Preview("tax bank"))
    assert v.via == "llm"
    assert sorted(fb.choose_calls[0]) == sorted(KEYWORDS)  # every category offered
    assert fb.choose_calls[0][0] in ("Finance/Taxes", "Finance/Statements")  # best guesses first
    assert v.category == fb.choose_calls[0][-1]
    assert v.confidence < 0.8  # LLM disagreed with the embedding top pick


def test_shortlist_option_cuts_choices():
    fb = FakeBackend()
    cfg = make_cfg(margin=0.9)
    cfg.strategy.shortlist = 2
    Classifier(cfg, fb).classify("x.pdf", "file", Preview("tax bank"))
    assert len(fb.choose_calls[0]) == 2


def test_llm_mode_offers_every_category_and_never_embeds():
    fb = FakeBackend()
    clf = Classifier(make_cfg(mode="llm"), fb)
    clf.classify("x.pdf", "file", Preview("whatever"))
    assert fb.choose_calls[0] == list(KEYWORDS) and fb.embed_calls == 0


def test_cache_hit_skips_models():
    fb = FakeBackend()
    cfg = make_cfg(margin=0.9)
    first = Classifier(cfg, fb)
    first.classify("y.pdf", "file", Preview("hotel flight"))
    first.save()
    v = Classifier(cfg, fb).classify("y.pdf", "file", Preview("hotel flight"))
    assert v.via == "cache" and len(fb.choose_calls) == 1


def test_unreadable_file_is_low_confidence():
    clf = Classifier(make_cfg(), FakeBackend())
    v = clf.classify("IMG_1.jpg", "file", Preview("[No readable text]", readable=False))
    assert v.confidence < 0.5


def test_corrections_shift_the_ranking():
    fb = FakeBackend()
    cfg = make_cfg()
    text = "bill invoice for hotel"
    before = Classifier(cfg, fb).rank(fb.embed([text])[0])[0][0]
    for i in range(3):
        add_example(f"trip{i}.pdf", "hotel invoice bill", "Recreation/Travel")
    after = Classifier(cfg, fb).rank(fb.embed([text])[0])[0][0]
    assert before == "Finance/Statements" and after == "Recreation/Travel"


def test_plan_applies_rules_review_and_folders(tmp_path):
    src = tmp_path / "Downloads"
    src.mkdir()
    (src / "a.txt").write_text("passport lease contract")
    (src / "Setup.dmg").write_bytes(b"\0")
    (src / ".hidden").write_text("x")
    (src / "blob.bin").write_bytes(b"\0\1" * 50)
    (src / "trip").mkdir()
    (src / "trip/plan.txt").write_text("flight hotel itinerary")
    cfg = make_cfg(tmp_path=tmp_path)
    plan = build_plan(cfg, src, Classifier(cfg, FakeBackend()))
    by = {i.name: i for i in plan.items}
    assert set(by) == {"a.txt", "Setup.dmg", "blob.bin", "trip"}
    assert by["Setup.dmg"].category is None
    assert by["a.txt"].category == "Keep/Important" and not by["a.txt"].review
    assert by["blob.bin"].review
    assert by["trip"].kind == "folder" and by["trip"].category == "Recreation/Travel"
    assert plan.items[-1].kind == "folder"  # files first


def test_organising_destination_itself_skips_owned_folders(tmp_path):
    home = tmp_path / "home"
    (home / "Finance").mkdir(parents=True)
    (home / "_Review").mkdir()
    (home / "w2.txt").write_text("tax w-2 return")
    cfg = make_cfg(tmp_path=tmp_path)
    plan = build_plan(cfg, home, Classifier(cfg, FakeBackend()))
    assert [i.name for i in plan.items] == ["w2.txt"]


def test_single_example_does_not_tilt_unrelated_files():
    """Regression: one taught example used to boost its category for every file."""
    fb = FakeBackend()
    cfg = make_cfg()
    add_example("lease.pdf", "passport lease contract", "Keep/Important")
    clf = Classifier(cfg, fb)
    ranked = clf.rank(fb.embed(["flight hotel itinerary booking"])[0])
    assert ranked[0][0] == "Recreation/Travel"
    plain = Classifier(cfg, fb)
    plain.examples = []
    assert dict(ranked)["Keep/Important"] == dict(plain.rank(fb.embed(["flight hotel itinerary booking"])[0]))["Keep/Important"]
