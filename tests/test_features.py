"""Offline guard, asking when unsure, editing categories, watching folders, discovering categories."""

import os
import time
import tomllib

import pytest

from fclass.classify import Classifier, load_examples
from fclass.config import (DEFAULT_CONFIG_TOML, Category, ModelSettings, OfflineError, add_categories,
                           check_offline, load_config, locality, parse_config, remove_category,
                           replace_categories)
from fclass.discover import clean_path, discover, over_split
from fclass.plan import Item, latest_journal, undo
from fclass import questions
from fclass.watch import Watcher

from test_classify import KEYWORDS, FakeBackend


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))


def cfg_for(tmp_path, **organize):
    data = tomllib.loads(DEFAULT_CONFIG_TOML)
    data["category"] = [{"path": p, "description": d} for p, d in KEYWORDS.items()]
    data["organize"].update(destination=str(tmp_path / "home"), **organize)
    data["watch"].update(settle_seconds=5)
    return parse_config(data)


# ── offline ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("url,where", [
    ("http://localhost:11434", "device"), ("http://127.0.0.1:1234/v1", "device"), ("http://[::1]:8080", "device"),
    ("http://192.168.1.20:11434", "network"), ("http://nas.local:11434", "network"),
    ("https://api.openai.com/v1", "remote"), ("http://8.8.8.8", "remote"),
])
def test_locality(url, where):
    assert locality(url) == where


def test_remote_model_server_is_refused_unless_allowed():
    with pytest.raises(OfflineError):
        check_offline(ModelSettings(backend="openai", url="https://api.example.com/v1"))
    check_offline(ModelSettings(backend="openai", url="https://api.example.com/v1", allow_remote=True))
    check_offline(ModelSettings(url="http://localhost:11434"))


def test_cloud_backend_is_gone():
    data = tomllib.loads(DEFAULT_CONFIG_TOML)
    data["model"]["backend"] = "anthropic"
    with pytest.raises(ValueError):
        parse_config(data)


# ── asking ──────────────────────────────────────────────────────────────────

def scripted(*answers):
    it = iter(answers)
    return lambda prompt: next(it)


def unsure_item(tmp_path, name="odd.pdf"):
    f = tmp_path / name
    f.write_text("x")
    return Item(str(f), "file", "Finance/Taxes", 0.3, "looks like a tax form", "llm",
                ["Keep/Important", "Finance/Statements"], True, "Form 16 part A", "PDF, 1 page")


def test_ask_enter_takes_best_guess_and_numbers_pick(tmp_path):
    cats = list(KEYWORDS)
    it = unsure_item(tmp_path)
    assert questions.ask(it, cats, scripted(""), say=lambda s: None).category == "Finance/Taxes"
    assert questions.ask(it, cats, scripted("2"), say=lambda s: None).category == "Keep/Important"
    assert questions.ask(it, cats, scripted("a", "4"), say=lambda s: None).category == cats[3]
    assert questions.ask(it, cats, scripted("s"), say=lambda s: None).action == "leave"
    assert questions.ask(it, cats, scripted("zzz", "q"), say=lambda s: None).action == "quit"


def test_ask_new_category(tmp_path):
    ans = questions.ask(unsure_item(tmp_path), list(KEYWORDS), scripted("n", "Work/Payslips", "Monthly salary slips"),
                        say=lambda s: None)
    assert (ans.action, ans.category, ans.description) == ("new", "Work/Payslips", "Monthly salary slips")
    bad = questions.ask(unsure_item(tmp_path), list(KEYWORDS), scripted("n", "../etc", "s"), say=lambda s: None)
    assert bad.action == "leave"


def test_queue_keeps_one_entry_per_file_and_drops_vanished(tmp_path):
    a, b = unsure_item(tmp_path, "a.pdf"), unsure_item(tmp_path, "b.pdf")
    assert questions.enqueue([a, b], tmp_path) == 2 and questions.enqueue([a], tmp_path) == 0
    os.remove(b.src)
    assert [i.name for i in questions.pending()] == ["a.pdf"]
    questions.resolve(a.src)
    assert questions.pending() == []


# ── editing categories in place ─────────────────────────────────────────────

def test_add_remove_replace_keep_comments(tmp_path):
    path = tmp_path / "config" / "fclass" / "config.toml"
    add_categories([Category("Work/Payslips", 'Salary "slips"')], path)
    text = path.read_text()
    assert "# ── Rules" in text and 'Salary \\"slips\\"' in text
    cfg = load_config(path)
    assert cfg.category_paths[-1] == "Work/Payslips"
    with pytest.raises(ValueError):
        add_categories([Category("Work/Payslips", "dup")], path)
    remove_category("Finance/Taxes", path)
    cfg = load_config(path)
    assert "Finance/Taxes" not in cfg.category_paths and "# ── Rules" in path.read_text()
    replace_categories([Category("A/B", "one"), Category("C", "two")], path)
    assert load_config(path).category_paths == ["A/B", "C"]
    assert "[watch]" in path.read_text() and path.with_suffix(".toml.bak").exists()


# ── watching ────────────────────────────────────────────────────────────────

class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def test_watch_waits_for_downloads_to_settle_then_files_or_asks(tmp_path):
    inbox = tmp_path / "Downloads"
    inbox.mkdir()
    (inbox / "old.txt").write_text("passport lease contract")  # there before watching: untouched
    cfg = cfg_for(tmp_path)
    clock, events = Clock(), []
    w = Watcher(cfg, Classifier(cfg, FakeBackend()), [inbox], lambda e, it, d: events.append((e, it.name)),
                clock=clock)
    (inbox / "trip.txt").write_text("flight hotel itinerary booking")
    (inbox / "big.iso.crdownload").write_text("partial")      # browser partial: ignored
    (inbox / "mystery.bin").write_bytes(b"\0\1\2" * 30)       # unreadable: a question
    assert w.tick() == 0                                       # first sighting: wait
    clock.t += 2
    (inbox / "trip.txt").write_text("flight hotel itinerary booking visa")  # still being written
    assert w.tick() == 0
    clock.t += 4                                               # t=1006: mystery unchanged 6s, trip only 4s
    assert w.tick() == 1                                       # mystery.bin settled
    clock.t += 2                                               # t=1008: trip unchanged 6s since its last write
    assert w.tick() == 1
    assert ("filed", "trip.txt") in events and ("asked", "mystery.bin") in events
    assert (tmp_path / "home/Recreation/Travel/trip.txt").exists()
    assert (inbox / "old.txt").exists() and (inbox / "mystery.bin").exists()
    assert [i.name for i in questions.pending()] == ["mystery.bin"]
    restored, _ = undo(latest_journal(), last=1)
    assert restored == 1 and (inbox / "trip.txt").exists()


# ── discovering ─────────────────────────────────────────────────────────────

class NamingBackend(FakeBackend):
    def generate(self, system, user, schema, model=None):
        text = user.lower()
        if "hotel" in text or "flight" in text:
            return {"reason": "", "fits_existing": "Recreation/Travel", "path": "", "description": ""}
        if "payslip" in text or "salary" in text:
            return {"reason": "", "fits_existing": "none", "path": "work / pay slips!", "description": "Salary slips"}
        return {"reason": "", "fits_existing": "none", "path": "Misc/Other", "description": "Other things"}


def test_over_split_makes_more_groups_than_obvious():
    vecs = [[1, 0, 0]] * 6 + [[0, 1, 0]] * 6 + [[0, 0, 1]] * 6
    k, labels = over_split(vecs, per_group=3)
    assert k == 6 and len(set(labels)) >= 3


def test_discover_names_merges_and_reuses_existing(tmp_path):
    root = tmp_path / "Messy"
    root.mkdir()
    for i in range(4):
        (root / f"hotel{i}.txt").write_text("hotel flight booking itinerary")
        (root / f"pay{i}.txt").write_text("salary payslip march salary")
    cfg = cfg_for(tmp_path)
    prop = discover(cfg, Classifier(cfg, NamingBackend()), root)
    by_path = {g.path: g for g in prop.groups}
    assert by_path["Recreation/Travel"].existing == "Recreation/Travel"
    assert "Work/Pay Slips" in by_path and by_path["Work/Pay Slips"].is_new
    assert sum(len(g.files) for g in prop.groups) + len(prop.ungrouped) == 8


def test_clean_path():
    assert clean_path(" home / utility bills ") == "Home/Utility Bills"
    assert clean_path("a/b/c/d") == "A/B"
