"""`fclass bench` runs in a throwaway state folder and reports what it measured."""

import os

from fclass import bench
from fclass.classify import add_example, load_examples
from fclass.config import load_config

from test_classify import FakeBackend


def test_bench_is_isolated_and_reports(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    add_example("mine.pdf", "my own example", "Keep/Important")
    monkeypatch.setattr(bench, "make_backend", lambda *a, **k: FakeBackend())
    docs = bench.items(quick=True)
    assert len(docs) == 24 and len({label for _, label, _ in docs}) == 12
    r = bench.run_mode(load_config(), "embed", docs)
    assert r.total == 24 and 0 <= r.correct <= 24 and r.llm_calls == 0
    assert os.environ["XDG_STATE_HOME"] == str(tmp_path / "state")      # restored
    assert [e["name"] for e in load_examples()] == ["mine.pdf"]          # your examples untouched
    report = bench.to_json([r], quick=True)
    assert report["results"][0]["total"] == 24 and "| embed |" in bench.markdown(report)
