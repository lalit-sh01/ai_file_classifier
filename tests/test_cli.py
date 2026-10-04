"""The interactive flows, driven through the real CLI with a fake model and scripted answers."""

import pytest

from fclass import backends, cli, questions
from fclass.classify import load_examples
from fclass.config import load_config

from test_classify import KEYWORDS, FakeBackend
from test_features import NamingBackend

CONFIG = """
[model]
backend = "ollama"
vision = "off"
[organize]
destination = "{dest}"
[strategy]
margin = 0.05
""" + "".join(f'\n[[category]]\npath = "{p}"\ndescription = "{d}"\n' for p, d in KEYWORDS.items())


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    cfg = tmp_path / "config.toml"
    cfg.write_text(CONFIG.format(dest=tmp_path / "home"))
    inbox = tmp_path / "Downloads"
    inbox.mkdir()
    (inbox / "passport.txt").write_text("passport lease contract passport")   # clear
    (inbox / "mystery.bin").write_bytes(b"\0\1\2" * 40)                       # unreadable: asked
    monkeypatch.setattr(cli, "interactive", lambda: True)
    backend = FakeBackend()
    monkeypatch.setattr(cli, "make_backend", lambda *a, **k: backend)
    return tmp_path, cfg, inbox


def answers(monkeypatch, *replies):
    it = iter(replies)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(it))


def test_sort_asks_and_creates_a_category_on_the_spot(env, monkeypatch, capsys):
    tmp, cfg, inbox = env
    # question for mystery.bin: new category, then confirm the move
    answers(monkeypatch, "n", "Software/Installers", "Programs and installers", "y")
    cli.main(["--config", str(cfg), "sort", str(inbox)])
    assert (tmp / "home/Keep/Important/passport.txt").exists()
    assert (tmp / "home/Software/Installers/mystery.bin").exists()
    assert "Software/Installers" in load_config(cfg).category_paths
    assert load_examples()[-1]["category"] == "Software/Installers"
    assert "needs your answer" in capsys.readouterr().out


def test_sort_yes_saves_questions_then_ask_files_them(env, monkeypatch):
    tmp, cfg, inbox = env
    other = tmp / "elsewhere"
    cli.main(["--config", str(cfg), "sort", str(inbox), "-y", "--dest", str(other)])
    assert (inbox / "mystery.bin").exists()                     # not moved, not guessed
    assert [q.name for q in questions.pending()] == ["mystery.bin"]
    answers(monkeypatch, "a", "4")                               # show all, pick the 4th
    cli.main(["--config", str(cfg), "ask"])
    assert (other / list(KEYWORDS)[3] / "mystery.bin").exists()  # answered into the --dest it was asked for
    assert questions.pending() == []
    cli.main(["--config", str(cfg), "undo", "--last", "1"])
    assert (inbox / "mystery.bin").exists()


def test_leave_it_means_it_stays(env, monkeypatch):
    tmp, cfg, inbox = env
    answers(monkeypatch, "s", "y")
    cli.main(["--config", str(cfg), "sort", str(inbox)])
    assert (inbox / "mystery.bin").exists() and questions.pending() == []


def test_discover_then_accept_adds_categories_and_examples(env, monkeypatch):
    tmp, cfg, _ = env
    messy = tmp / "Messy"
    messy.mkdir()
    for i in range(4):
        (messy / f"pay{i}.txt").write_text("salary payslip march salary")
        (messy / f"hotel{i}.txt").write_text("hotel flight booking itinerary")
    monkeypatch.setattr(cli, "make_backend", lambda *a, **k: NamingBackend())
    answers(monkeypatch, "y")
    cli.main(["--config", str(cfg), "discover", str(messy)])
    paths = load_config(cfg).category_paths
    assert "Work/Pay Slips" in paths and paths.count("Recreation/Travel") == 1
    assert any(e["category"] == "Work/Pay Slips" for e in load_examples())


def test_remote_server_refused(env, monkeypatch, capsys):
    tmp, cfg, inbox = env
    cfg.write_text(cfg.read_text().replace('backend = "ollama"', 'backend = "openai"\nurl = "https://api.example.com/v1"'))
    monkeypatch.setattr(cli, "make_backend", backends.make_backend)  # the real one, with the offline guard
    with pytest.raises(SystemExit) as e:
        cli.main(["--config", str(cfg), "sort", str(inbox)])
    assert "not on this computer" in str(e.value.code)


def test_no_terminal_no_yes_moves_nothing(env, monkeypatch, capsys):
    tmp, cfg, inbox = env
    monkeypatch.setattr(cli, "interactive", lambda: False)
    cli.main(["--config", str(cfg), "sort", str(inbox)])
    assert (inbox / "passport.txt").exists() and "Nothing moved" in capsys.readouterr().out
