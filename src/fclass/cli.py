"""fclass: sort files by what they contain, with a small model on your own computer.

  fclass sort ~/Downloads          plan, show, ask, move
  fclass plan ~/Downloads          plan only (saved for later)
  fclass apply [PLAN]              execute a saved plan
  fclass undo                      put the last run back
  fclass teach FILE CATEGORY       correct a decision; future runs learn from it
  fclass init | doctor | categories
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from . import __version__
from .backends import BackendError, make_backend
from .classify import Classifier, add_example
from .config import default_config_path, load_config, write_default_config
from .extract import pdf_support, preview_file, preview_folder
from .plan import Plan, apply, latest_journal, latest_plan, undo
from .planner import build_plan

# ── tiny terminal toolkit ──────────────────────────────────────────────────

COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def c(text, code):
    return f"\033[{code}m{text}\033[0m" if COLOR else str(text)


def dim(t): return c(t, "2")
def bold(t): return c(t, "1")
def green(t): return c(t, "32")
def yellow(t): return c(t, "33")
def red(t): return c(t, "31")
def cyan(t): return c(t, "36")


def meter(conf: float) -> str:
    filled = round(conf * 5)
    bar = "●" * filled + "○" * (5 - filled)
    return green(bar) if conf >= 0.75 else yellow(bar) if conf >= 0.5 else red(bar)


VIA = {"rule": "rule", "embed": "fast", "llm": "llm", "cache": "cached", "user": "you", "skip": "rule", "error": "error"}


def show_plan(plan: Plan) -> None:
    groups: dict[str, list] = {}
    for it in plan.items:
        key = "· skipped" if it.category is None else (plan.review_folder if it.review else it.category)
        groups.setdefault(key, []).append(it)

    print()
    print(bold(f"  {plan.destination}/"))
    order = sorted(k for k in groups if k not in (plan.review_folder, "· skipped"))
    order += [k for k in (plan.review_folder, "· skipped") if k in groups]
    last_top = None
    for key in order:
        if key == "· skipped":
            print(f"\n  {dim('· left in place')}")
        elif key == plan.review_folder:
            print(f"\n  {yellow(key + '/')}  {dim('← needs your eyes')}")
        else:
            top, _, sub = key.partition("/")
            if top != last_top:
                print(f"  {cyan(top + '/')}")
                last_top = top
            print(f"    {cyan(sub + '/') if sub else ''}")
        for it in groups[key]:
            name = it.name + ("/" if it.kind == "folder" else "")
            name = name if len(name) <= 44 else name[:41] + "…"
            if key == "· skipped":
                print(f"      {dim(name)}")
                continue
            hint = f"best guess {it.category}" if it.review else it.reason
            hint = (hint or "")[:60]
            print(f"      {meter(it.confidence)} {name:<45} {dim(VIA.get(it.via, it.via)):<8} {dim(hint)}")

    moving = sum(1 for i in plan.items if i.category and not i.review)
    review = sum(1 for i in plan.items if i.category and i.review)
    skipped = sum(1 for i in plan.items if i.category is None)
    print(f"\n  {green(moving)} to file · {yellow(review)} to review · {dim(skipped)} left in place"
          f"   {dim('model: ' + plan.model)}\n")


# ── commands ───────────────────────────────────────────────────────────────

def _load(args):
    cfg = load_config(Path(args.config).expanduser() if args.config else None)
    if getattr(args, "model", None):
        cfg.model.name = args.model
    if getattr(args, "mode", None):
        cfg.strategy.mode = args.mode
    if getattr(args, "dest", None):
        cfg.destination = Path(args.dest).expanduser()
    if getattr(args, "vision", False):
        cfg.model.vision = True
    return cfg


def _classifier(cfg):
    return Classifier(cfg, make_backend(cfg.model, cfg.strategy.embed_model))


def _make_plan(args, cfg) -> Plan:
    source = Path(args.source).expanduser()
    if not source.is_dir():
        sys.exit(red(f"Not a folder: {source}"))
    clf = _classifier(cfg)
    started = time.time()

    def progress(i, n, item):
        tag = "skip" if item.category is None else (cfg.review_folder if item.review else item.category)
        line = f"  [{i}/{n}] {item.name[:40]:<40} → {tag}"
        print(("\r" + line.ljust(90)) if COLOR else line, end="" if COLOR else "\n", flush=True)

    try:
        plan = build_plan(cfg, source, clf, progress)
    except BackendError as e:
        sys.exit(red(f"\n{e}\nRun `fclass doctor` to check your setup."))
    if COLOR:
        print("\r" + " " * 90 + "\r", end="")
    s = clf.stats
    print(dim(f"  {len(plan.items)} items in {time.time() - started:.1f}s · "
              f"rules {s['rule']} · cached {s['cache']} · fast {s['embed']} · llm {s['llm']}"))
    return plan


def cmd_plan(args):
    cfg = _load(args)
    plan = _make_plan(args, cfg)
    show_plan(plan)
    path = plan.save(Path(args.output) if args.output else None)
    print(f"  Saved plan → {path}\n  Apply it with: {bold('fclass apply')}\n")


def cmd_sort(args):
    cfg = _load(args)
    plan = _make_plan(args, cfg)
    show_plan(plan)
    if not any(i.category for i in plan.items):
        return
    if args.yes:
        answer = "y"
    else:
        answer = input(f"  Move them? {bold('[y]')}es · {bold('[r]')}eview each · {bold('[n]')}o  ").strip().lower()
    if answer.startswith("r"):
        review(plan, cfg)
        show_plan(plan)
        answer = input(f"  Move them now? {bold('[y]')}/n  ").strip().lower() or "y"
    if not answer.startswith("y"):
        path = plan.save()
        print(f"  Nothing moved. Plan saved → {path}")
        return
    _apply(plan)


def review(plan: Plan, cfg) -> None:
    """Walk through items; corrections become examples the classifier learns from."""
    paths = cfg.category_paths
    menu = "  ".join(f"{dim(str(i + 1))} {p}" for i, p in enumerate(paths))
    print(f"\n  {menu}\n  {dim('Enter = accept · number = change · s = skip · q = done')}\n")
    for it in plan.items:
        if it.category is None:
            continue
        where = f"{plan.review_folder} (guess {it.category})" if it.review else it.category
        ans = input(f"  {meter(it.confidence)} {it.name[:50]:<50} → {where}  ").strip().lower()
        if ans == "q":
            break
        if ans == "s":
            it.category, it.via = None, "user"
        elif ans.isdigit() and 1 <= int(ans) <= len(paths):
            new = paths[int(ans) - 1]
            if new != it.category or it.review:
                add_example(it.name, it.snippet, new)
            it.category, it.review, it.confidence, it.via, it.reason = new, False, 1.0, "user", "you chose this"
        elif ans == "" and it.review:
            it.review, it.via = False, "user"
            add_example(it.name, it.snippet, it.category)


def _apply(plan: Plan):
    journal, moved, errors = apply(plan)
    print(f"\n  {green('✓')} Moved {moved} item(s).", end="")
    if moved:
        print(f"  Changed your mind? {bold('fclass undo')}")
    else:
        print()
    for e in errors:
        print(f"  {red('✗')} {e}")


def cmd_apply(args):
    path = Path(args.plan) if args.plan else latest_plan()
    if not path or not path.exists():
        sys.exit(red("No plan found. Run `fclass plan <folder>` first."))
    plan = Plan.from_json(path.read_text(encoding="utf-8"))
    show_plan(plan)
    if args.yes or input(f"  Apply this plan? {bold('[y]')}/n  ").strip().lower() in ("", "y", "yes"):
        _apply(plan)
        path.rename(path.with_suffix(".applied"))


def cmd_undo(args):
    journal = latest_journal()
    if not journal:
        sys.exit("Nothing to undo.")
    restored, problems = undo(journal)
    print(f"  {green('↺')} Restored {restored} item(s) to where they were.")
    for p in problems:
        print(f"  {yellow('!')} {p}")


def cmd_teach(args):
    cfg = _load(args)
    if args.category not in cfg.category_paths:
        sys.exit(red(f"Unknown category {args.category!r}. See `fclass categories`."))
    path = Path(args.file).expanduser()
    p = preview_folder(path) if path.is_dir() else preview_file(path)
    add_example(path.name, p.text, args.category)
    print(f"  {green('✓')} Learned: files like {path.name} → {args.category}")


def cmd_init(args):
    path = Path(args.config).expanduser() if args.config else default_config_path()
    if write_default_config(path, force=args.force):
        print(f"  {green('✓')} Wrote {path}\n  Edit categories, rules and model there, then run `fclass doctor`.")
    else:
        print(f"  {path} already exists (use --force to overwrite).")


def cmd_categories(args):
    cfg = _load(args)
    last = None
    for cat in cfg.categories:
        top, _, sub = cat.path.partition("/")
        if top != last:
            print(cyan(f"{top}/"))
            last = top
        print(f"  {sub + '/' if sub else '':<16} {dim(cat.description)}")
    print(dim(f"\nconfig: {cfg.source_path or 'built-in defaults (run `fclass init` to customise)'}"))


RECOMMENDED = [
    ("qwen3:4b", "2.5 GB", "best accuracy per GB in our benchmark", 8),
    ("gemma3:4b", "3.3 GB", "also reads images (set vision = true)", 8),
    ("qwen3:1.7b", "1.4 GB", "for 8 GB machines; pair with hybrid mode", 6),
    ("qwen3:8b", "5.2 GB", "16 GB+ machines, most careful", 16),
]


def cmd_doctor(args):
    cfg = _load(args)
    ok = lambda b: green("✓") if b else red("✗")  # noqa: E731
    print(f"  {ok(True)} config      {cfg.source_path or 'built-in defaults'}")
    backend = make_backend(cfg.model, cfg.strategy.embed_model)
    st = backend.status()
    print(f"  {ok(st['ok'])} backend     {cfg.model.backend} at {cfg.model.url} — {st['detail']}")
    models = st["models"]

    def have(name):
        return any(m == name or m.split(":")[0] == name or m == f"{name}:latest" for m in models)

    need = []
    if cfg.strategy.mode != "embed":
        need.append(("llm", cfg.model.name))
    if cfg.strategy.mode != "llm":
        need.append(("embedder", cfg.strategy.embed_model))
    for role, name in need:
        present = have(name) if st["ok"] else False
        hint = "" if present or cfg.model.backend != "ollama" else dim(f"  → ollama pull {name}")
        print(f"  {ok(present)} {role:<11} {name}{hint}")
    pdf = pdf_support()
    print(f"  {ok(bool(pdf))} pdf         {pdf or 'not available'}"
          + ("" if pdf else dim("  → pip install 'fclass[pdf]'")))
    print(f"  {ok(True)} strategy    {cfg.strategy.mode}  ·  destination {cfg.destination}")
    try:
        ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, OSError, AttributeError):
        ram = 0
    if ram:
        print(dim(f"\n  This machine has {ram:.0f} GB RAM. Models that fit comfortably:"))
        for name, size, note, min_ram in RECOMMENDED:
            if ram >= min_ram:
                print(dim(f"    {name:<12} {size:>7}   {note}"))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="fclass", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"fclass {__version__}")
    ap.add_argument("--config", help="config file (default: ~/.config/fclass/config.toml)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def with_model_opts(p):
        p.add_argument("--model", help="override the LLM, e.g. qwen3:4b")
        p.add_argument("--mode", choices=["hybrid", "llm", "embed"], help="override the strategy")
        p.add_argument("--dest", help="override the destination folder")
        p.add_argument("--vision", action="store_true", help="send images to a vision model")
        return p

    p = with_model_opts(sub.add_parser("sort", help="plan, show, confirm and move"))
    p.add_argument("source")
    p.add_argument("-y", "--yes", action="store_true", help="don't ask, just move")
    p.set_defaults(fn=cmd_sort)

    p = with_model_opts(sub.add_parser("plan", help="plan only; nothing moves"))
    p.add_argument("source")
    p.add_argument("-o", "--output", help="where to save the plan JSON")
    p.set_defaults(fn=cmd_plan)

    p = sub.add_parser("apply", help="execute a saved plan (latest by default)")
    p.add_argument("plan", nargs="?")
    p.add_argument("-y", "--yes", action="store_true")
    p.set_defaults(fn=cmd_apply)

    sub.add_parser("undo", help="restore everything the last run moved").set_defaults(fn=cmd_undo)

    p = sub.add_parser("teach", help="tell fclass where a file belongs; it learns from this")
    p.add_argument("file")
    p.add_argument("category")
    p.set_defaults(fn=cmd_teach)

    p = sub.add_parser("init", help="write an editable config file")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_init)

    sub.add_parser("doctor", help="check models, backend and PDF support").set_defaults(fn=cmd_doctor)
    sub.add_parser("categories", help="show the category tree").set_defaults(fn=cmd_categories)

    args = ap.parse_args(argv)
    try:
        args.fn(args)
    except KeyboardInterrupt:
        print("\n  Stopped. Nothing further was moved.")
        sys.exit(130)
    except BrokenPipeError:  # e.g. `fclass categories | head`
        sys.stderr.close()


if __name__ == "__main__":
    main()
