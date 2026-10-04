"""fclass: an offline file organiser. Small models on your own computer read your
files, sort them into your categories, and ask you when they are not sure.

  fclass sort ~/Downloads          plan, show, ask about doubts, move
  fclass watch                     sort new downloads as they land
  fclass ask                       answer the questions fclass saved for you
  fclass discover ~/Documents      propose categories from a folder you already have
  fclass undo [--last N]           put things back
  fclass bench                     accuracy and speed on this machine
  fclass plan | apply | teach | categories | init | doctor
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from . import __version__
from .backends import BackendError, make_backend
from .classify import Classifier, add_example
from .config import (Category, OfflineError, add_categories, default_config_path, load_config, locality,
                     remove_category, replace_categories, state_dir, write_default_config)
from .extract import pdf_support, preview_file, preview_folder
from .plan import Item, Plan, apply, latest_journal, latest_plan, move, new_journal, undo
from .planner import build_plan
from . import questions

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


STYLE = {"q": yellow, "b": bold, "d": dim}


def style(text, role):
    return STYLE.get(role, str)(text)


def interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def meter(conf: float) -> str:
    filled = round(conf * 5)
    bar = "●" * filled + "○" * (5 - filled)
    return green(bar) if conf >= 0.75 else yellow(bar) if conf >= 0.5 else red(bar)


VIA = {"rule": "rule", "embed": "fast", "llm": "llm", "vision": "vision", "cache": "cached", "user": "you",
       "skip": "rule", "error": "error", "queued": "asked"}


def _tidy(reason: str) -> str:
    """'The document is a packing list for…' -> 'Packing list for…' (display only;
    the full sentence matters: it's the model's reasoning before it answers)."""
    r = re.sub(r"^(the (user|file|folder|document|content|image|picture)\b[^.]*?\b(is|are|contains|shows|appears to be|seems to be)\s+)",
               "", (reason or "").strip(), flags=re.I)
    r = re.sub(r"^(an?|the)\s+", "", r, flags=re.I)
    return r[:1].upper() + r[1:]


ASK_GROUP = "? needs your answer"
LEFT_GROUP = "· left in place"


def show_plan(plan: Plan) -> None:
    groups: dict[str, list] = {}
    for it in plan.items:
        if it.category is None:
            key = LEFT_GROUP
        elif it.review:
            key = plan.review_folder if plan.when_unsure == "review_folder" else (
                ASK_GROUP if plan.when_unsure == "ask" else LEFT_GROUP)
        else:
            key = it.category
        groups.setdefault(key, []).append(it)

    print()
    print(bold(f"  {plan.destination}/"))
    tail = (plan.review_folder, ASK_GROUP, LEFT_GROUP)
    order = sorted(k for k in groups if k not in tail) + [k for k in tail if k in groups]
    last_top = None
    for key in order:
        if key == LEFT_GROUP:
            print(f"\n  {dim(key)}")
        elif key == ASK_GROUP:
            print(f"\n  {yellow(key)}")
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
            if key == LEFT_GROUP and not it.review:
                print(f"      {dim(name)}")
                continue
            hint = f"best guess {it.category}" if it.review else _tidy(it.reason)
            print(f"      {meter(it.confidence)} {name:<45} {dim(VIA.get(it.via, it.via)):<8} {dim((hint or '')[:60])}")

    moving = sum(1 for i in plan.items if plan.target(i) is not None)
    unsure = sum(1 for i in plan.items if i.category and i.review)
    left = sum(1 for i in plan.items if i.category is None)
    word = {"ask": "to ask about", "review_folder": "to review", "leave": "unsure, left"}[plan.when_unsure]
    print(f"\n  {green(moving)} to file · {yellow(unsure)} {word} · {dim(left)} left in place"
          f"   {dim('models: ' + plan.model)}\n")


# ── setup ──────────────────────────────────────────────────────────────────

def _config_path(args) -> Path | None:
    return Path(args.config).expanduser() if args.config else None


def _load(args):
    cfg = load_config(_config_path(args))
    if getattr(args, "model", None):
        cfg.model.name = args.model
    if getattr(args, "mode", None):
        cfg.strategy.mode = args.mode
    if getattr(args, "dest", None):
        cfg.destination = Path(args.dest).expanduser()
    if getattr(args, "no_vision", False):
        cfg.model.vision = "off"
    return cfg


def _classifier(cfg) -> Classifier:
    try:
        return Classifier(cfg, make_backend(cfg.model, cfg.strategy.embed_model))
    except OfflineError as e:
        sys.exit(red(str(e)))


def _progress(cfg):
    def show(i, n, item):
        tag = "skip" if item.category is None else ("?" if item.review else item.category)
        line = f"  [{i}/{n}] {item.name[:40]:<40} → {tag}"
        print(("\r" + line.ljust(90)) if COLOR else line, end="" if COLOR else "\n", flush=True)
    return show


def _make_plan(args, cfg, clf) -> Plan:
    source = Path(args.source).expanduser()
    if not source.is_dir():
        sys.exit(red(f"Not a folder: {source}"))
    started = time.time()
    try:
        plan = build_plan(cfg, source, clf, _progress(cfg))
    except BackendError as e:
        sys.exit(red(f"\n{e}\nRun `fclass doctor` to check your setup."))
    if COLOR:
        print("\r" + " " * 90 + "\r", end="")
    s = clf.stats
    print(dim(f"  {len(plan.items)} items in {time.time() - started:.1f}s · rules {s['rule']} · cached {s['cache']}"
              f" · fast {s['embed']} · llm {s['llm']} · vision {s['vision']}"))
    return plan


# ── asking ─────────────────────────────────────────────────────────────────

def _terminal_asker(it, cats):
    return questions.ask(it, cats, ask_fn=input, say=print, style=style)


def _record(ans, it, cfg, args) -> str | None:
    """Act on an answer: create the category if new, remember the example. Returns the category, or None."""
    if ans.action == "new":
        add_categories([Category(ans.category, ans.description)], _config_path(args))
        cfg.categories.append(Category(ans.category, ans.description))
        print(f"    {green('+')} New category {bold(ans.category)} added to your config.")
    if ans.action in ("file", "new"):
        add_example(it.name, it.snippet, ans.category)
        return ans.category
    return None


def _answer_loop(items: list[Item], cfg, args, on_answer, asker=_terminal_asker) -> list[Item]:
    """Ask about each item. on_answer(item, category or None). Returns the items left unanswered."""
    print(f"  {yellow(str(len(items)))} {'item needs' if len(items) == 1 else 'items need'} your answer. "
          f"{dim('Answers are remembered, so similar files are sorted on their own next time.')}")
    for n, it in enumerate(items):
        ans = asker(it, cfg.category_paths)
        if ans.action == "quit":
            return items[n:]
        try:
            on_answer(it, _record(ans, it, cfg, args))
        except ValueError as e:
            print(f"    {red(str(e))}")
            return items[n:]
    return []


def _save_for_later(items: list[Item], destination) -> None:
    if items:
        added = questions.enqueue(items, destination)
        if added:
            print(f"  {yellow('?')} {added} question{'s' if added != 1 else ''} saved. "
                  f"Those files stay put until you run {bold('fclass ask')}.")


# ── commands ───────────────────────────────────────────────────────────────

def cmd_sort(args):
    cfg = _load(args)
    clf = _classifier(cfg)
    plan = _make_plan(args, cfg, clf)
    show_plan(plan)
    unsure = [i for i in plan.items if i.category and i.review]
    leftover: list[Item] = []
    if unsure and cfg.when_unsure == "ask":
        if interactive() and not args.yes:
            def settle(it, cat):
                if cat is None:
                    it.category, it.via = None, "user"
                else:
                    it.category, it.review, it.confidence, it.via, it.reason = cat, False, 1.0, "user", "you chose this"
            leftover = _answer_loop(unsure, cfg, args, settle)
            if len(leftover) < len(unsure):
                show_plan(plan)
        else:
            leftover = unsure
    if not any(plan.target(i) for i in plan.items):
        _save_for_later(leftover, plan.destination)
        return
    if not args.yes:
        if not interactive():  # no one to confirm with: never move without -y
            path = plan.save()
            print(f"  Nothing moved (not a terminal). Plan saved → {path}. Use `sort -y` or `fclass apply -y`.")
            return
        answer = input(f"  Move them? {bold('[y]')}es · {bold('[n]')}o  ").strip().lower() or "y"
        if not answer.startswith("y"):
            path = plan.save()
            print(f"  Nothing moved. Plan saved → {path}")
            return
    _apply(plan)
    _save_for_later(leftover, plan.destination)


def _apply(plan: Plan):
    journal, moved, errors = apply(plan)
    print(f"\n  {green('✓')} Moved {moved} item(s)." + (f"  Changed your mind? {bold('fclass undo')}" if moved else ""))
    for e in errors:
        print(f"  {red('✗')} {e}")


def cmd_plan(args):
    cfg = _load(args)
    plan = _make_plan(args, cfg, _classifier(cfg))
    show_plan(plan)
    path = plan.save(Path(args.output) if args.output else None)
    print(f"  Saved plan → {path}\n  Apply it with: {bold('fclass apply')}\n")


def cmd_apply(args):
    path = Path(args.plan) if args.plan else latest_plan()
    if not path or not path.exists():
        sys.exit(red("No plan found. Run `fclass plan <folder>` first."))
    plan = Plan.from_json(path.read_text(encoding="utf-8"))
    show_plan(plan)
    if args.yes or not interactive() or input(f"  Apply this plan? {bold('[y]')}/n  ").strip().lower() in ("", "y", "yes"):
        _apply(plan)
        if plan.when_unsure == "ask":
            _save_for_later([i for i in plan.items if i.category and i.review], plan.destination)
        path.rename(path.with_suffix(".applied"))


def cmd_ask(args):
    cfg = _load(args)
    items = questions.pending()
    if not items:
        print(f"  {green('✓')} No questions. Everything fclass has seen is sorted.")
        return
    if not interactive() and not args.dialog:
        for it in items:
            print(f"  ? {it.name}  (best guess {it.category})")
        print(dim("  Run `fclass ask` in a terminal, or `fclass ask --dialog` on a Mac."))
        return
    asker = (lambda it, cats: questions.ask_dialog(it, cats)) if args.dialog else _terminal_asker
    journal = new_journal("ask")

    def settle(it: Item, cat):
        questions.resolve(it.src)
        if cat is None:
            print(f"    {dim('Left where it is.')}")
            return
        src = Path(it.src)
        dest = Path(it.dest).expanduser() if it.dest else cfg.destination.expanduser()  # where it was headed
        if src.exists():
            dst = move(src, dest / cat / src.name, journal)
            print(f"    {green('✓')} → {dst.parent}")

    left = _answer_loop(items, cfg, args, settle, asker)
    if left:
        print(f"  {len(left)} question(s) kept for later.")


def cmd_undo(args):
    journal = latest_journal()
    if not journal:
        sys.exit("Nothing to undo.")
    restored, problems = undo(journal, args.last)
    print(f"  {green('↺')} Restored {restored} item(s) to where they were.")
    for p in problems:
        print(f"  {yellow('!')} {p}")


def cmd_watch(args):
    from .watch import Watcher, desktop_notify, service_file

    if args.print_service:
        where, body = service_file()
        print(f"# Save as {where}\n{body}")
        return
    cfg = _load(args)
    clf = _classifier(cfg)
    folders = [Path(f).expanduser() for f in args.folders] or cfg.watch.folders
    missing = [f for f in folders if not f.is_dir()]
    if missing:
        sys.exit(red("Not a folder: " + ", ".join(map(str, missing))))

    def log(event, item, detail):
        stamp = dim(time.strftime("%H:%M:%S"))
        name = item.name + ("/" if item.kind == "folder" else "")
        if event == "filed":
            print(f"  {stamp}  {green('✓')} {name} → {item.category}  {dim(VIA.get(item.via, item.via))}", flush=True)
        elif event == "asked":
            print(f"  {stamp}  {yellow('?')} {name}  {dim('not sure (best guess ' + str(item.category) + '); run fclass ask')}",
                  flush=True)
        elif event == "review":
            print(f"  {stamp}  {yellow('?')} {name} → {detail}", flush=True)
        else:
            print(f"  {stamp}  {dim('·')} {name}  {dim(detail)}", flush=True)

    ask_now = None
    if cfg.watch.ask_with == "dialog":
        from .native import IS_MAC
        if not IS_MAC:
            print(dim("  ask_with = \"dialog\" needs macOS; saving questions for `fclass ask` instead."))
        else:
            def ask_now(item):
                ans = questions.ask_dialog(item, cfg.category_paths, timeout=120)
                if ans.action == "quit":
                    return None  # no answer in time: save the question
                return _record(ans, item, cfg, args) or ""

    w = Watcher(cfg, clf, folders, log, include_existing=args.existing or args.once,
                notify=desktop_notify if cfg.watch.notify else None, ask_now=ask_now)
    if args.once:
        n = w.sweep()
        print(dim(f"  Handled {n} item(s)."))
        return
    print(f"  Watching {', '.join(str(f) for f in folders)} {dim('(Ctrl-C to stop)')}")
    print(dim(f"  New items are handled once unchanged for {cfg.watch.settle_seconds:g}s. "
              f"Undo the latest with `fclass undo --last 1`."))
    try:
        w.run()
    except BackendError as e:
        sys.exit(red(f"{e}\nRun `fclass doctor` to check your setup."))


def cmd_discover(args):
    from .discover import discover

    cfg = _load(args)
    clf = _classifier(cfg)
    root = Path(args.folder).expanduser()
    if not root.is_dir():
        sys.exit(red(f"Not a folder: {root}"))
    started = time.time()

    def progress(stage, i, n):
        line = {"reading": f"  Reading {i}/{n} files", "grouping": f"  Grouping {n} files by meaning",
                "naming": f"  Naming group {i}/{n}"}[stage]
        print(("\r" + line.ljust(60)) if COLOR else line, end="" if COLOR else "\n", flush=True)

    try:
        prop = discover(cfg, clf, root, limit=args.limit, fresh=args.fresh, progress=progress)
    except BackendError as e:
        sys.exit(red(f"\n{e}\nRun `fclass doctor` to check your setup."))
    if COLOR:
        print("\r" + " " * 60 + "\r", end="")
    total = sum(len(g.files) for g in prop.groups) + len(prop.ungrouped)
    print(dim(f"  {total} files read in {time.time() - started:.0f}s"))
    print(f"\n  {bold('Proposed categories for ' + str(root))}\n")
    last_top = None
    for g in sorted(prop.groups, key=lambda g: g.path):
        top, _, sub = g.path.partition("/")
        if top != last_top:
            print(f"  {cyan(top + '/')}")
            last_top = top
        mark = dim("✓ yours") if not g.is_new else green("+ new  ")
        print(f"    {mark} {(sub or top) + '/':<26} {len(g.files):>3} files   {dim(g.description[:70])}")
        print(f"            {dim('e.g. ' + ', '.join(p.name for p, _ in g.central(3))[:80])}")
    if prop.ungrouped:
        print(f"\n  {dim(f'· {len(prop.ungrouped)} files did not group with anything')}")
    saved = state_dir() / f"discover-{time.strftime('%Y%m%d-%H%M%S')}.json"
    saved.write_text(json.dumps([{"path": g.path, "description": g.description, "existing": g.existing,
                                  "files": [str(f) for f in g.files]} for g in prop.groups], indent=1),
                     encoding="utf-8")
    new = [g for g in prop.groups if g.is_new]
    if not prop.groups:
        return
    print()
    if args.yes:
        choice = "y"
    elif not interactive():
        print(dim(f"  Proposal saved → {saved}. Run in a terminal to accept it."))
        return
    else:
        what = f"Add the {len(new)} new categor{'y' if len(new) == 1 else 'ies'}" if new else "Nothing new to add"
        choice = input(f"  {what}? {bold('[y]')}es · {bold('[e]')}dit one by one · "
                       f"{bold('[r]')}eplace all my categories with these · {bold('[n]')}o  ").strip().lower() or "y"
    if choice.startswith("e"):
        kept = []
        for g in new:
            ans = input(f"    {green('+')} {g.path}  {dim(g.description[:60])}\n      Enter = keep · type a new path · s = skip  › ").strip()
            if ans.lower() == "s":
                continue
            if ans:
                g.path = ans.strip("/")
            kept.append(g)
        new, choice = kept, "y"
    if choice.startswith("y") and new:
        add_categories([Category(g.path, g.description) for g in new], _config_path(args))
        _teach_central(new)
        print(f"  {green('✓')} Added {len(new)} categories to your config, each with 3 example files.")
    elif choice.startswith("r"):
        replace_categories([Category(g.path, g.description) for g in prop.groups], _config_path(args))
        _teach_central(prop.groups)
        print(f"  {green('✓')} Your categories are now these {len(prop.groups)}. "
              f"The old config was kept as config.toml.bak.")
    else:
        print(dim(f"  Nothing changed. Proposal saved → {saved}"))


def cmd_bench(args):
    from . import bench

    cfg = _load(args)
    try:
        from .config import check_offline
        check_offline(cfg.model)
    except OfflineError as e:
        sys.exit(red(str(e)))
    docs = bench.items(args.quick)
    modes = ["embed", "hybrid"] + (["llm"] if args.llm else [])
    print(f"  Benchmark: {len(docs)} documents, 12 categories, filenames that don't help. "
          f"{dim('Your cache and examples are not used.')}")

    def progress(mode, i, n):
        line = f"  {mode:<7} {i}/{n}"
        print(("\r" + line.ljust(40)) if COLOR else line, end="" if COLOR else "\n", flush=True)

    results = []
    for mode in modes:
        try:
            r = bench.run_mode(cfg, mode, docs, progress if COLOR else None)
        except BackendError as e:
            sys.exit(red(f"\n{e}\nRun `fclass doctor` to check your setup."))
        results.append(r)
        if COLOR:
            print("\r" + " " * 40 + "\r", end="")
        print(f"  {mode:<7} {r.models:<34} {green(f'{r.accuracy:.1%}'):>8}  {r.per_file:6.2f} s/file  "
              f"{dim(f'{r.llm_calls} LLM calls')}")
    report = bench.to_json(results, args.quick)
    out = state_dir() / f"bench-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\n{bench.markdown(report)}")
    print(dim(f"  Saved → {out}. Sharing the table above in an issue helps others choose models."))


def _teach_central(groups) -> None:
    for g in groups:
        for path, snippet in g.central(3):
            add_example(path.name, snippet, g.path)


def cmd_teach(args):
    cfg = _load(args)
    if args.category not in cfg.category_paths:
        sys.exit(red(f"Unknown category {args.category!r}. See `fclass categories`."))
    path = Path(args.file).expanduser()
    if not path.exists():
        sys.exit(red(f"No such file: {path}"))
    p = preview_folder(path) if path.is_dir() else preview_file(path)
    add_example(path.name, p.text, args.category)
    print(f"  {green('✓')} Learned: files like {path.name} → {args.category}")


def cmd_categories(args):
    if args.action == "add":
        if not args.path or not args.description:
            sys.exit(red('Usage: fclass categories add "Work/Payslips" "Monthly salary slips"'))
        try:
            where = add_categories([Category(args.path, args.description)], _config_path(args))
        except ValueError as e:
            sys.exit(red(str(e)))
        print(f"  {green('+')} {args.path.strip('/')} added to {where}")
        return
    if args.action == "remove":
        try:
            where = remove_category(args.path or "", _config_path(args))
        except ValueError as e:
            sys.exit(red(str(e)))
        print(f"  {green('−')} {args.path.strip('/')} removed from {where} (files already sorted are not touched)")
        return
    cfg = _load(args)
    last = None
    for cat in cfg.categories:
        top, _, sub = cat.path.partition("/")
        if top != last:
            print(cyan(f"{top}/"))
            last = top
        print(f"  {(sub + '/') if sub else '':<18} {dim(cat.description)}")
    print(dim(f"\nconfig: {cfg.source_path or 'built-in defaults (run `fclass init` to customise)'}"))


def cmd_init(args):
    path = _config_path(args) or default_config_path()
    if write_default_config(path, force=args.force):
        print(f"  {green('✓')} Wrote {path}\n  Edit categories, rules and models there, then run `fclass doctor`.")
    else:
        print(f"  {path} already exists (use --force to overwrite).")


RECOMMENDED = [
    ("qwen3:4b", "2.5 GB", "reads documents; best accuracy per GB in our benchmark", 8),
    ("gemma3:4b", "3.3 GB", "reads photos, screenshots and scanned PDFs", 8),
    ("embeddinggemma", "0.6 GB", "the fast first pass", 4),
    ("qwen3:1.7b", "1.4 GB", "for 8 GB machines, in hybrid mode", 6),
]


def cmd_doctor(args):
    cfg = _load(args)
    ok = lambda b: green("✓") if b else red("✗")  # noqa: E731
    soft = lambda b: green("✓") if b else yellow("○")  # noqa: E731
    print(f"  {ok(True)} config      {cfg.source_path or 'built-in defaults'}")
    where = locality(cfg.model.url)
    if where == "remote" and not cfg.model.allow_remote:
        print(f"  {ok(False)} offline     {cfg.model.url} is on the internet; fclass refuses it (allow_remote = false)")
        return
    label = {"device": "on this computer", "network": "on your local network",
             "remote": "ON THE INTERNET (allow_remote = true)"}[where]
    print(f"  {ok(where != 'remote')} offline     models run {label}")
    backend = make_backend(cfg.model, cfg.strategy.embed_model)
    st = backend.status()
    print(f"  {ok(st['ok'])} backend     {cfg.model.backend} at {cfg.model.url}: {st['detail']}")

    def have(name):
        return st["ok"] and any(m == name or m.split(":")[0] == name or m == f"{name}:latest" for m in st["models"])

    hint = lambda name: dim(f"  → ollama pull {name}") if cfg.model.backend == "ollama" else ""  # noqa: E731
    if cfg.strategy.mode != "embed":
        print(f"  {ok(have(cfg.model.name))} documents   {cfg.model.name}{'' if have(cfg.model.name) else hint(cfg.model.name)}")
    if cfg.strategy.mode != "llm":
        e = cfg.strategy.embed_model
        print(f"  {ok(have(e))} fast pass   {e}{'' if have(e) else hint(e)}")
    v = cfg.model.vision_model
    if cfg.model.vision == "off":
        print(f"  {soft(False)} pictures    vision off: photos and scans are sorted by name and type, or asked about")
    else:
        print(f"  {soft(have(v))} pictures    {v}" + ("" if have(v) else
              dim(f"  not installed: photos and scans will be asked about → ollama pull {v}")))
    pdf = pdf_support()
    print(f"  {ok(bool(pdf))} pdf         {pdf or 'not available'}" + ("" if pdf else dim("  → pip install pypdf")))
    q = len(questions.pending())
    print(f"  {soft(q == 0)} questions   {q} waiting" + (dim("  → fclass ask") if q else ""))
    print(f"  {ok(True)} sorting     {cfg.strategy.mode} into {cfg.destination} · when unsure: {cfg.when_unsure}")
    print(f"  {ok(True)} watching    {', '.join(str(f) for f in cfg.watch.folders)} (with `fclass watch`)")
    try:
        ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, OSError, AttributeError):
        ram = 0
    if ram:
        print(dim(f"\n  This machine has {ram:.0f} GB RAM. Models that fit comfortably:"))
        for name, size, note, min_ram in RECOMMENDED:
            if ram >= min_ram:
                print(dim(f"    {name:<15} {size:>7}   {note}"))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="fclass", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"fclass {__version__}")
    ap.add_argument("--config", help="config file (default: ~/.config/fclass/config.toml)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def with_model_opts(p):
        p.add_argument("--model", help="override the document model, e.g. qwen3:4b")
        p.add_argument("--mode", choices=["hybrid", "llm", "embed"], help="override the strategy")
        p.add_argument("--dest", help="override the destination folder")
        p.add_argument("--no-vision", action="store_true", help="don't send pictures to a vision model")
        return p

    p = with_model_opts(sub.add_parser("sort", help="plan, show, ask about doubts, move"))
    p.add_argument("source")
    p.add_argument("-y", "--yes", action="store_true", help="don't ask now: move what's certain, save questions for later")
    p.set_defaults(fn=cmd_sort)

    p = with_model_opts(sub.add_parser("watch", help="sort new arrivals as they land"))
    p.add_argument("folders", nargs="*", help="folders to watch (default: [watch] folders in your config)")
    p.add_argument("--existing", action="store_true", help="also sort what is already there")
    p.add_argument("--once", action="store_true", help="sort what is there now and exit")
    p.add_argument("--print-service", action="store_true", help="print a login service to run watch in the background")
    p.set_defaults(fn=cmd_watch)

    p = sub.add_parser("ask", help="answer the questions fclass saved for you")
    p.add_argument("--dialog", action="store_true", help="ask with native macOS dialogs instead of the terminal")
    p.set_defaults(fn=cmd_ask)

    p = with_model_opts(sub.add_parser("discover", help="propose categories from a folder you already have"))
    p.add_argument("folder")
    p.add_argument("--limit", type=int, default=300, help="read at most this many files (default 300)")
    p.add_argument("--fresh", action="store_true", help="ignore your current categories and start from scratch")
    p.add_argument("-y", "--yes", action="store_true", help="add the proposed new categories without asking")
    p.set_defaults(fn=cmd_discover)

    p = with_model_opts(sub.add_parser("plan", help="plan only; nothing moves"))
    p.add_argument("source")
    p.add_argument("-o", "--output", help="where to save the plan JSON")
    p.set_defaults(fn=cmd_plan)

    p = sub.add_parser("apply", help="execute a saved plan (latest by default)")
    p.add_argument("plan", nargs="?")
    p.add_argument("-y", "--yes", action="store_true")
    p.set_defaults(fn=cmd_apply)

    p = sub.add_parser("undo", help="put back what the last run (or watch) moved")
    p.add_argument("--last", type=int, metavar="N", help="only the last N moves")
    p.set_defaults(fn=cmd_undo)

    p = sub.add_parser("teach", help="tell fclass where a file belongs; it learns from this")
    p.add_argument("file")
    p.add_argument("category")
    p.set_defaults(fn=cmd_teach)

    p = sub.add_parser("categories", help="show, add or remove categories")
    p.add_argument("action", nargs="?", choices=["list", "add", "remove"], default="list")
    p.add_argument("path", nargs="?")
    p.add_argument("description", nargs="?")
    p.set_defaults(fn=cmd_categories)

    p = sub.add_parser("init", help="write an editable config file")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_init)

    sub.add_parser("doctor", help="check models, offline status, PDF and picture support").set_defaults(fn=cmd_doctor)

    p = with_model_opts(sub.add_parser("bench", help="measure accuracy and speed on this machine"))
    p.add_argument("--quick", action="store_true", help="24 documents instead of 60")
    p.add_argument("--llm", action="store_true", help="also time the LLM reading every file (slowest)")
    p.set_defaults(fn=cmd_bench)

    args = ap.parse_args(argv)
    try:
        args.fn(args)
    except KeyboardInterrupt:
        print("\n  Stopped. Nothing further was moved.")
        sys.exit(130)
    except BrokenPipeError:  # e.g. `fclass categories | head`
        sys.stderr.close()
    except OfflineError as e:
        sys.exit(red(str(e)))


if __name__ == "__main__":
    main()
