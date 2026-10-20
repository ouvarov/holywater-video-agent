"""uv run video "idea" [--line ...] [--lang uk|en] [--profile dev|premium] [--no-interactive]"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from hwva import manifest as mf
from hwva import pipeline
from hwva.context import Context
from hwva.providers import registry
from hwva.providers.base import ProviderError
from hwva.schemas import Manifest, StageStatus
from hwva.strategy import preflight

EXIT_OK, EXIT_FAILED, EXIT_USAGE, EXIT_NOTHING_TO_RUN = 0, 1, 2, 3


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="video", description=__doc__)
    p.add_argument("idea", nargs="?", help="1-3 sentences; put the dialogue line in quotes")
    p.add_argument("--line", help="exact dialogue line, numbers in words (skips the question)")
    p.add_argument("--lang", choices=["uk", "en"], help="default: detected from the line")
    p.add_argument("--profile", default="dev", help="config/profiles.yaml (default: dev, free)")
    p.add_argument("--no-interactive", action="store_true", help="never ask questions")
    p.add_argument("--resume", metavar="RUN_ID", help="continue a failed run (other flags ignored)")
    return p.parse_args(argv)


def load_env() -> None:
    """Keys from ./.env (or $HWVA_DOTENV). A variable already set in the environment wins."""
    load_dotenv(Path(os.environ.get("HWVA_DOTENV", ".env")), override=False)


def main(argv: list[str] | None = None) -> int:
    load_env()
    args = parse_args(argv)
    try:
        if args.resume:
            manifest = mf.load(args.resume)
            providers = registry.build(manifest.profile, refs=manifest.models)  # pinned at start
            try:
                max_cost_usd = registry.load_max_cost_usd(manifest.profile)  # not pinned: may rise
            except KeyError:  # profile gone from the yaml: the pinned refs above still run
                max_cost_usd = registry.DEFAULT_MAX_COST_USD
        elif args.idea:
            max_cost_usd = registry.load_max_cost_usd(args.profile)
            providers = registry.build(args.profile)
            # before anything is created or paid for: can this video model follow our audio?
            chosen = preflight(providers.video.capabilities, providers.refs["video"].provider)
            params = {k: v for k, v in {"line": args.line, "lang": args.lang}.items() if v}
            manifest = mf.new_run(args.idea, args.profile, providers.refs, params)
            manifest.strategy = chosen
            mf.save(manifest)
        else:
            print("Give an idea or --resume RUN_ID", file=sys.stderr)
            return EXIT_USAGE
    # ProviderError: a provider refused its config at build (e.g. a plain-HTTP base URL)
    except (FileNotFoundError, KeyError, ValueError, NotImplementedError, ProviderError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE

    if not pipeline.STAGES:
        print(f"run {manifest.run_id}: no stages implemented yet", file=sys.stderr)
        return EXIT_NOTHING_TO_RUN

    ctx = Context(
        manifest=manifest,
        providers=providers,
        run_dir=mf.run_dir(manifest.run_id),
        interactive=not args.no_interactive and sys.stdin.isatty(),
        max_cost_usd=max_cost_usd,
    )
    result = pipeline.run(ctx)
    failed = [(n, s.error) for n, s in result.stages.items() if s.status is StageStatus.FAILED]
    code = EXIT_OK
    if failed:
        name, error = failed[0]
        print(f"run {result.run_id}: FAILED at {name}: {error}", file=sys.stderr)
        print(f"fix and continue: uv run video --resume {result.run_id}", file=sys.stderr)
        code = EXIT_FAILED
    else:
        print(f"run {result.run_id}: done -> {mf.run_dir(result.run_id)}")
    warning = _sync_warning(result)
    if warning:  # soft (specs/sync.md "Record"): once, at the very end, never changes the exit
        print(f"warning: {warning}", file=sys.stderr)
    return code


def _sync_warning(manifest: Manifest) -> str | None:
    record = manifest.stages.get("shot_1")
    sync = record.data.get("sync") if record is not None else None
    warning = sync.get("warning") if isinstance(sync, dict) else None
    return " ".join(warning.split()) if isinstance(warning, str) and warning.strip() else None


if __name__ == "__main__":
    raise SystemExit(main())
