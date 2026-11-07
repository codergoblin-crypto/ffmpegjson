#!/usr/bin/env python3
"""
edit_plan.py - parse and validate a Video Edit Plan JSON file.

Three layers of checking:
  1. JSON syntax           (clear line/column error messages)
  2. JSON Schema (2020-12) (types, enums, ranges, required keys)
  3. Semantic rules        (things a schema can't express: end > start,
                            effects fitting inside the clip, unique ids, ...)

On success it returns typed dataclasses with schema defaults applied, ready
for an FFmpeg command builder.

Usage:
    python edit_plan.py                 # run it, then paste your JSON into the terminal
    python edit_plan.py plan.json       # or validate a file directly
    python edit_plan.py --check-files   # also verify source/music/sfx files exist

Exit code: 0 = valid (warnings allowed), 1 = invalid.

Time convention assumed by the semantic checks:
    source_start / source_end  -> seconds in the SOURCE video
    effect / caption / sfx times -> seconds in the OUTPUT clip, i.e. AFTER speed
    clip duration = (source_end - source_start) / speed
(The schema doesn't specify this; change `Clip.duration` if you decide otherwise.)
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from jsonschema import Draft202012Validator

DEFAULT_SCHEMA = Path(__file__).with_name("video_edit_plan.schema.json")
EPS = 1e-6


# --------------------------------------------------------------------------
# Result / error types
# --------------------------------------------------------------------------
class PlanError(Exception):
    """Raised when a plan is invalid. `.errors` holds every problem found."""

    def __init__(self, errors: list[str], warnings: Optional[list[str]] = None):
        self.errors = errors
        self.warnings = warnings or []
        super().__init__("\n".join(errors))


# --------------------------------------------------------------------------
# Typed model (defaults from the schema are applied here)
# --------------------------------------------------------------------------
@dataclass
class Output:
    file: str
    aspect_ratio: str
    width: int
    height: int
    fps: float
    video_codec: str
    audio_codec: str
    crf: int = 18
    preset: str = "medium"


@dataclass
class Audio:
    preserve_original: bool
    music_volume: float
    duck_under_dialogue: bool
    music_file: Optional[str] = None
    duck_volume: Optional[float] = None
    fade_in: float = 0.0
    fade_out: float = 0.0


@dataclass
class EditingRules:
    remove_dead_air: bool = True
    hard_cuts: bool = True
    prioritize_original_audio: bool = True
    effects_only_on_impact: bool = True
    vertical_crop: Optional[str] = None
    do_not_force_exact_duration: bool = True


@dataclass
class Zoom:
    start: float
    end: float
    duration: float
    start_at: float = 0.0


@dataclass
class Shake:
    start_at: float
    duration: float
    intensity: float


@dataclass
class Flash:
    start_at: float
    duration: float
    color: str


@dataclass
class FreezeFrame:
    at: float
    duration: float


@dataclass
class Effects:
    zoom: Optional[Zoom] = None
    shake: Optional[Shake] = None
    flash: Optional[Flash] = None
    freeze_frame: Optional[FreezeFrame] = None
    saturation: Optional[float] = None
    contrast: Optional[float] = None


@dataclass
class Caption:
    text: str
    start: float
    end: float
    position: str
    font_size: int
    color: str
    outline_color: str
    animation: str = "pop"


@dataclass
class SoundEffect:
    file: str
    at: float
    volume: float


@dataclass
class Clip:
    id: int
    source_start: float
    source_end: float
    description: str
    effects: Effects
    captions: list[Caption]
    importance: int = 5
    speed: float = 1.0
    crop: str = "auto"
    keep_original_audio: bool = True
    sound_effects: list[SoundEffect] = field(default_factory=list)

    @property
    def duration(self) -> float:
        """Output duration in seconds (after speed change)."""
        return (self.source_end - self.source_start) / self.speed


@dataclass
class EditPlan:
    version: str
    project_name: str
    source: str
    output: Output
    audio: Audio
    editing_rules: EditingRules
    clips: list[Clip]

    @property
    def total_duration(self) -> float:
        return sum(c.duration for c in self.clips)


# --------------------------------------------------------------------------
# Layer 1: JSON loading
# --------------------------------------------------------------------------
def load_json(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        raise PlanError([f"Cannot read {path}: {e}"])
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise PlanError([f"Invalid JSON in {path.name} at line {e.lineno}, column {e.colno}: {e.msg}"])


# --------------------------------------------------------------------------
# Layer 2: schema validation
# --------------------------------------------------------------------------
def _fmt_path(err) -> str:
    parts = []
    for p in err.absolute_path:
        parts.append(f"[{p}]" if isinstance(p, int) else f".{p}")
    return "".join(parts).lstrip(".") or "<root>"


def schema_errors(data: Any, schema: dict) -> list[str]:
    validator = Draft202012Validator(schema)
    out = []
    for err in sorted(validator.iter_errors(data), key=lambda e: list(map(str, e.absolute_path))):
        out.append(f"{_fmt_path(err)}: {err.message}")
    return out


# --------------------------------------------------------------------------
# Build typed model from (already schema-valid) data
# --------------------------------------------------------------------------
def _build_effects(d: dict) -> Effects:
    return Effects(
        zoom=Zoom(**d["zoom"]) if "zoom" in d else None,
        shake=Shake(**d["shake"]) if "shake" in d else None,
        flash=Flash(**d["flash"]) if "flash" in d else None,
        freeze_frame=FreezeFrame(**d["freeze_frame"]) if "freeze_frame" in d else None,
        saturation=d.get("saturation"),
        contrast=d.get("contrast"),
    )


def build_plan(d: dict) -> EditPlan:
    clips = []
    for c in d["clips"]:
        clips.append(
            Clip(
                id=c["id"],
                source_start=c["source_start"],
                source_end=c["source_end"],
                description=c["description"],
                effects=_build_effects(c["effects"]),
                captions=[Caption(**cap) for cap in c["captions"]],
                importance=c.get("importance", 5),
                speed=c.get("speed", 1.0),
                crop=c.get("crop", "auto"),
                keep_original_audio=c.get("keep_original_audio", True),
                sound_effects=[SoundEffect(**s) for s in c.get("sound_effects", [])],
            )
        )
    return EditPlan(
        version=d["version"],
        project_name=d["project"]["name"],
        source=d["project"]["source"],
        output=Output(**d["output"]),
        audio=Audio(**d["audio"]),
        editing_rules=EditingRules(**d["editing_rules"]),
        clips=clips,
    )


# --------------------------------------------------------------------------
# Layer 3: semantic validation
# --------------------------------------------------------------------------
_RATIOS = {"9:16": 9 / 16, "16:9": 16 / 9, "1:1": 1.0}


def semantic_check(plan: EditPlan, check_files: bool = False,
                   base_dir: Optional[Path] = None) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    base = base_dir or Path.cwd()

    # ---- output ----
    o = plan.output
    if o.aspect_ratio in _RATIOS:
        expected = _RATIOS[o.aspect_ratio]
        actual = o.width / o.height
        if abs(actual - expected) / expected > 0.01:
            errors.append(
                f"output: aspect_ratio {o.aspect_ratio} conflicts with {o.width}x{o.height} "
                f"(ratio {actual:.3f}, expected {expected:.3f})"
            )
    if o.video_codec == "libx264" and (o.width % 2 or o.height % 2):
        errors.append(f"output: width/height must be even for yuv420p H.264/H.265 ({o.width}x{o.height})")

    # ---- audio ----
    a = plan.audio
    if a.duck_under_dialogue and a.duck_volume is None:
        warnings.append("audio: duck_under_dialogue is true but duck_volume is not set (pick a default in your code)")
    if not a.duck_under_dialogue and a.duck_volume is not None:
        warnings.append("audio: duck_volume is set but duck_under_dialogue is false (ignored)")
    if a.music_file is None and a.music_volume > 0:
        warnings.append("audio: music_volume > 0 but music_file is null (no music will be mixed)")
    if a.music_file is None and a.duck_under_dialogue:
        warnings.append("audio: duck_under_dialogue has no effect without a music_file")
    if not a.preserve_original and a.music_file is None:
        warnings.append("audio: preserve_original is false and no music_file -> output will be silent")
    total = plan.total_duration
    if a.fade_in + a.fade_out > total + EPS:
        errors.append(f"audio: fade_in + fade_out ({a.fade_in + a.fade_out:.2f}s) exceeds total duration ({total:.2f}s)")

    # ---- editing rules vs output ----
    vc = plan.editing_rules.vertical_crop
    if vc and o.aspect_ratio != "9:16" and vc != "original":
        warnings.append(f"editing_rules.vertical_crop is '{vc}' but output aspect_ratio is {o.aspect_ratio}")

    # ---- clips ----
    seen_ids: set[int] = set()
    for idx, c in enumerate(plan.clips):
        p = f"clips[{idx}] (id={c.id})"
        D = c.duration

        if c.id in seen_ids:
            errors.append(f"{p}: duplicate clip id")
        seen_ids.add(c.id)

        if c.source_end <= c.source_start:
            errors.append(f"{p}: source_end ({c.source_end}) must be greater than source_start ({c.source_start})")
            continue  # duration is meaningless; skip timing checks for this clip

        if D < 0.2:
            warnings.append(f"{p}: very short clip ({D:.2f}s)")

        e = c.effects
        if e.zoom:
            end_t = e.zoom.start_at + e.zoom.duration
            if end_t > D + EPS:
                errors.append(f"{p}: zoom ends at {end_t:.2f}s but clip is {D:.2f}s long")
            if e.zoom.start == e.zoom.end:
                warnings.append(f"{p}: zoom start == end ({e.zoom.start}); effect does nothing")
        if e.shake and e.shake.start_at + e.shake.duration > D + EPS:
            errors.append(f"{p}: shake ends at {e.shake.start_at + e.shake.duration:.2f}s but clip is {D:.2f}s long")
        if e.flash and e.flash.start_at + e.flash.duration > D + EPS:
            errors.append(f"{p}: flash ends at {e.flash.start_at + e.flash.duration:.2f}s but clip is {D:.2f}s long")
        if e.freeze_frame and e.freeze_frame.at >= D:
            errors.append(f"{p}: freeze_frame.at ({e.freeze_frame.at}s) is outside the clip ({D:.2f}s)")

        # captions
        for ci, cap in enumerate(c.captions):
            cp = f"{p}.captions[{ci}]"
            if cap.end <= cap.start:
                errors.append(f"{cp}: end ({cap.end}) must be greater than start ({cap.start})")
            if cap.end > D + EPS:
                errors.append(f"{cp}: ends at {cap.end}s but clip is {D:.2f}s long")
        caps = sorted(
            ((cap.start, cap.end, i, cap.position) for i, cap in enumerate(c.captions)),
        )
        for (s1, e1, i1, pos1), (s2, e2, i2, pos2) in zip(caps, caps[1:]):
            if s2 < e1 - EPS and pos1 == pos2:
                warnings.append(f"{p}: captions[{i1}] and captions[{i2}] overlap at the same position '{pos1}'")

        # sound effects
        for si, sfx in enumerate(c.sound_effects):
            if sfx.at >= D:
                errors.append(f"{p}.sound_effects[{si}]: at={sfx.at}s is outside the clip ({D:.2f}s)")

        # crop consistency
        if c.crop in _RATIOS and o.aspect_ratio in _RATIOS and c.crop != o.aspect_ratio:
            warnings.append(f"{p}: crop '{c.crop}' differs from output aspect_ratio '{o.aspect_ratio}'")

        # impact-only effects rule
        if plan.editing_rules.effects_only_on_impact and c.importance <= 3:
            used = [n for n in ("zoom", "shake", "flash", "freeze_frame") if getattr(e, n)]
            if used:
                warnings.append(
                    f"{p}: effects_only_on_impact is on but low-importance clip ({c.importance}) uses {', '.join(used)}"
                )

        if not c.keep_original_audio and not plan.audio.preserve_original:
            pass  # both off is legal; the global silent warning above covers the music-less case

    # overlapping source ranges (informational)
    ranges = sorted((c.source_start, c.source_end, c.id) for c in plan.clips if c.source_end > c.source_start)
    for (s1, e1, id1), (s2, e2, id2) in zip(ranges, ranges[1:]):
        if s2 < e1 - EPS:
            warnings.append(f"clips {id1} and {id2} overlap in the source ({s2}s-{min(e1, e2)}s)")

    # ---- file existence (opt-in) ----
    if check_files:
        def need(path: str, what: str):
            if not (base / path).exists():
                errors.append(f"{what}: file not found: {path}")

        need(plan.source, "project.source")
        if a.music_file:
            need(a.music_file, "audio.music_file")
        for c in plan.clips:
            for s in c.sound_effects:
                need(s.file, f"clip {c.id} sound effect")

    return errors, warnings


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def validate_data(data: Any, schema: dict, check_files: bool = False,
                  base_dir: Optional[Path] = None) -> tuple[EditPlan, list[str]]:
    """Validate already-parsed JSON. Returns (EditPlan, warnings) or raises PlanError."""
    errs = schema_errors(data, schema)
    if errs:  # don't attempt semantic checks on structurally invalid data
        raise PlanError(errs)
    plan = build_plan(data)
    errs, warns = semantic_check(plan, check_files=check_files, base_dir=base_dir)
    if errs:
        raise PlanError(errs, warns)
    return plan, warns


def load_plan(plan_path: str | Path, schema_path: str | Path = DEFAULT_SCHEMA,
              check_files: bool = False) -> tuple[EditPlan, list[str]]:
    """Parse + validate a plan file."""
    plan_path = Path(plan_path)
    schema = load_json(Path(schema_path))
    data = load_json(plan_path)
    return validate_data(data, schema, check_files, plan_path.parent)


def load_plan_from_text(text: str, schema_path: str | Path = DEFAULT_SCHEMA,
                        check_files: bool = False) -> tuple[EditPlan, list[str]]:
    """Parse + validate JSON given as a string (e.g. pasted into the terminal)."""
    schema = load_json(Path(schema_path))
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise PlanError([f"Invalid JSON at line {e.lineno}, column {e.colno}: {e.msg}"])
    return validate_data(data, schema, check_files, Path.cwd())


# --------------------------------------------------------------------------
# CLI: run the program, paste JSON, get the result
# --------------------------------------------------------------------------
def report(run) -> bool:
    """`run` is a zero-arg callable that returns (plan, warnings) or raises PlanError."""
    try:
        plan, warnings = run()
    except PlanError as e:
        print(f"\n\u274c INVALID: {len(e.errors)} error(s)")
        for msg in e.errors:
            print(f"  [error]   {msg}")
        for msg in e.warnings:
            print(f"  [warning] {msg}")
        return False
    print(f"\n\u2705 VALID: '{plan.project_name}' - {len(plan.clips)} clip(s), "
          f"{plan.total_duration:.2f}s total, {plan.output.width}x{plan.output.height}@{plan.output.fps}fps")
    for msg in warnings:
        print(f"  [warning] {msg}")
    return True


def read_pasted_json() -> Optional[str]:
    """
    Read pasted text from the terminal. Returns as soon as the text is complete,
    valid JSON, so you don't need to press anything extra. If the JSON is broken
    (so it can never "complete"), press Enter on two empty lines, or type END,
    to validate what you pasted anyway. Type q to quit. Returns None on quit/EOF.
    """
    lines: list[str] = []
    blanks = 0
    while True:
        try:
            line = input()
        except EOFError:
            return "\n".join(lines) if any(l.strip() for l in lines) else None
        if not lines and line.strip().lower() in ("q", "quit", "exit"):
            return None
        if line.strip() == "END":
            return "\n".join(lines)
        lines.append(line)
        blanks = blanks + 1 if not line.strip() else 0
        text = "\n".join(lines).strip()
        if not text:
            lines.clear()
            continue
        try:
            json.loads(text)
            return text                      # complete JSON -> validate now
        except json.JSONDecodeError:
            if blanks >= 2:                  # pasted something broken -> validate to show the error
                return text


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Paste a Video Edit Plan JSON and validate it.")
    ap.add_argument("plan", nargs="?", help="optional: validate a JSON file instead of pasting")
    ap.add_argument("--schema", default=str(DEFAULT_SCHEMA), help="path to the JSON schema")
    ap.add_argument("--check-files", action="store_true",
                    help="also verify source/music/sfx files exist (relative to the current folder)")
    args = ap.parse_args(argv)

    if args.plan:
        return 0 if report(lambda: load_plan(args.plan, args.schema, args.check_files)) else 1

    print("Video Edit Plan validator")
    print("Paste your JSON below (it validates automatically when the JSON is complete).")
    print("If it doesn't react (broken JSON), press Enter twice or type END. Type q to quit.\n")
    all_ok = True
    while True:
        text = read_pasted_json()
        if text is None:
            return 0 if all_ok else 1
        all_ok &= report(lambda: load_plan_from_text(text, args.schema, args.check_files))
        print("\n" + "-" * 60)
        print("Paste another JSON, or type q to quit.\n")


if __name__ == "__main__":
    sys.exit(main())
