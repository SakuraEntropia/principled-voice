#!/usr/bin/env python3
"""Local-only intake checks for a private voice corpus.

Reads metadata; never uploads, copies, edits or synthesises audio. Does not infer
speaker, language, transcript, emotion or license terms from audio. Everything that
is not explicitly declared stays null and blocks a "cleared" status.

Commands:
  validate  check corpus.json + manifest.jsonl structure, references and scope
  plan      deterministically group clips and split train/dev/test
  leakage   check that no speaker/session/script/text/audio crosses splits
  inventory hash and probe files in an inbox folder (no manifest is guessed)
  self-test exercise the checks on synthetic metadata in a temporary directory
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
import unicodedata
from collections import defaultdict
from datetime import date
from pathlib import Path

USES = {
    "recording", "model_training", "synthesis", "voice_cloning", "commercial",
    "broadcast", "animation_production", "redistribution",
}
REQUIRED_USES = {"model_training", "synthesis"}
SIGNER_ROLES = {"speaker", "guardian", "rights_holder", "licensee", "other"}
SPLITS = ("train", "dev", "test")
STATUSES = {"cleared", "pending", "rejected", "unknown"}
TRANSCRIPT_STATUSES = {"human_verified", "provided_unverified", "asr_draft", None}
AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".aac", ".aiff", ".aif", ".ogg", ".opus"}
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
SHA_RE = re.compile(r"^[a-f0-9]{64}$")
CLIP_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
LANG_RE = re.compile(r"^[A-Za-z]{2,3}([-_][A-Za-z0-9]{2,8})*$")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{n}: invalid JSON ({exc})") from exc
    return rows


def check_date(value, where: str, errors: list[str], allow_null: bool = False) -> None:
    if value is None and allow_null:
        return
    if not isinstance(value, str) or not DATE_RE.match(value):
        errors.append(f"{where}: expected YYYY-MM-DD")
        return
    try:
        date.fromisoformat(value)
    except ValueError:
        errors.append(f"{where}: invalid calendar date")


def check_uses(uses, where: str, errors: list[str]) -> None:
    if not isinstance(uses, list) or not uses:
        errors.append(f"{where}: scope must be a non-empty list")
        return
    unknown = sorted(set(uses) - USES)
    if unknown:
        errors.append(f"{where}: unknown uses {unknown}")


def validate_corpus(corpus: dict, root: Path, check_files: bool):
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(corpus, dict):
        return ["corpus.json must be a JSON object"], warnings, {}, {}
    if corpus.get("schema_version") != 1:
        errors.append("corpus.schema_version must be 1")
    for key in ("corpus_id", "received_on", "licenses", "authorizations"):
        if key not in corpus:
            errors.append(f"corpus.{key} is required")
    check_date(corpus.get("received_on"), "corpus.received_on", errors)
    licenses: dict[str, dict] = {}
    authorizations: dict[str, dict] = {}
    for i, lic in enumerate(corpus.get("licenses") or []):
        where = f"licenses[{i}]"
        if not isinstance(lic, dict):
            errors.append(f"{where}: must be an object")
            continue
        lid = lic.get("license_id")
        if not lid:
            errors.append(f"{where}.license_id is required")
            continue
        if lid in licenses:
            errors.append(f"{where}: duplicate license_id {lid!r}")
        licenses[lid] = lic
        for key in ("name", "grantor"):
            if not lic.get(key):
                errors.append(f"{where}.{key} is required")
        check_uses(lic.get("scope"), f"{where}.scope", errors)
        check_date(lic.get("granted_on"), f"{where}.granted_on", errors, allow_null=True)
        check_date(lic.get("expires_on"), f"{where}.expires_on", errors, allow_null=True)
        _check_document(lic, where, root, check_files, errors)
    for i, auth in enumerate(corpus.get("authorizations") or []):
        where = f"authorizations[{i}]"
        if not isinstance(auth, dict):
            errors.append(f"{where}: must be an object")
            continue
        cid = auth.get("consent_id")
        if not cid:
            errors.append(f"{where}.consent_id is required")
            continue
        if cid in authorizations:
            errors.append(f"{where}: duplicate consent_id {cid!r}")
        authorizations[cid] = auth
        if auth.get("signer_role") not in SIGNER_ROLES:
            errors.append(f"{where}.signer_role must be one of {sorted(SIGNER_ROLES)}")
        check_date(auth.get("signed_on"), f"{where}.signed_on", errors)
        check_uses(auth.get("scope"), f"{where}.scope", errors)
        check_date(auth.get("valid_until"), f"{where}.valid_until", errors, allow_null=True)
        if auth.get("signer_role") == "licensee":
            warnings.append(f"{where}: a licensee signature alone is not speaker consent")
        _check_document(auth, where, root, check_files, errors)
    return errors, warnings, licenses, authorizations


def _check_document(record: dict, where: str, root: Path, check_files: bool,
                    errors: list[str]) -> None:
    path, digest = record.get("document_path"), record.get("document_sha256")
    if digest is not None and not (isinstance(digest, str) and SHA_RE.match(digest)):
        errors.append(f"{where}.document_sha256 must be a lowercase sha256")
    if path is None:
        return
    if not isinstance(path, str) or path.startswith("/") or ".." in Path(path).parts:
        errors.append(f"{where}.document_path must be a relative path under data/")
        return
    if check_files:
        target = (root / path).resolve()
        if root.resolve() not in target.parents:
            errors.append(f"{where}.document_path escapes data root")
        elif not target.is_file():
            errors.append(f"{where}.document_path does not exist: {path}")
        elif digest and sha256_file(target) != digest:
            errors.append(f"{where}.document_sha256 does not match {path}")


def validate_clips(clips: list[dict], licenses: dict, authorizations: dict, root: Path,
                   check_files: bool):
    errors: list[str] = []
    warnings: list[str] = []
    seen: set[str] = set()
    today = date.today()
    for i, clip in enumerate(clips):
        where = f"manifest[{i}]"
        if not isinstance(clip, dict):
            errors.append(f"{where}: must be an object")
            continue
        if clip.get("schema_version") != 1:
            errors.append(f"{where}.schema_version must be 1")
        cid = clip.get("clip_id")
        if not isinstance(cid, str) or not CLIP_ID_RE.match(cid):
            errors.append(f"{where}.clip_id invalid or missing")
        elif cid in seen:
            errors.append(f"{where}: duplicate clip_id {cid!r}")
        else:
            seen.add(cid)
        for key in ("speaker_id", "session_id", "script_id"):
            if not isinstance(clip.get(key), str) or not clip[key].strip():
                errors.append(f"{where}.{key} is required and must be non-empty")
        rel = clip.get("file")
        if not isinstance(rel, str) or not rel or rel.startswith("/") or ".." in Path(rel).parts:
            errors.append(f"{where}.file must be a non-empty relative path without '..'")
        if not LANG_RE.match(str(clip.get("language", ""))):
            errors.append(f"{where}.language must be a BCP-47-like tag")
        digest = clip.get("sha256")
        if digest is not None and not (isinstance(digest, str) and SHA_RE.match(digest)):
            errors.append(f"{where}.sha256 must be null or a lowercase sha256")
        if clip.get("transcript_status") not in TRANSCRIPT_STATUSES:
            errors.append(f"{where}.transcript_status invalid")
        if clip.get("split") is not None and clip["split"] not in SPLITS:
            errors.append(f"{where}.split invalid")
        status = clip.get("usage_status")
        if status not in STATUSES:
            errors.append(f"{where}.usage_status invalid")
            continue
        if clip.get("permitted_uses"):
            check_uses(clip["permitted_uses"], f"{where}.permitted_uses", errors)
        lic = licenses.get(clip.get("license_id") or "")
        auth = authorizations.get(clip.get("authorization_id") or "")
        if status != "cleared":
            if status == "rejected":
                warnings.append(f"{where}: rejected clip remains in manifest")
            continue
        if lic is None:
            errors.append(f"{where}: cleared but license_id does not resolve")
        if auth is None:
            errors.append(f"{where}: cleared but authorization_id does not resolve")
        if lic is not None:
            missing = REQUIRED_USES - set(lic.get("scope") or [])
            if missing:
                errors.append(f"{where}: license lacks scope {sorted(missing)}")
            expiry = lic.get("expires_on")
            if expiry and expiry < today.isoformat():
                errors.append(f"{where}: license expired on {expiry}")
        if auth is not None:
            missing = REQUIRED_USES - set(auth.get("scope") or [])
            if missing:
                errors.append(f"{where}: authorization lacks scope {sorted(missing)}")
            expiry = auth.get("valid_until")
            if expiry and expiry < today.isoformat():
                errors.append(f"{where}: authorization expired on {expiry}")
            if clip.get("speaker_role") == "target_speaker" and "voice_cloning" not in set(
                    auth.get("scope") or []):
                warnings.append(f"{where}: target speaker has no explicit voice_cloning consent")
        if clip.get("speaker_role") == "target_speaker" and lic is not None and \
                "voice_cloning" not in set(lic.get("scope") or []):
            warnings.append(f"{where}: license has no explicit voice_cloning scope")
        if check_files and isinstance(rel, str):
            target = (root / rel).resolve()
            if root.resolve() not in target.parents:
                errors.append(f"{where}.file escapes data root")
            elif not target.is_file():
                errors.append(f"{where}.file does not exist: {rel}")
            elif digest and sha256_file(target) != digest:
                errors.append(f"{where}.sha256 does not match {rel}")
    return errors, warnings


def components(clips: list[dict]) -> list[dict]:
    """Connected components over speaker/session/script keys (union-find)."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for clip in clips:
        keys = [f"speaker:{clip['speaker_id']}", f"session:{clip['session_id']}",
                f"script:{clip['script_id']}"]
        for key in keys:
            find(key)
        for key in keys[1:]:
            union(keys[0], key)
    groups: dict[str, list[dict]] = defaultdict(list)
    for clip in clips:
        groups[find(f"speaker:{clip['speaker_id']}")].append(clip)
    out = []
    for members in groups.values():
        weight = sum(float(m.get("duration_s") or 0.0) for m in members)
        missing = sum(1 for m in members if not m.get("duration_s"))
        if weight <= 0:
            weight = float(len(members))
            missing = len(members)
        out.append({
            "clip_ids": sorted(m["clip_id"] for m in members),
            "weight": weight,
            "duration_missing": missing,
            "key": "|".join(sorted(f"{k}={m[k]}" for m in members
                                   for k in ("speaker_id", "session_id", "script_id"))),
        })
    return out


def plan_splits(clips: list[dict], seed: int, ratios: dict[str, float]) -> dict:
    comps = components(clips)
    total = sum(c["weight"] for c in comps) or 1.0

    def rand_key(comp: dict) -> str:
        return hashlib.sha256(f"{seed}:{comp['key']}".encode()).hexdigest()

    comps.sort(key=lambda c: (-c["weight"], rand_key(c)))
    assigned = {s: 0.0 for s in SPLITS}
    assignment: dict[str, str] = {}
    for comp in comps:
        # Greedy: give each component to the split furthest below its target.
        target = max(SPLITS, key=lambda s: (ratios[s] * total - assigned[s],
                                            -SPLITS.index(s)))
        assigned[target] += comp["weight"]
        for cid in comp["clip_ids"]:
            assignment[cid] = target
        comp["split"] = target
    # Keep a disjoint holdout possible: with >= 3 groups, never leave dev/test empty.
    if len(comps) >= 3:
        for split in SPLITS:
            if any(v == split for v in assignment.values()):
                continue
            donors = [s for s in SPLITS
                      if sum(1 for v in assignment.values() if v == s) > 1]
            if not donors:
                break
            donor = max(donors, key=lambda s: (sum(1 for v in assignment.values() if v == s),
                                               -SPLITS.index(s)))
            candidates = [c for c in comps if c["split"] == donor]
            moved = min(candidates, key=lambda c: (c["weight"], rand_key(c)))
            moved["split"] = split
            for cid in moved["clip_ids"]:
                assignment[cid] = split
    return {
        "schema_version": 1,
        "seed": seed,
        "ratios": ratios,
        "generated_on": date.today().isoformat(),
        "duration_weight_complete": all(not c["duration_missing"] for c in comps),
        "assignment": dict(sorted(assignment.items())),
        "components": [{k: c[k] for k in ("clip_ids", "split", "weight", "duration_missing")}
                       for c in comps],
    }


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return " ".join(text.split())


def _ngrams(text: str, n: int = 5) -> set[str]:
    return {text[i:i + n] for i in range(max(0, len(text) - n + 1))}


def check_leakage(clips: list[dict], assignment: dict[str, str]) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    by_id = {c["clip_id"]: c for c in clips}
    missing = sorted(set(by_id) - set(assignment))
    if missing:
        errors.append(f"{len(missing)} clips have no split assignment, e.g. {missing[:3]}")
    stats: dict[str, dict] = {}
    for split in SPLITS:
        members = [c for c in clips if assignment.get(c["clip_id"]) == split]
        stats[split] = {
            "clips": len(members),
            "duration_s": round(sum(float(c.get("duration_s") or 0) for c in members), 3),
            "speakers": len({c["speaker_id"] for c in members}),
            "sessions": len({c["session_id"] for c in members}),
            "scripts": len({c["script_id"] for c in members}),
        }
        if not members:
            errors.append(f"split {split!r} is empty (need >= 3 independent groups)")
    for comp in components(clips):
        splits = {assignment.get(cid) for cid in comp["clip_ids"]}
        if len(splits) > 1:
            errors.append(f"component crosses splits {sorted(map(str, splits))}: "
                          f"{comp['clip_ids'][:4]}")
    by_hash: dict[str, list[str]] = defaultdict(list)
    for clip in clips:
        if clip.get("sha256"):
            by_hash[clip["sha256"]].append(clip["clip_id"])
    for digest, ids in by_hash.items():
        splits = {assignment.get(cid) for cid in ids}
        if len(splits) > 1:
            errors.append(f"sha256 {digest[:12]} appears in multiple splits: {ids[:4]}")
        elif len(ids) > 1:
            warnings.append(f"duplicate recording within one split: {ids[:4]}")
    unique_texts: dict[str, list[str]] = defaultdict(list)
    for clip in clips:
        if clip.get("transcript") and clip.get("transcript_status") in (
                "human_verified", "provided_unverified"):
            text = _normalize(clip["transcript"])
            if text:
                unique_texts[text].append(clip["clip_id"])
    for text, ids in unique_texts.items():
        splits = {assignment.get(cid) for cid in ids}
        if len(splits) > 1:
            errors.append(f"identical transcript in multiple splits: {ids[:4]}")
    texts = list(unique_texts)
    if len(texts) > 2000:
        warnings.append(f"transcript similarity skipped: {len(texts)} unique texts")
    else:
        grams = {t: _ngrams(t) for t in texts}
        for i, a in enumerate(texts):
            split_a = assignment.get(unique_texts[a][0])
            for b in texts[i + 1:]:
                if assignment.get(unique_texts[b][0]) == split_a:
                    continue
                ga, gb = grams[a], grams[b]
                if not ga or not gb:
                    continue
                jaccard = len(ga & gb) / len(ga | gb)
                if jaccard >= 0.8:
                    warnings.append(
                        f"near-duplicate text across splits (Jaccard {jaccard:.2f}): "
                        f"{unique_texts[a][0]} vs {unique_texts[b][0]}")
    return {"stats": stats, "errors": errors, "warnings": warnings}


def cmd_validate(args) -> int:
    root = Path(args.data).resolve()
    corpus_path, manifest_path = root / "corpus.json", root / "manifest.jsonl"
    if not corpus_path.is_file() or not manifest_path.is_file():
        print(f"nothing to validate in {root}: expected corpus.json and manifest.jsonl")
        print("create them once the authorized corpus arrives; see docs/DATA_INTAKE.md")
        return 1
    corpus = load_json(corpus_path)
    clips = load_jsonl(manifest_path)
    errors, warnings, licenses, authorizations = validate_corpus(
        corpus, root, args.check_files)
    clip_errors, clip_warnings = validate_clips(
        clips, licenses, authorizations, root, args.check_files)
    errors += clip_errors
    warnings += clip_warnings
    cleared = sum(1 for c in clips if c.get("usage_status") == "cleared")
    print(f"corpus={corpus.get('corpus_id')!r} clips={len(clips)} cleared={cleared} "
          f"licenses={len(licenses)} authorizations={len(authorizations)}")
    for message in errors:
        print(f"ERROR {message}")
    for message in warnings:
        print(f"WARN  {message}")
    print(f"result: {'FAIL' if errors else 'OK'} ({len(errors)} errors, "
          f"{len(warnings)} warnings)")
    return 1 if errors else 0


def cmd_plan(args) -> int:
    root = Path(args.data).resolve()
    manifest_path = root / "manifest.jsonl"
    if not manifest_path.is_file():
        print(f"no manifest at {manifest_path}; nothing to plan")
        return 1
    clips = load_jsonl(manifest_path)
    ratios = {s: float(v) for s, v in zip(SPLITS, args.ratios.split(","))}
    if abs(sum(ratios.values()) - 1) > 1e-6 or any(v <= 0 for v in ratios.values()):
        print("ratios must be positive and sum to 1, e.g. 0.8,0.1,0.1")
        return 2
    plan = plan_splits(clips, args.seed, ratios)
    plan["dataset_manifest_sha256"] = sha256_file(manifest_path)
    counts = {s: 0 for s in SPLITS}
    for split in plan["assignment"].values():
        counts[split] += 1
    print(f"components={len(plan['components'])} clips={len(clips)} seed={args.seed} "
          f"counts={counts} duration_weight_complete={plan['duration_weight_complete']}")
    if len(plan["components"]) < 3:
        print("WARN  fewer than 3 independent speaker/session/script groups: "
              "a disjoint train/dev/test split is impossible")
    if any(counts[s] == 0 for s in SPLITS):
        print("WARN  an empty split remains; leakage check will fail by design")
    if not plan["duration_weight_complete"]:
        print("WARN  some durations are null; split ratios use per-clip unit weights")
    if args.write:
        out = root / "derived" / "splits.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
        print(f"wrote {out}")
    return 0


def cmd_leakage(args) -> int:
    root = Path(args.data).resolve()
    clips = load_jsonl(root / "manifest.jsonl")
    splits_path = Path(args.splits).resolve() if args.splits else root / "derived" / "splits.json"
    if splits_path.is_file():
        assignment = load_json(splits_path)["assignment"]
    else:
        assignment = {c["clip_id"]: c["split"] for c in clips if c.get("split")}
        if not assignment:
            print(f"no {splits_path}; run 'plan --write' first")
            return 1
    report = check_leakage(clips, assignment)
    print(json.dumps(report["stats"], indent=2, ensure_ascii=False))
    for message in report["errors"]:
        print(f"ERROR {message}")
    for message in report["warnings"]:
        print(f"WARN  {message}")
    print(f"result: {'FAIL' if report['errors'] else 'OK'} "
          f"({len(report['errors'])} errors, {len(report['warnings'])} warnings)")
    return 1 if report["errors"] else 0


def cmd_inventory(args) -> int:
    import soundfile as sf

    inbox = Path(args.inbox).resolve()
    out = Path(args.out).resolve()
    if not inbox.is_dir():
        print(f"no inbox directory: {inbox}")
        return 1
    files = sorted(p for p in inbox.rglob("*")
                   if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS)
    out.parent.mkdir(parents=True, exist_ok=True)
    total = 0.0
    with out.open("w", encoding="utf-8") as handle:
        for path in files:
            rel = path.relative_to(Path(args.data).resolve()).as_posix() \
                if Path(args.data).resolve() in path.parents else path.name
            record = {"file": rel, "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
            try:
                info = sf.info(str(path))
                record.update({"duration_s": info.frames / info.samplerate,
                               "sample_rate": info.samplerate, "channels": info.channels,
                               "format": info.format, "subtype": info.subtype})
                total += record["duration_s"]
            except Exception as exc:  # noqa: BLE001 - report unreadable file, never guess
                record["probe_error"] = f"{type(exc).__name__}: {exc}"
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"inventoried {len(files)} audio files ({total / 60:.1f} min) -> {out}")
    print("this draft has no speaker/session/script/license values; fill those by hand")
    return 0


def cmd_self_test(_args) -> int:
    def fake_sha(tag: str) -> str:
        return hashlib.sha256(tag.encode()).hexdigest()

    licenses = [{"license_id": "L1", "name": "test grant", "grantor": "rights holder",
                 "granted_on": "2026-01-01", "scope": sorted(USES)}]
    auths = [{"consent_id": f"C{s}", "signer_role": "speaker", "signed_on": "2026-01-02",
              "scope": sorted(USES)} for s in "abcde"]
    corpus = {"schema_version": 1, "corpus_id": "selftest", "received_on": "2026-01-03",
              "licenses": licenses, "authorizations": auths}
    clips = []
    for i, speaker in enumerate("abcde"):
        clips.append({"schema_version": 1, "clip_id": f"c{i}", "file": f"audio/c{i}.wav",
                      "sha256": fake_sha(f"clip{i}"), "duration_s": 2.0,
                      "speaker_id": speaker, "session_id": f"s{i}", "script_id": f"x{i}",
                      "language": "zh-CN", "license_id": "L1", "authorization_id": f"C{speaker}",
                      "usage_status": "cleared", "split": None})
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        errors, _warnings, lic, auth = validate_corpus(corpus, root, False)
        assert not errors, errors
        errors, _ = validate_clips(clips, lic, auth, root, False)
        assert not errors, errors
        plan = plan_splits(clips, 42, {"train": 0.6, "dev": 0.2, "test": 0.2})
        assert set(plan["assignment"].values()) == set(SPLITS), plan["assignment"]
        report = check_leakage(clips, plan["assignment"])
        assert not report["errors"], report["errors"]
        same_group = [dict(clips[0], clip_id="m0", sha256=fake_sha("m0")),
                      dict(clips[0], clip_id="m1", sha256=fake_sha("m1"))]
        assert check_leakage(same_group, {"m0": "train", "m1": "test"})["errors"], \
            "cross-split component not caught"
        duplicate = [dict(c) for c in clips]
        duplicate[1]["sha256"] = duplicate[0]["sha256"]
        mapping = {"c0": "train", "c1": "test", "c2": "train", "c3": "dev", "c4": "dev"}
        assert check_leakage(duplicate, mapping)["errors"], "duplicate audio not caught"
        broken = [dict(c) for c in clips]
        broken[0]["authorization_id"] = None
        errors, _ = validate_clips(broken, lic, auth, root, False)
        assert errors, "cleared clip without authorization not caught"
        bad_lang = [dict(clips[0], language="not a tag")]
        assert validate_clips(bad_lang, lic, auth, root, False)[0]
    print("self-test ok")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate", help="check metadata, references and scope")
    validate.add_argument("--data", type=Path, default=Path("data"))
    validate.add_argument("--check-files", action="store_true",
                          help="also verify file existence and hashes")
    validate.set_defaults(func=cmd_validate)
    plan = sub.add_parser("plan", help="deterministic speaker/session/script split")
    plan.add_argument("--data", type=Path, default=Path("data"))
    plan.add_argument("--seed", type=int, default=42)
    plan.add_argument("--ratios", default="0.8,0.1,0.1")
    plan.add_argument("--write", action="store_true")
    plan.set_defaults(func=cmd_plan)
    leakage = sub.add_parser("leakage", help="cross-split leakage checks")
    leakage.add_argument("--data", type=Path, default=Path("data"))
    leakage.add_argument("--splits", type=Path, default=None)
    leakage.set_defaults(func=cmd_leakage)
    inventory = sub.add_parser("inventory", help="hash/probe audio without guessing metadata")
    inventory.add_argument("--data", type=Path, default=Path("data"))
    inventory.add_argument("--inbox", type=Path, default=Path("data/inbox"))
    inventory.add_argument("--out", type=Path, default=Path("data/derived/inventory.jsonl"))
    inventory.set_defaults(func=cmd_inventory)
    test = sub.add_parser("self-test", help="exercise checks on synthetic metadata")
    test.set_defaults(func=cmd_self_test)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
