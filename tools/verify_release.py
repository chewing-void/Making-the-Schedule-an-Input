#!/usr/bin/env python3
"""Dependency-free packaging checks, not runtime or legal certification."""
import ast
import hashlib
import json
import platform
import re
import sys
import tokenize
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "RELEASE_MANIFEST.json"
REPORT = "RELEASE_CHECKS.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    errors = []
    files = sorted(p for p in ROOT.rglob("*") if p.is_file())
    rels = {p.relative_to(ROOT).as_posix() for p in files}
    inventory = json.loads((ROOT / MANIFEST).read_text(encoding="utf-8"))
    expected = inventory["files"]
    actual = rels - {MANIFEST, REPORT}
    if actual != set(expected):
        errors.append({"inventory_mismatch": {"extra": sorted(actual - set(expected)),
                                               "missing": sorted(set(expected) - actual)}})
    for rel, row in expected.items():
        if not (ROOT / rel).is_file() or digest(ROOT / rel) != row["sha256"]:
            errors.append({"digest_mismatch": rel})
    python_files = [p for p in files if p.suffix == ".py"]
    for p in python_files:
        try:
            with tokenize.open(p) as handle:
                ast.parse(handle.read(), filename=str(p.relative_to(ROOT)))
        except (SyntaxError, UnicodeError, OSError) as exc:
            errors.append({"python_parse": p.relative_to(ROOT).as_posix(), "error": str(exc)})
    source_map = json.loads((ROOT / "SOURCE_MAP.json").read_text(encoding="utf-8"))
    for rel in source_map["excluded_files"]:
        if rel in rels:
            errors.append({"excluded_file_present": rel})
    mapping = {row["file"] for row in source_map["source_files"]}
    source_files = {p.relative_to(ROOT).as_posix() for p in files
                    if p.suffix in {".py", ".patch"} and p.parent.name != "tools"}
    if mapping != source_files:
        errors.append({"source_mapping_mismatch": sorted(mapping ^ source_files)})
    for row in source_map["license_files"]:
        if digest(ROOT / row["path"]) != row["sha256"]:
            errors.append({"license_digest": row["path"]})
    prohibited_names = {".git", ".hg", ".svn", "__pycache__", ".venv", "node_modules",
                        ".env", "id_rsa", "id_ed25519", "credentials", ".DS_Store"}
    suspicious_extensions = {".pem", ".key", ".pfx", ".p12", ".pt", ".pth", ".npz",
                             ".npy", ".pyc", ".pyo", ".zip", ".exe", ".dll", ".so"}
    for p in ROOT.rglob("*"):
        rel = p.relative_to(ROOT).as_posix()
        if p.is_symlink() or p.name in prohibited_names or p.name.startswith(".env."):
            errors.append({"prohibited_path": rel})
        if p.is_file() and p.suffix.lower() in suspicious_extensions:
            errors.append({"unexpected_binary_or_secret_file": rel})
    # Only high-confidence recognizable token formats. Never print matched values.
    patterns = {
        "private_key_block": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
        "github_token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{60,})\b"),
        "aws_access_key": re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
        "openai_key": re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{40,}\b"),
        "slack_token": re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b"),
        "credential_in_url": re.compile(r"https?://[^\s/@:]+:[^\s/@]+@"),
    }
    local_link_count = 0
    for p in files:
        rel = p.relative_to(ROOT).as_posix()
        try:
            text = p.read_text(encoding="utf-8-sig")
        except UnicodeError:
            errors.append({"non_text_file": rel})
            continue
        if "\ufffd" in text:
            errors.append({"replacement_character": rel})
        for label, pattern in patterns.items():
            if pattern.search(text):
                errors.append({"possible_secret": label, "file": rel})
        if p.suffix == ".md":
            if sum(1 for line in text.splitlines() if line.startswith("```")) % 2:
                errors.append({"unbalanced_code_fence": rel})
            for link in re.findall(r"\[[^\]]*\]\(([^)]+)\)", text):
                if re.match(r"[a-zA-Z][a-zA-Z0-9+.-]*:", link) or link.startswith("#"):
                    continue
                target = unquote(link.split("#", 1)[0])
                if target:
                    local_link_count += 1
                    if not (p.parent / target).exists():
                        errors.append({"broken_local_link": {"file": rel, "target": target}})
    result = {
        "status": "passed" if not errors else "failed",
        "scope": "Static packaging and syntax only; no imports, runtime, build, benchmark, legal clearance or scientific reproduction validation.",
        "python_version": platform.python_version(),
        "file_count": len(files),
        "python_files_parsed": len(python_files),
        "mapped_source_and_patch_files": len(mapping),
        "exact_license_files_verified": len(source_map["license_files"]),
        "excluded_files_checked": len(source_map["excluded_files"]),
        "local_markdown_links_checked": local_link_count,
        "secret_scan": "High-confidence heuristic patterns only; absence of findings is not a guarantee.",
        "errors": errors,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
