#!/usr/bin/env python3
"""Check local import targets, task manifests, and an optional slime base tree."""

import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check_tito_binary(root):
    path = root / "terminus/bin/tito-linux-x86_64"
    data = path.read_bytes()
    metadata = json.loads(path.with_suffix(".json").read_text())
    if not data.startswith(b"\x7fELF") or data[18:20] != b"\x3e\x00":
        raise ValueError("Expected the Linux x86_64 TiTO executable")
    if hashlib.sha256(data).hexdigest() != metadata["binary_sha256"]:
        raise ValueError("TiTO executable checksum mismatch")
    if not path.stat().st_mode & 0o111:
        raise ValueError("TiTO executable mode is missing")
    print("TiTO executable: Linux x86_64, checksum and executable mode verified")


def local_import_errors(root):
    packages = {
        "compactionrl": root / "buffer/src/compactionrl",
        "harbor": root / "terminus/src/harbor",
        "slime": root / "slime/slime",
        "slime_plugins": root / "slime/slime_plugins",
    }
    errors = []
    count = 0
    for package, base in packages.items():
        for path in sorted(base.rglob("*.py")):
            count += 1
            relative = path.relative_to(base).with_suffix("").parts
            context = ".".join([package, *relative[:-1]])
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                modules = []
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    name = "." * node.level + (node.module or "")
                    modules = [
                        importlib.util.resolve_name(name, context)
                        if node.level
                        else name
                    ]
                for name in modules:
                    parts = name.split(".")
                    if parts[0] not in packages:
                        continue
                    target = packages[parts[0]].joinpath(*parts[1:])
                    if not target.is_dir() and not target.with_suffix(".py").is_file():
                        errors.append(f"{path.relative_to(root)}:{node.lineno}: {name}")
    return count, errors


def source_entry(path, base):
    ignored = {".git", "__pycache__", ".pytest_cache", ".ruff_cache", "build"}
    relative = path.relative_to(base)
    # CompactionRL configs/docs/tests are component assets, separate from the
    # reconstructable upstream framework and its code patch.
    if relative.parts[0] == "compactionrl":
        return False
    return (path.is_file() or path.is_symlink()) and not any(
        part in ignored or part.endswith(".egg-info")
        for part in relative.parts
    )


def check_upstream(root, base):
    manifest = json.loads((root / "SOURCE_MANIFEST.json").read_text())["sources"][
        "slime"
    ]
    allowed = set(manifest["overlay_files"])
    current = root / "slime"
    native = {
        path.relative_to(base): path
        for path in base.rglob("*")
        if source_entry(path, base)
    }
    if not native:
        raise ValueError("The upstream directory is empty")
    changed = set()
    for relative, original in native.items():
        target = current / relative
        if original.is_symlink():
            if not target.is_symlink() or os.readlink(original) != os.readlink(target):
                raise ValueError(f"Upstream symbolic link changed: {relative}")
        elif not target.is_file():
            raise ValueError(f"Upstream file removed: {relative}")
        else:
            if original.stat().st_mode & 0o111 != target.stat().st_mode & 0o111:
                raise ValueError(f"Upstream executable mode changed: {relative}")
            if original.read_bytes() != target.read_bytes():
                changed.add(str(relative))
                if str(relative) not in allowed:
                    raise ValueError(f"Undocumented upstream change: {relative}")
    extras = {
        str(path.relative_to(current))
        for path in current.rglob("*")
        if source_entry(path, current)
        and path.relative_to(current) not in native
    }
    if extras - allowed:
        raise ValueError(f"Undocumented additions: {sorted(extras - allowed)}")
    if changed | extras != allowed:
        raise ValueError(f"Stale overlay manifest entries: {sorted(allowed - changed - extras)}")
    print(
        f"slime: all {len(native)} upstream file/link entries retained; {len(allowed)} documented overlay files"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--slime-upstream", type=Path, help="Extracted pinned official slime tree"
    )
    args = parser.parse_args()
    check_tito_binary(ROOT)
    count, errors = local_import_errors(ROOT)
    if errors:
        raise SystemExit("Missing local import targets:\n" + "\n".join(errors))
    print(f"Local imports: {count} Python modules checked")
    lists = []
    for name, expected in [
        ("terminus/data/manifests/swe_verified_200.json", 200),
        ("buffer/data/manifests/swedev_reference_1105.json", 1105),
    ]:
        items = json.loads((ROOT / name).read_text())
        if (
            not isinstance(items, list)
            or len(items) != expected
            or len(set(items)) != expected
        ):
            raise ValueError(f"Invalid task manifest: {name}")
        lists.append(set(items))
    if lists[0] & lists[1]:
        raise ValueError("Training and fixed evaluation task IDs overlap")
    print("Task manifests: 200 evaluation / 1105 training IDs, zero overlap")
    if args.slime_upstream:
        check_upstream(ROOT, args.slime_upstream.resolve())


if __name__ == "__main__":
    main()
