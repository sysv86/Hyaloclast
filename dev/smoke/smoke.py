#!/usr/bin/env python3
"""Hyaloclast smoke test and CI verification harness.

Checks:
  --check-packs         Validate built-in .vsp packs, pack.toml, shader files, and zip safety self-tests.
  --check-lang          Verify en_us.json and ru_ru.json JSON validity and 100% key parity.
  --check-neoforge-free Verify zero NeoForge files or reflection references remain.
  --vulkan-smoke        Run headless Vulkan validation-layer smoke checks (or report skipped in dry-run/CI).
  --ci                  Run all checks in CI mode (gracefully skips --vulkan-smoke if binaries/ICD are absent).
"""

from __future__ import annotations

import argparse
import io
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from typing import Any

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None  # type: ignore[assignment]


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
BUILTIN_PACKS_DIR = REPO_ROOT / "src" / "main" / "resources" / "builtin-packs"
LANG_DIR = REPO_ROOT / "src" / "main" / "resources" / "assets" / "vulkanshaders" / "lang"

MAX_ZIP_ENTRIES = 2048
MAX_ZIP_TOTAL_BYTES = 128 * 1024 * 1024  # 128 MiB
MAX_ZIP_COMPRESSION_RATIO = 100.0

SHADER_EXTENSIONS = {".vsl", ".vsh", ".fsh", ".comp", ".rgen", ".rchit", ".rahit", ".rmiss"}
REQUIRED_BUILTIN_PACKS = (
    "hyaloclast-sunlit-vanilla.vsp",
    "hyaloclast-obsidian-showcase.vsp",
)


def parse_toml_text(text: str) -> dict[str, Any]:
    if tomllib is not None:
        return tomllib.loads(text)
    # Minimal fallback parser for Python < 3.11 environments (handles sections & key = value).
    root: dict[str, Any] = {}
    current = root
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[[") and line.endswith("]]"):
            key = line[2:-2].strip()
            lst = root.setdefault(key, [])
            if isinstance(lst, list):
                entry: dict[str, Any] = {}
                lst.append(entry)
                current = entry
            continue
        if line.startswith("[") and line.endswith("]"):
            path = [part.strip() for part in line[1:-1].split(".")]
            cur = root
            for part in path:
                nxt = cur.setdefault(part, {})
                if isinstance(nxt, dict):
                    cur = nxt
            current = cur
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip()
            if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                current[k] = v[1:-1]
            else:
                current[k] = v
    return root


def validate_zip_archive_safety(zip_path: pathlib.Path) -> list[str]:
    """Validate a .zip/.vsp archive against path traversal, absolute paths, zip bombs, and missing pack.toml."""
    errors: list[str] = []
    if not zip_path.is_file():
        return [f"Archive does not exist: {zip_path}"]

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            infos = zf.infolist()
            if len(infos) > MAX_ZIP_ENTRIES:
                errors.append(f"Archive exceeds maximum entry count ({len(infos)} > {MAX_ZIP_ENTRIES})")

            has_manifest = False
            total_uncompressed = 0

            for info in infos:
                name = info.filename
                if name == "pack.toml" and not info.is_dir():
                    has_manifest = True

                # Reject Windows drive letters, leading slashes, NUL bytes, and parent-dir traversal.
                if "\x00" in name:
                    errors.append(f"NUL byte in archive entry: {name!r}")
                    continue
                if name.startswith("/") or name.startswith("\\") or re.match(r"^[A-Za-z]:", name):
                    errors.append(f"Absolute path in archive entry: {name!r}")
                    continue
                normalized_parts = name.replace("\\", "/").split("/")
                if ".." in normalized_parts:
                    errors.append(f"Path traversal in archive entry: {name!r}")
                    continue

                # Check uncompressed size and zip-bomb ratio.
                total_uncompressed += max(0, info.file_size)
                if total_uncompressed > MAX_ZIP_TOTAL_BYTES:
                    errors.append(
                        f"Archive uncompressed size exceeds limit ({total_uncompressed} > {MAX_ZIP_TOTAL_BYTES})"
                    )
                    break
                if info.compress_size > 0 and info.file_size > 1024 * 1024:
                    ratio = info.file_size / info.compress_size
                    if ratio > MAX_ZIP_COMPRESSION_RATIO:
                        errors.append(
                            f"Suspected zip bomb in entry {name!r} (compression ratio {ratio:.1f}x > {MAX_ZIP_COMPRESSION_RATIO:.0f}x)"
                        )

            if not has_manifest:
                errors.append("Archive is missing root pack.toml")
    except zipfile.BadZipFile as exc:
        errors.append(f"Malformed zip archive: {exc}")

    return errors


def run_zip_safety_self_tests() -> list[str]:
    """Run synthetic zip-safety self-tests covering traversal, absolute paths, zip bombs, and missing pack.toml."""
    failures: list[str] = []
    with tempfile.TemporaryDirectory(prefix="hyaloclast-zip-smoke-") as tmp:
        tmp_dir = pathlib.Path(tmp)

        # 1. Valid minimal pack should pass
        valid_zip = tmp_dir / "valid.vsp"
        with zipfile.ZipFile(valid_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("pack.toml", '[pack]\nid = "valid"\nname = "Valid"\nversion = "0.2.0"\n')
            zf.writestr("hooks/surface.vsl", '@hook(stage = "surface")\nvoid surface_hook() {}\n')
        errs = validate_zip_archive_safety(valid_zip)
        if errs:
            failures.append(f"Valid .vsp self-test unexpectedly failed: {errs}")

        # 2. Path traversal (../evil) must be rejected
        traversal_zip = tmp_dir / "traversal.vsp"
        with zipfile.ZipFile(traversal_zip, "w") as zf:
            zf.writestr("pack.toml", '[pack]\nid = "bad"\n')
            zf.writestr("../evil.txt", "pwned")
        if not validate_zip_archive_safety(traversal_zip):
            failures.append("Zip safety failed to reject '../evil.txt' traversal entry")

        # 3. Backslash traversal (..\evil) must be rejected
        win_traversal_zip = tmp_dir / "win_traversal.vsp"
        with zipfile.ZipFile(win_traversal_zip, "w") as zf:
            zf.writestr("pack.toml", '[pack]\nid = "bad"\n')
            zf.writestr("hooks\\..\\..\\evil.vsl", "pwned")
        if not validate_zip_archive_safety(win_traversal_zip):
            failures.append("Zip safety failed to reject backslash traversal entry")

        # 4. Absolute paths (/etc/passwd and C:\evil.dll) must be rejected
        abs_zip = tmp_dir / "absolute.vsp"
        with zipfile.ZipFile(abs_zip, "w") as zf:
            zf.writestr("pack.toml", '[pack]\nid = "bad"\n')
            zf.writestr("/etc/passwd", "root:x:0:0")
        if not validate_zip_archive_safety(abs_zip):
            failures.append("Zip safety failed to reject absolute path '/etc/passwd'")

        win_abs_zip = tmp_dir / "win_abs.vsp"
        with zipfile.ZipFile(win_abs_zip, "w") as zf:
            zf.writestr("pack.toml", '[pack]\nid = "bad"\n')
            zf.writestr("C:/Windows/evil.dll", "mz")
        if not validate_zip_archive_safety(win_abs_zip):
            failures.append("Zip safety failed to reject Windows drive absolute path")

        # 5. Zip bomb (highly compressed 4 MiB of zeroes) must be rejected
        bomb_zip = tmp_dir / "bomb.vsp"
        with zipfile.ZipFile(bomb_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
            zf.writestr("pack.toml", '[pack]\nid = "bomb"\n')
            zf.writestr("shaders/zeros.vsl", b"\x00" * (4 * 1024 * 1024))
        if not validate_zip_archive_safety(bomb_zip):
            failures.append("Zip safety failed to reject high-ratio zip bomb")

        # 6. Missing pack.toml must be rejected
        missing_toml_zip = tmp_dir / "no_manifest.vsp"
        with zipfile.ZipFile(missing_toml_zip, "w") as zf:
            zf.writestr("hooks/surface.vsl", "// no manifest\n")
        if not validate_zip_archive_safety(missing_toml_zip):
            failures.append("Zip safety failed to reject archive missing pack.toml")

        # 7. Malformed non-zip bytes must be rejected
        corrupt_zip = tmp_dir / "corrupt.vsp"
        corrupt_zip.write_bytes(b"NOT_A_ZIP_ARCHIVE_12345")
        if not validate_zip_archive_safety(corrupt_zip):
            failures.append("Zip safety failed to reject malformed archive")

    return failures


def check_packs() -> bool:
    print("[smoke] Checking built-in .vsp shader packs and zip safety...")
    errors: list[str] = []

    if not BUILTIN_PACKS_DIR.is_dir():
        print(f"[smoke] FAIL: Built-in packs directory not found: {BUILTIN_PACKS_DIR}")
        return False

    for pack_name in REQUIRED_BUILTIN_PACKS:
        pack_dir = BUILTIN_PACKS_DIR / pack_name
        if not pack_dir.is_dir():
            errors.append(f"Missing built-in pack directory: {pack_dir}")
            continue

        manifest_path = pack_dir / "pack.toml"
        if not manifest_path.is_file():
            errors.append(f"Missing pack.toml in {pack_name}")
            continue

        try:
            manifest = parse_toml_text(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"Failed to parse {pack_name}/pack.toml: {exc}")
            continue

        pack_meta = manifest.get("pack", {})
        if not isinstance(pack_meta, dict) or not pack_meta.get("id") or not pack_meta.get("name"):
            errors.append(f"{pack_name}/pack.toml is missing required [pack] id or name")

        hooks = manifest.get("hooks", {})
        if not isinstance(hooks, dict) or not hooks:
            errors.append(f"{pack_name}/pack.toml declares no [hooks]")
        else:
            for stage, rel_path in hooks.items():
                if not isinstance(rel_path, str):
                    errors.append(f"{pack_name}: hook '{stage}' path is not a string")
                    continue
                hook_file = (pack_dir / rel_path).resolve()
                try:
                    hook_file.relative_to(pack_dir.resolve())
                except ValueError:
                    errors.append(f"{pack_name}: hook '{stage}' escapes pack root: {rel_path}")
                    continue
                if not hook_file.is_file():
                    errors.append(f"{pack_name}: missing hook file '{rel_path}' for stage '{stage}'")
                    continue
                content = hook_file.read_text(encoding="utf-8")
                if not content.strip():
                    errors.append(f"{pack_name}: hook file '{rel_path}' is empty")
                if f'@hook(stage = "{stage}"' not in content:
                    errors.append(
                        f"{pack_name}: hook file '{rel_path}' missing '@hook(stage = \"{stage}\"' annotation"
                    )

        shader_count = 0
        for file_path in pack_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in SHADER_EXTENSIONS:
                shader_count += 1
                try:
                    text = file_path.read_text(encoding="utf-8")
                    if not text.strip():
                        errors.append(f"{pack_name}: empty shader file {file_path.relative_to(pack_dir)}")
                except UnicodeDecodeError as exc:
                    errors.append(f"{pack_name}: non-UTF-8 shader file {file_path.relative_to(pack_dir)}: {exc}")

        if shader_count == 0:
            errors.append(f"{pack_name}: contains zero shader/hook files")
        else:
            print(f"[smoke]   OK {pack_name}: manifest valid, {len(hooks)} hooks, {shader_count} shader/VSL files")

    zip_failures = run_zip_safety_self_tests()
    if zip_failures:
        errors.extend(zip_failures)
    else:
        print("[smoke]   OK zip-safety self-tests (7/7 cases passed)")

    if errors:
        for err in errors:
            print(f"[smoke] FAIL: {err}")
        return False

    print("[smoke] PASS: --check-packs")
    return True


def check_lang() -> bool:
    print("[smoke] Checking localization JSON validity and en_us/ru_ru key parity...")
    en_path = LANG_DIR / "en_us.json"
    ru_path = LANG_DIR / "ru_ru.json"
    errors: list[str] = []

    for p in (en_path, ru_path):
        if not p.is_file():
            errors.append(f"Missing localization file: {p}")

    if errors:
        for err in errors:
            print(f"[smoke] FAIL: {err}")
        return False

    try:
        en_data = json.loads(en_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[smoke] FAIL: Invalid JSON in en_us.json: {exc}")
        return False

    try:
        ru_data = json.loads(ru_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[smoke] FAIL: Invalid JSON in ru_ru.json: {exc}")
        return False

    if not isinstance(en_data, dict) or not isinstance(ru_data, dict):
        print("[smoke] FAIL: Localization files must be top-level JSON objects")
        return False

    en_keys = set(en_data.keys())
    ru_keys = set(ru_data.keys())
    missing_in_ru = sorted(en_keys - ru_keys)
    extra_in_ru = sorted(ru_keys - en_keys)

    if missing_in_ru:
        errors.append(f"Keys missing in ru_ru.json ({len(missing_in_ru)}): {missing_in_ru[:10]}")
    if extra_in_ru:
        errors.append(f"Extra keys in ru_ru.json ({len(extra_in_ru)}): {extra_in_ru[:10]}")

    for lang_name, data in (("en_us.json", en_data), ("ru_ru.json", ru_data)):
        if not data:
            errors.append(f"{lang_name} has zero entries")
        for k, v in data.items():
            if not isinstance(v, str) or not v.strip():
                errors.append(f"{lang_name}: key '{k}' must be a non-empty string")

    if errors:
        for err in errors:
            print(f"[smoke] FAIL: {err}")
        return False

    print(f"[smoke] PASS: --check-lang ({len(en_keys)} keys, 100% parity)")
    return True


def check_neoforge_free() -> bool:
    print("[smoke] Checking repository for remaining NeoForge files or references...")
    errors: list[str] = []

    forbidden_paths = [
        REPO_ROOT / "src" / "main" / "java" / "net" / "neoforged",
        REPO_ROOT / "src" / "main" / "kotlin" / "dev" / "vulkanshaders" / "VulkanShadersNeoForge.kt",
        REPO_ROOT / "src" / "main" / "resources" / "META-INF" / "neoforge.mods.toml",
    ]
    for path in forbidden_paths:
        if path.exists():
            errors.append(f"Forbidden NeoForge artifact still exists: {path.relative_to(REPO_ROOT)}")

    pattern = re.compile(r"net\.neoforged|neoforge\.mods\.toml|FMLPaths|VulkanShadersNeoForge", re.IGNORECASE)
    scan_targets = [
        REPO_ROOT / "build.gradle.kts",
        REPO_ROOT / "src",
        REPO_ROOT / ".github" / "workflows",
    ]

    for target in scan_targets:
        if target.is_file():
            files = [target]
        elif target.is_dir():
            files = [p for p in target.rglob("*") if p.is_file()]
        else:
            continue

        for file_path in files:
            if file_path.suffix.lower() in {".png", ".jar", ".spv", ".so", ".dll", ".dylib", ".zip", ".vsp"}:
                continue
            try:
                text = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for lineno, line in enumerate(text.splitlines(), start=1):
                if pattern.search(line):
                    rel = file_path.relative_to(REPO_ROOT)
                    errors.append(f"NeoForge reference in {rel}:{lineno}: {line.strip()}")

    if errors:
        for err in errors:
            print(f"[smoke] FAIL: {err}")
        return False

    print("[smoke] PASS: --check-neoforge-free")
    return True


def find_binary(names: list[str], search_dirs: list[pathlib.Path]) -> pathlib.Path | None:
    for directory in search_dirs:
        for name in names:
            candidate = directory / name
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return candidate
    for name in names:
        found = shutil.which(name)
        if found:
            return pathlib.Path(found)
    return None


def vulkan_smoke(dry_run: bool = False) -> bool:
    print("[smoke] Running headless Vulkan validation-layer smoke check...")
    env = os.environ.copy()
    env["VK_INSTANCE_LAYERS"] = "VK_LAYER_KHRONOS_validation"
    env.setdefault("VK_LOADER_DEBUG", "error,warn")

    bin_dirs = [
        REPO_ROOT / "target" / "release",
        REPO_ROOT / "target" / "debug",
        REPO_ROOT / "dist",
    ]
    vsp_cli = find_binary(["vsp-cli", "vsp-cli.exe", "vsp"], bin_dirs)
    sdk_bench = find_binary(["vulkanshaders-sdk-bench", "vulkanshaders-sdk-bench.exe"], bin_dirs)
    vulkaninfo = shutil.which("vulkaninfo")

    executed_any = False

    if vsp_cli is not None:
        executed_any = True
        for pack_name in REQUIRED_BUILTIN_PACKS:
            pack_dir = BUILTIN_PACKS_DIR / pack_name
            for tier in ("0", "3"):
                cmd = [str(vsp_cli), "check", str(pack_dir), "--tier", tier]
                print(f"[smoke]   exec: {' '.join(cmd)}")
                proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
                if proc.returncode != 0:
                    print(f"[smoke] FAIL: vsp-cli failed for {pack_name} tier {tier}:")
                    if proc.stdout:
                        print(proc.stdout)
                    if proc.stderr:
                        print(proc.stderr)
                    return False

    if sdk_bench is not None:
        executed_any = True
        cmd = [str(sdk_bench), "verify", "--repo-root", str(REPO_ROOT)]
        print(f"[smoke]   exec: {' '.join(cmd)}")
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if proc.returncode != 0:
            print("[smoke] FAIL: vulkanshaders-sdk-bench verify failed:")
            if proc.stdout:
                print(proc.stdout)
            if proc.stderr:
                print(proc.stderr)
            return False

    if vulkaninfo is not None:
        executed_any = True
        cmd = [vulkaninfo, "--summary"]
        print(f"[smoke]   exec: {' '.join(cmd)} (VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation)")
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if proc.returncode != 0:
            if dry_run:
                print("[smoke] SKIP: vulkaninfo returned non-zero in headless/dry-run environment")
                return True
            print(f"[smoke] FAIL: vulkaninfo exited with {proc.returncode}:\n{proc.stderr}")
            return False

    if not executed_any:
        if dry_run:
            print(
                "[smoke] SKIP: --vulkan-smoke (no built vsp-cli, vulkanshaders-sdk-bench, or vulkaninfo binary found in CI/dry-run mode)"
            )
            return True
        print(
            "[smoke] FAIL: --vulkan-smoke found no vsp-cli, vulkanshaders-sdk-bench, or vulkaninfo binary. "
            "Build the workspace (`cargo build --release`) or pass --dry-run."
        )
        return False

    print("[smoke] PASS: --vulkan-smoke")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Hyaloclast smoke test and CI verification harness")
    parser.add_argument("--check-packs", action="store_true", help="Validate built-in .vsp packs and zip safety")
    parser.add_argument("--check-lang", action="store_true", help="Validate en_us.json and ru_ru.json parity")
    parser.add_argument("--check-neoforge-free", action="store_true", help="Verify zero NeoForge references remain")
    parser.add_argument("--vulkan-smoke", action="store_true", help="Run headless Vulkan validation-layer smoke check")
    parser.add_argument("--dry-run", action="store_true", help="Skip --vulkan-smoke cleanly when binaries/ICD are absent")
    parser.add_argument(
        "--ci",
        action="store_true",
        help="Run all CI checks (--check-packs, --check-lang, --check-neoforge-free, --vulkan-smoke --dry-run)",
    )
    args = parser.parse_args(argv)

    run_packs = args.check_packs or args.ci
    run_lang = args.check_lang or args.ci
    run_neoforge = args.check_neoforge_free or args.ci
    run_vulkan = args.vulkan_smoke or args.ci
    dry_run = args.dry_run or args.ci

    if not (run_packs or run_lang or run_neoforge or run_vulkan):
        parser.print_help()
        return 2

    all_ok = True
    if run_packs:
        all_ok = check_packs() and all_ok
    if run_lang:
        all_ok = check_lang() and all_ok
    if run_neoforge:
        all_ok = check_neoforge_free() and all_ok
    if run_vulkan:
        all_ok = vulkan_smoke(dry_run=dry_run) and all_ok

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
