"""Preflight the local release artifact and optionally verify updater discovery."""
import argparse
import hashlib
import subprocess
from pathlib import Path

from strata_launcher_release import check_launcher_release, release_version


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version")
    parser.add_argument("asset", type=Path)
    parser.add_argument("--verify-from", help="Verify public discovery from this older version")
    args = parser.parse_args()
    version = args.version
    name = f"AI-Manzi-Strata-Launcher-v{version}-windows-x64.exe"
    if not release_version(version) or args.asset.name != name or not args.asset.is_file():
        parser.error(f"Expected a version and an existing release asset named {name}")
    size = args.asset.stat().st_size
    if not 0 < size <= 512 * 1024**2:
        parser.error("Invalid asset size")
    for check in (["--check-release-version", version], ["--self-test"]):
        subprocess.run([str(args.asset.resolve()), *check], check=True, timeout=90)
    with args.asset.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if args.verify_from:
        release = check_launcher_release(args.verify_from)
        actual = (release.version, release.name, release.size, release.sha256) if release else None
        if actual != (version, name, size, digest):
            raise SystemExit("The public updater did not find the exact verified build")
    print(f"{digest}  {name}")


if __name__ == "__main__":
    main()
