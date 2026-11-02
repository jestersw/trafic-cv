from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

ARCHIVES = {
    "DETRAC-Images.zip": "1fQC6CEWOeL9pnJ4ZdD7U2gromcnWDRgm",
    "DETRAC-Train-Annotations-XML.zip": "1f22g3puG164gemY_njcPG8beuCcuMjqb",
    "DETRAC-Test-Annotations-XML.zip": "1zc8KZWUa3Cw0DbcFqTCMBmUZzbfuhGKr",
}


def drive_url(file_id: str) -> str:
    return f"https://drive.google.com/file/d/{file_id}/view"


def fetch(archives: Path, names: list[str]) -> list[str]:
    import gdown

    missing = []
    for name in names:
        file_id = ARCHIVES[name]
        target = archives / name
        if target.exists() and zipfile.is_zipfile(target):
            continue
        try:
            gdown.download(id=file_id, output=str(target), quiet=False, resume=True)
        except Exception as e:
            print(f"[gdown] {name}: {type(e).__name__}")
        if not (target.exists() and zipfile.is_zipfile(target)):
            missing.append(name)
    return missing


def extract(archive: Path, dest: Path) -> None:
    with zipfile.ZipFile(archive) as zf:
        members = [m for m in zf.namelist() if not m.startswith("__MACOSX")]
        print(f"extracting {archive.name}: {len(members)} entries")
        zf.extractall(dest, members)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Download and unpack UA-DETRAC")
    ap.add_argument("--out", default="data/raw/UA-DETRAC")
    ap.add_argument("--archives", default=None, help="folder with the zip files")
    ap.add_argument("--no-download", action="store_true", help="only extract existing zips")
    ap.add_argument("--delete-archives", action="store_true")
    args = ap.parse_args(argv)

    out = Path(args.out)
    archives = Path(args.archives) if args.archives else out.parent / "_archives"
    archives.mkdir(parents=True, exist_ok=True)

    unpacked = {n for n in ARCHIVES if (archives / n[:-4]).is_dir() or (out / n[:-4]).is_dir()}
    pending = [n for n in ARCHIVES if n not in unpacked]
    if args.no_download:
        missing = [n for n in pending if not (archives / n).exists()]
    else:
        missing = fetch(archives, pending)
    if missing:
        print(f"\nCould not download automatically. Download in a browser into {archives}/:")
        for name in missing:
            print(f"  {name}: {drive_url(ARCHIVES[name])}")
        print(
            "then run from the repo root: uv run python -m traffic_cv.data.download --no-download"
        )
        raise SystemExit(1)

    out.mkdir(parents=True, exist_ok=True)
    for name in ARCHIVES:
        folder = archives / name[:-4]
        if folder.is_dir() and not (out / folder.name).exists():
            print(f"moving unpacked {folder.name} -> {out}")
            shutil.move(str(folder), str(out / folder.name))
        elif name in pending:
            extract(archives / name, out)
    if args.delete_archives:
        for name in pending:
            (archives / name).unlink()
    print(f"done: {out}")


if __name__ == "__main__":
    main()
