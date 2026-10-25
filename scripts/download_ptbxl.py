"""下载 PTB-XL v1.0.3 metadata 和 records100 WFDB 文件。

默认只下载小型 metadata 文件；使用 ``--waveforms`` 后，会依据
ptbxl_database.csv 的 filename_lr 逐条下载 *_lr.hea 和 *_lr.dat。
脚本只使用 Python 标准库，支持断点续传、重试和官方 SHA256SUMS 校验。
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import os
import re
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

DEFAULT_BASE_URL = "https://physionet.org/files/ptb-xl/1.0.3/"
DEFAULT_FILES = (
    "ptbxl_database.csv",
    "scp_statements.csv",
    "RECORDS",
    "LICENSE.txt",
    "SHA256SUMS.txt",
)


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_destination(root: Path, relative: str) -> Path:
    """将远端相对路径限制在 output-dir 内，防止路径穿越。"""
    relative = relative.replace("\\", "/").lstrip("/")
    target = (root / relative).resolve()
    root_resolved = root.resolve()
    if target != root_resolved and root_resolved not in target.parents:
        raise ValueError(f"非法远端路径: {relative}")
    return target


def build_request(url: str, start: int, username: str | None, password: str | None) -> Request:
    headers = {"User-Agent": "PTBXL-iAIM-downloader/1.0"}
    if start:
        headers["Range"] = f"bytes={start}-"
    if username and password:
        token = base64.b64encode(f"{username}:{password}".encode()).decode()
        headers["Authorization"] = f"Basic {token}"
    return Request(url, headers=headers)


def download_file(
    url: str,
    destination: Path,
    expected_sha256: str | None = None,
    username: str | None = None,
    password: str | None = None,
    timeout: int = 120,
    retries: int = 5,
    resume: bool = True,
) -> None:
    """下载一个文件；网络中断时保留部分文件并自动重试。"""
    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        existing = destination.stat().st_size if resume and destination.exists() else 0
        try:
            with urlopen(build_request(url, existing, username, password), timeout=timeout) as response:
                append = existing > 0 and getattr(response, "status", 200) == 206
                mode = "ab" if append else "wb"
                with destination.open(mode) as handle:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
            if expected_sha256:
                actual = sha256(destination)
                if actual.lower() != expected_sha256.lower():
                    raise IOError(
                        f"SHA-256 校验失败 {destination}: expected={expected_sha256}, actual={actual}"
                    )
            return
        except (HTTPError, URLError, TimeoutError, OSError, IOError) as exc:
            if attempt == retries:
                raise RuntimeError(f"下载失败（{retries} 次尝试）: {url}\n{exc}") from exc
            wait = min(2 ** (attempt - 1), 30)
            print(f"  网络中断，{wait}s 后重试 ({attempt}/{retries}): {exc}")
            time.sleep(wait)


def resolve_url(base_url: str, relative: str) -> str:
    return base_url.rstrip("/") + "/" + quote(relative.lstrip("/"), safe="/:._-")


def load_manifest(path: str | None) -> tuple[str, list[dict]]:
    if path is None:
        return DEFAULT_BASE_URL, [{"path": item} for item in DEFAULT_FILES]
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    return manifest.get("base_url", DEFAULT_BASE_URL), manifest.get("files", [])


def download_metadata(
    root: Path,
    base_url: str,
    items: list[dict],
    username: str | None,
    password: str | None,
    dry_run: bool,
    resume: bool,
) -> list[dict]:
    downloaded: list[dict] = []
    for item in items:
        relative = str(item["path"]).replace("\\", "/")
        if relative.endswith("/"):
            print(f"跳过目录项（不会递归下载）: {relative}")
            continue
        url = item.get("url") or resolve_url(base_url, relative)
        destination = safe_destination(root, relative)
        print(f"{'计划' if dry_run else '下载'} {url} -> {destination}")
        if not dry_run:
            download_file(url, destination, item.get("sha256"), username, password, resume=resume)
        downloaded.append({"path": relative, "url": url, "destination": destination})
    return downloaded


def read_record_bases(root: Path, limit: int | None = None) -> list[str]:
    metadata = root / "ptbxl_database.csv"
    if not metadata.exists():
        raise FileNotFoundError(f"缺少 {metadata}，请先下载 metadata")
    with metadata.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if "filename_lr" not in (reader.fieldnames or []):
            raise ValueError("ptbxl_database.csv 缺少 filename_lr 列")
        bases: list[str] = []
        seen: set[str] = set()
        for row in reader:
            value = (row.get("filename_lr") or "").strip().replace("\\", "/")
            if not value or value in seen:
                continue
            if not value.startswith("records100/"):
                raise ValueError(f"filename_lr 不是 records100 路径: {value}")
            seen.add(value)
            bases.append(value)
            if limit is not None and len(bases) >= limit:
                break
    return bases


def verify_downloads(root: Path) -> tuple[int, int]:
    """按官方 SHA256SUMS.txt 校验本地已存在文件，返回 (通过, 失败)。"""
    sums_path = root / "SHA256SUMS.txt"
    if not sums_path.exists():
        raise FileNotFoundError(f"缺少 {sums_path}")
    passed = failed = 0
    pattern = re.compile(r"^([0-9a-fA-F]{64})\\s+[*]?(.+?)\\s*$")
    for line in sums_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.match(line)
        if not match:
            continue
        expected, relative = match.groups()
        relative = relative.lstrip("./").replace("\\", "/")
        path = safe_destination(root, relative)
        if not path.exists():
            continue
        actual = sha256(path)
        if actual.lower() == expected.lower():
            passed += 1
        else:
            failed += 1
            print(f"校验失败: {relative}")
    return passed, failed


def main() -> None:
    parser = argparse.ArgumentParser(description="PTB-XL v1.0.3 metadata/100 Hz 波形下载器")
    parser.add_argument("--output-dir", default=os.getenv("PTBXL_DATA_DIR", "data/ptbxl"))
    parser.add_argument("--manifest", default=None, help="JSON manifest；默认使用 v1.0.3 官方文件清单")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--metadata-only", action="store_true", help="只下载 metadata、许可证和校验清单（默认行为）")
    parser.add_argument("--waveforms", action="store_true", help="依据 filename_lr 下载 records100 的 .hea/.dat")
    parser.add_argument("--limit", type=int, default=None, help="仅下载前 N 条 records100 记录，用于 smoke test")
    parser.add_argument("--dry-run", action="store_true", help="只显示计划，不写入文件")
    parser.add_argument("--verify", action="store_true", help="下载后用官方 SHA256SUMS 校验已存在文件")
    parser.add_argument("--no-resume", action="store_true", help="不续传，重新覆盖每个文件")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit 必须为正整数")
    if args.metadata_only and args.waveforms:
        parser.error("--metadata-only 与 --waveforms 不能同时使用")

    base_url, items = load_manifest(args.manifest)
    base_url = args.base_url or base_url
    username, password = os.getenv("PHYSIONET_USERNAME"), os.getenv("PHYSIONET_PASSWORD")
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    metadata_items = download_metadata(root, base_url, items, username, password, args.dry_run, not args.no_resume)

    if args.waveforms:
        if args.dry_run:
            bases = read_record_bases(root, args.limit) if (root / "ptbxl_database.csv").exists() else []
            if not bases:
                print("dry-run 无法枚举波形：请先下载 ptbxl_database.csv，或去掉 --dry-run")
        else:
            bases = read_record_bases(root, args.limit)
        for base in bases:
            for suffix in (".hea", ".dat"):
                relative = base + suffix
                url = resolve_url(base_url, relative)
                destination = safe_destination(root, relative)
                print(f"{'计划' if args.dry_run else '下载'} {url} -> {destination}")
                if not args.dry_run:
                    download_file(url, destination, username=username, password=password, resume=not args.no_resume)

    if args.verify and not args.dry_run:
        passed, failed = verify_downloads(root)
        print(f"SHA-256 校验：通过 {passed}，失败 {failed}")
        if failed:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
