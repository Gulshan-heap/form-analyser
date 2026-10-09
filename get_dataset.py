"""Download one exercise from the Mendeley multi-view fitness video dataset (CC BY 4.0).
https://data.mendeley.com/datasets/kgbb3yn47p/3

The full zip is several GB because it holds every video three times (one copy per folder layout), so this
reads the zip's index and fetches only the chosen videos with HTTP range requests (~400 MB for squat).
Meant to run on Colab / Kaggle (see train_mendeley.ipynb), not on a laptop.

  python get_dataset.py squat                     # -> data/videos/squat/subject_001_squat_good_front.mp4 ...
  python get_dataset.py squat --max-subjects 3    # quick test
  python get_dataset.py push_up --version 2
"""
import argparse
import hashlib
import os
import re
import struct
import urllib.error
import urllib.request
import zlib
import zipfile
from concurrent.futures import ThreadPoolExecutor

URL = "https://data.mendeley.com/public-api/zip/kgbb3yn47p/download/3"
HEADERS = {"User-Agent": "curl/8.4.0"}      # Mendeley's CDN rejects Python's default user agent
FILE_RX = re.compile(r"subject_(\d+)_(.+)_(good|bad)_(front|side|diagonal)\.mp4$", re.I)


def set_version(version):
    global URL
    URL = f"https://data.mendeley.com/public-api/zip/kgbb3yn47p/download/{version}"


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None


def signed_url():
    """The zip endpoint answers with a short-lived signed S3 link, so ask again for every request."""
    try:
        urllib.request.build_opener(_NoRedirect).open(urllib.request.Request(URL, headers=HEADERS), timeout=60)
    except urllib.error.HTTPError as e:
        if e.code in (301, 302, 307):
            return e.headers["location"]
        raise
    raise RuntimeError("Mendeley did not return a download link")


def get_range(a, b):
    req = urllib.request.Request(signed_url(), headers={**HEADERS, "Range": f"bytes={a}-{b}"})
    return urllib.request.urlopen(req, timeout=300)


class RemoteFile:
    """Seekable read-only file over HTTP range requests, just enough for zipfile to read the index."""

    def __init__(self):
        self.pos = 0
        self.size = int(get_range(0, 0).headers["Content-Range"].split("/")[1])

    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def read(self, n=-1):
        n = self.size - self.pos if n < 0 else min(n, self.size - self.pos)
        if n <= 0:
            return b""
        data = get_range(self.pos, self.pos + n - 1).read()
        self.pos += len(data)
        return data


def list_videos():
    """Every labelled video in the zip, once (the zip stores each video in three folder layouts)."""
    seen = {}
    for info in zipfile.ZipFile(RemoteFile()).infolist():
        name = os.path.basename(info.filename)
        m = FILE_RX.match(name)
        if m and name not in seen:
            seen[name] = (info, m.group(2).lower(), m.group(3).lower(), m.group(4).lower(), m.group(1))
    return seen


def fetch_member(info, dest):
    """Download one zip member (stored or deflated) straight from its byte range."""
    head = get_range(info.header_offset, info.header_offset + 30 + 2048).read()
    method = struct.unpack("<H", head[8:10])[0]
    name_len, extra_len = struct.unpack("<HH", head[26:30])
    start = info.header_offset + 30 + name_len + extra_len
    raw = get_range(start, start + info.compress_size - 1).read()
    data = raw if method == 0 else zlib.decompressobj(-15).decompress(raw)
    if len(data) != info.file_size:
        raise IOError(f"{info.filename}: got {len(data)} bytes, expected {info.file_size}")
    with open(dest + ".part", "wb") as f:
        f.write(data)
    os.replace(dest + ".part", dest)


def find_conflicts(folder):
    """Videos with identical bytes but different good/bad labels: one label must be wrong, so drop both."""
    groups = {}
    for f in sorted(os.listdir(folder)):
        if f.endswith(".mp4"):
            groups.setdefault(hashlib.md5(open(os.path.join(folder, f), "rb").read()).hexdigest(), []).append(f)
    bad = []
    for names in groups.values():
        if len(names) > 1 and len({FILE_RX.match(n).group(3).lower() for n in names}) > 1:
            print("WARNING identical videos with different labels, excluded from training:", names)
            bad += names
    with open(os.path.join(folder, "DUPLICATES.txt"), "w") as f:
        f.write("\n".join(bad))
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exercise", help="squat, push_up, bicep_curl, shoulder, tricep, abs or back")
    ap.add_argument("--out", default="data/videos")
    ap.add_argument("--version", type=int, default=3)
    ap.add_argument("--workers", type=int, default=6, help="parallel downloads")
    ap.add_argument("--subject", action="append", help="only these subject ids, e.g. 001 (repeatable)")
    ap.add_argument("--max-subjects", type=int, default=0, help="only the first N subjects (quick test)")
    args = ap.parse_args()
    set_version(args.version)

    videos = list_videos()
    exercises = sorted({v[1] for v in videos.values()})
    want = {n: v for n, v in videos.items() if v[1] == args.exercise.lower()}
    if not want:
        raise SystemExit(f"No videos found for '{args.exercise}'. Exercises in version {args.version}: {exercises}")
    subjects = sorted({v[4] for v in want.values()})
    if args.subject:
        subjects = [s for s in subjects if s in set(args.subject)]
    if args.max_subjects:
        subjects = subjects[:args.max_subjects]
    want = {n: v for n, v in want.items() if v[4] in subjects}
    print(f"version {args.version}: exercises {exercises}")
    print(f"{len(want)} {args.exercise} videos from {len(subjects)} subjects, "
          f"{sum(v[0].file_size for v in want.values()) / 1e6:.0f} MB")

    out = os.path.join(args.out, args.exercise.lower())
    os.makedirs(out, exist_ok=True)

    def download(item):
        name, (info, *_) = item
        dest = os.path.join(out, name)
        if os.path.exists(dest) and os.path.getsize(dest) == info.file_size:
            return
        for attempt in range(4):
            try:
                return fetch_member(info, dest)
            except Exception as e:                      # signed links expire / CDN hiccups: just retry
                print(f"  retry {name}: {e}")
        raise RuntimeError(f"failed: {dest}")

    with ThreadPoolExecutor(args.workers) as pool:      # several videos at once: the CDN is slow per request
        for n, _ in enumerate(pool.map(download, sorted(want.items())), 1):
            if n % 10 == 0 or n == len(want):
                print(f"  [{n}/{len(want)}] downloaded", flush=True)
    find_conflicts(out)


if __name__ == "__main__":
    main()
