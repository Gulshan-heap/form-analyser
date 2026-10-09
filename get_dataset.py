"""Download one exercise from the Mendeley multi-view fitness video dataset (CC BY 4.0).
https://data.mendeley.com/datasets/kgbb3yn47p/2

The full zip is 7.6 GB because it holds every video three times (one copy per folder layout), so this
reads the zip's index and fetches only the chosen videos with HTTP range requests (~400 MB for squat).

  python get_dataset.py squat            # -> data/videos/squat/subject_001_squat_good_front.mp4 ...
  python get_dataset.py push_up --subject 001 --subject 002
"""
import argparse
import hashlib
import os
import struct
import urllib.request
import zlib
import zipfile

URL = "https://data.mendeley.com/public-api/zip/kgbb3yn47p/download/2"
HEADERS = {"User-Agent": "curl/8.4.0"}      # Mendeley's CDN rejects Python's default user agent
LAYOUT = "Dataset Exercise Quality-wise"    # one flat folder per exercise/quality


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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exercise", help="squat, push_up, bicep_curl, shoulder, tricep, abs or back")
    ap.add_argument("--out", default="data/videos")
    ap.add_argument("--subject", action="append", help="only these subject ids, e.g. 001 (repeatable)")
    args = ap.parse_args()

    infos = zipfile.ZipFile(RemoteFile()).infolist()
    want = [i for i in infos if i.filename.endswith(".mp4") and f"/{LAYOUT}/{args.exercise}/" in i.filename
            and (not args.subject or any(f"subject_{s}_" in i.filename for s in args.subject))]
    if not want:
        raise SystemExit(f"No videos found for '{args.exercise}'.")
    out = os.path.join(args.out, args.exercise)
    os.makedirs(out, exist_ok=True)
    print(f"{len(want)} videos, {sum(i.file_size for i in want) / 1e6:.0f} MB -> {out}")
    for n, info in enumerate(sorted(want, key=lambda i: i.filename), 1):
        dest = os.path.join(out, os.path.basename(info.filename))
        if os.path.exists(dest) and os.path.getsize(dest) == info.file_size:
            continue
        for attempt in range(3):
            try:
                fetch_member(info, dest)
                break
            except Exception as e:                      # signed links expire; just retry
                print(f"  retry {os.path.basename(dest)}: {e}")
        else:
            raise SystemExit(f"Failed: {dest}")
        print(f"  [{n}/{len(want)}] {os.path.basename(dest)}")

    seen = {}
    for f in sorted(os.listdir(out)):                  # flag mislabelled duplicates (same bytes, different label)
        h = hashlib.md5(open(os.path.join(out, f), "rb").read()).hexdigest()
        seen.setdefault(h, []).append(f)
    for names in seen.values():
        if len(names) > 1:
            print("WARNING identical files with different names (one label must be wrong):", names)


if __name__ == "__main__":
    main()
