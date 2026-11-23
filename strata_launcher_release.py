"""Updates for the Windows launcher from its own GitHub Releases."""
import hashlib
import json
import os
import re
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path


REPO = "xumanzi/strata-launcher"
RELEASES = f"https://github.com/{REPO}/releases"
API = f"https://api.github.com/repos/{REPO}/releases/latest"
HEADERS = {"User-Agent": "AI-Manzi-Strata-Launcher", "Accept": "application/vnd.github+json"}


@dataclass(frozen=True)
class LauncherRelease:
    version: str
    name: str
    size: int
    sha256: str
    notes: str

    @property
    def url(self):
        return f"{RELEASES}/download/v{self.version}/{self.name}"


def release_version(value):
    if not isinstance(value, str) or not re.fullmatch(r"v?(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)", value):
        return ()
    return tuple(map(int, value.removeprefix("v").split(".")))


def parse_release(payload, current):
    tag = payload.get("tag_name")
    version = release_version(tag)
    if not version or version <= release_version(current) or payload.get("draft") or payload.get("prerelease"):
        return None
    clean = tag.removeprefix("v")
    if tag != "v" + clean:
        raise ValueError("发布标签须使用 v主版本.次版本.修订号。")
    name = f"AI-Manzi-Strata-Launcher-v{clean}-windows-x64.exe"
    candidates = [asset for asset in payload.get("assets", []) if asset.get("name") == name]
    if len(candidates) != 1 or candidates[0].get("state") != "uploaded":
        raise ValueError(f"GitHub 发布缺少完整的 Windows 安装文件：{name}")
    asset = candidates[0]
    digest, size = asset.get("digest"), asset.get("size")
    if not isinstance(digest, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
        raise ValueError("GitHub 发布缺少 SHA-256 校验值。")
    if type(size) is not int or not 0 < size <= 512 * 1024**2:
        raise ValueError("GitHub 发布的文件大小无效。")
    return LauncherRelease(clean, name, size, digest[7:], (payload.get("body") or "")[:4000])


class ReleaseAssetDigest(HTMLParser):
    def __init__(self, name):
        super().__init__()
        self.label = f"Copy to clipboard digest for {name}"
        self.digest = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "clipboard-copy" and attrs.get("aria-label") == self.label:
            self.digest = attrs.get("value")


def check_release_page(current):
    """Use the same repository's public release page when the REST quota is exhausted."""
    headers = {"User-Agent": HEADERS["User-Agent"]}
    with urllib.request.urlopen(urllib.request.Request(RELEASES + "/latest", headers=headers), timeout=15) as response:
        url = response.geturl()
    prefix = RELEASES + "/tag/"
    if not url.startswith(prefix):
        raise ValueError("GitHub 未返回可验证的最新发布标签。")
    tag = url[len(prefix):]
    version = release_version(tag)
    if not version or version <= release_version(current):
        return None
    name = f"AI-Manzi-Strata-Launcher-{tag}-windows-x64.exe"
    parser = ReleaseAssetDigest(name)
    with urllib.request.urlopen(urllib.request.Request(f"{RELEASES}/expanded_assets/{tag}", headers=headers),
                                timeout=15) as response:
        parser.feed(response.read(512 * 1024).decode("utf-8"))
    if not parser.digest or not re.fullmatch(r"sha256:[0-9a-f]{64}", parser.digest):
        raise ValueError("GitHub 公开发布页缺少目标安装文件的 SHA-256。")
    address = f"{RELEASES}/download/{tag}/{name}"
    with urllib.request.urlopen(urllib.request.Request(address, headers=headers, method="HEAD"),
                                timeout=15) as response:
        size = int(response.headers.get("Content-Length") or 0)
    return parse_release({"tag_name": tag, "assets": [{"name": name, "state": "uploaded",
                         "digest": parser.digest, "size": size}],
                         "body": "请在 GitHub 发布页查看此次更新说明。"}, current)


def check_launcher_release(current):
    try:
        with urllib.request.urlopen(urllib.request.Request(API, headers=HEADERS), timeout=15) as response:
            return parse_release(json.load(response), current)
    except urllib.error.HTTPError as error:
        if error.code in (403, 429):
            return check_release_page(current)
        if error.code == 404:
            return None
        raise


def stage_launcher(release, directory):
    """Keep the previous EXE available and publish the new one only after verification."""
    directory = Path(directory)
    target = directory / release.name
    if target.is_file() and target.stat().st_size == release.size:
        with target.open("rb") as stream:
            checksum = hashlib.sha256()
            for block in iter(lambda: stream.read(1024**2), b""):
                checksum.update(block)
            if checksum.hexdigest() == release.sha256:
                return target
    temporary = None
    try:
        request = urllib.request.Request(release.url, headers={"User-Agent": HEADERS["User-Agent"]})
        checksum = hashlib.sha256()
        total = 0
        with urllib.request.urlopen(request, timeout=45) as response:
            with tempfile.NamedTemporaryFile(dir=directory, prefix=".strata-launcher-", suffix=".part",
                                             delete=False) as stream:
                temporary = Path(stream.name)
                for block in iter(lambda: response.read(1024**2), b""):
                    total += len(block)
                    if total > release.size:
                        raise ValueError("下载文件超过发布标注的大小。")
                    checksum.update(block)
                    stream.write(block)
        if total != release.size or checksum.hexdigest() != release.sha256:
            raise ValueError("下载文件大小或 SHA-256 校验失败。")
        os.replace(temporary, target)
        return target
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
