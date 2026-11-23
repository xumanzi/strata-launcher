"""Parallel HTTP transport for the official updater; its checksum/signature checks stay intact."""
import concurrent.futures
import json
import http.client
import inspect
import runpy
import sys
import threading
import time
import urllib.request
from pathlib import Path


class RangeUnavailable(OSError):
    pass


class DownloadCancelled(OSError):
    pass


def discard_segments(folder):
    """Best-effort cache cleanup must never turn a completed transfer into another download."""
    for path in [*folder.glob('segment-*.part'), folder / 'state.json']:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
    try:
        folder.rmdir()
    except OSError:
        pass


def source_urls(url):
    import re
    match = re.fullmatch(r'https://github.com/([^/]+/[^/]+)/archive/refs/(tags|heads)/([^/?]+)\.zip', url)
    if match:
        repo, kind, ref = match.groups()
        return [f'https://codeload.github.com/{repo}/zip/refs/{kind}/{ref}', url]
    return [url]


def download_source(url, destination, log=print):
    last_error = None
    for address in source_urls(url):
        for attempt in range(2):
            try:
                log(f'正在下载 {address}')
                began, last_emit, done = time.monotonic(), 0.0, 0
                request = urllib.request.Request(address, headers={'User-Agent': 'StrataLauncher'})
                with urllib.request.urlopen(request, timeout=30) as response:
                    total = int(response.headers.get('Content-Length') or 0)
                    with Path(destination).open('wb') as stream:
                        for block in iter(lambda: response.read(1024*1024), b''):
                            stream.write(block)
                            done += len(block)
                            now = time.monotonic()
                            if now-last_emit >= 2 or done == total:
                                speed = done/max(now-began, .001)/1048576
                                log(f'  {done/1048576:.1f}/{total/1048576:.1f} MB，{speed:.2f} MB/s' if total else f'  {done/1048576:.1f} MB，{speed:.2f} MB/s')
                                last_emit = now
                if total and done != total:
                    raise OSError(f'源码下载不完整：{done}/{total} 字节')
                log('下载完成。')
                return destination
            except (OSError, TimeoutError, http.client.HTTPException) as exc:
                last_error = exc
                log(f'源码连接失败，正在重试或切换官方入口：{exc}')
                if attempt == 0:
                    time.sleep(1)
    raise OSError(f'官方源码下载失败：{last_error}') from last_error


def parallel_transfer(url, partial, have, size, progress, cancel, timeout,
                      workers=8, opener=None):
    partial = Path(partial)
    open_url = opener.open if opener is not None else urllib.request.urlopen
    stopped = threading.Event()

    def check_cancel():
        if stopped.is_set() or (cancel is not None and cancel.is_set()):
            raise DownloadCancelled('下载已取消；已有分段保留供下次续传。')

    check_cancel()
    request = urllib.request.Request(url, headers={'User-Agent': 'StrataLauncher', 'Range': 'bytes=0-0', 'Accept-Encoding': 'identity'})
    with open_url(request, timeout=min(timeout, 25)) as response:
        if response.status != 206 or response.headers.get('Content-Range') != f'bytes 0-0/{size}':
            raise RangeUnavailable('服务器未确认精确分段范围，使用官方单连接续传。')
        final_url = response.geturl()
        validator = response.headers.get('ETag') or response.headers.get('Last-Modified')
        if len(response.read(2)) != 1:
            raise RangeUnavailable('分段探测响应长度无效。')
    if have >= size:
        return
    workers = min(max(1, int(workers)), max(1, size - have))
    folder = partial.with_name(partial.name + '.ranges')
    folder.mkdir(parents=True, exist_ok=True)
    ranges = [(have + (size-have)*i//workers, have + (size-have)*(i+1)//workers - 1) for i in range(workers)]
    identity = {'url': url, 'size': size, 'prefix': have, 'validator': validator, 'ranges': [list(x) for x in ranges]}
    meta_path = folder / 'state.json'
    try:
        previous = json.loads(meta_path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        previous = None
    if previous != identity:
        for part in folder.glob('segment-*.part'):
            part.unlink()
        meta_path.write_text(json.dumps(identity), encoding='utf-8')
    files = [folder / f'segment-{i}.part' for i in range(workers)]
    lengths = []
    for path, (start, end) in zip(files, ranges):
        length = path.stat().st_size if path.exists() else 0
        if length > end-start+1:
            path.unlink()
            length = 0
        lengths.append(length)
    lock = threading.Lock()
    last_emit = [0.0]

    def emit(force=False):
        with lock:
            now = time.monotonic()
            if progress and (force or now-last_emit[0] >= .5):
                progress(have + sum(lengths), size)
                last_emit[0] = now

    emit(True)

    def fetch(index):
        start, end = ranges[index]
        for attempt in range(3):
            check_cancel()
            offset = files[index].stat().st_size if files[index].exists() else 0
            if offset == end-start+1:
                return
            headers = {'User-Agent': 'StrataLauncher', 'Range': f'bytes={start+offset}-{end}', 'Accept-Encoding': 'identity'}
            if validator:
                headers['If-Range'] = validator
            try:
                with open_url(urllib.request.Request(final_url, headers=headers), timeout=min(timeout, 25)) as response:
                    expected = f'bytes {start+offset}-{end}/{size}'
                    if response.status != 206 or response.headers.get('Content-Range') != expected:
                        raise RangeUnavailable('服务器返回的分段边界不匹配。')
                    returned_validator = response.headers.get('ETag') or response.headers.get('Last-Modified')
                    if validator and returned_validator and validator != returned_validator:
                        raise RangeUnavailable('下载期间服务器文件发生变化。')
                    remaining = end-start+1-offset
                    with files[index].open('ab') as stream:
                        while remaining:
                            check_cancel()
                            block = response.read(min(256*1024, remaining))
                            if not block:
                                raise OSError('分段连接提前结束')
                            stream.write(block)
                            remaining -= len(block)
                            with lock:
                                lengths[index] += len(block)
                            emit()
                        if response.read(1):
                            raise RangeUnavailable('服务器发送了超出请求范围的数据。')
                return
            except (RangeUnavailable, DownloadCancelled):
                raise
            except (OSError, TimeoutError, http.client.HTTPException):
                if attempt == 2:
                    raise
                # File close flushes the last successful block; resume only persisted bytes.
                with lock:
                    lengths[index] = files[index].stat().st_size if files[index].exists() else 0
                if stopped.wait(.5 * (attempt+1)):
                    check_cancel()

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, i) for i in range(workers)]
        try:
            for future in concurrent.futures.as_completed(futures):
                future.result()
        except BaseException:
            stopped.set()
            for future in futures:
                future.cancel()
            raise
    check_cancel()
    # Never preallocate the official partial: an incomplete parallel run must not look complete.
    try:
        with partial.open('r+b' if partial.exists() else 'w+b') as target:
            target.seek(have)
            for path in files:
                with path.open('rb') as source:
                    for block in iter(lambda: source.read(1024*1024), b''):
                        check_cancel()
                        target.write(block)
            target.truncate(size)
    except BaseException:
        with partial.open('r+b') as target:
            target.truncate(have)
        raise
    emit(True)
    discard_segments(folder)


def install_transport(transfer, threshold=16*1024*1024):
    if not all(hasattr(transfer, name) for name in ('_transfer', '_opener', 'kind_of', 'check_url_policy')):
        print(json.dumps({'event': 'phase', 'message': '此版官方传输接口不适用加速，继续使用原下载器。'}, ensure_ascii=False), flush=True)
        return
    original = transfer._transfer
    try:
        parameters = tuple(inspect.signature(original).parameters)
    except (TypeError, ValueError):
        parameters = ()
    # Releases with a pre-opened response or new options must keep the official transfer.
    if parameters != ('location', 'partial', 'have', 'size', 'progress', 'cancel', 'timeout'):
        print(json.dumps({'event': 'phase', 'message': '使用此发行版自带的下载器。'}, ensure_ascii=False), flush=True)
        return

    def accelerated(location, partial, have, size, progress, cancel, timeout):
        if size-have < threshold or transfer.kind_of(location) not in ('http', 'https'):
            return original(location, partial, have, size, progress, cancel, timeout)
        transfer.check_url_policy(location)
        try:
            print(json.dumps({'event': 'phase', 'message': '正在使用 8 连接分段下载（支持续传），完成后继续官方完整校验。'}, ensure_ascii=False), flush=True)
            return parallel_transfer(location, partial, have, size, progress, cancel, timeout,
                                     opener=transfer._opener)
        except DownloadCancelled:
            raise transfer.Cancelled()
        except (OSError, TimeoutError, http.client.HTTPException) as exc:
            print(json.dumps({'event': 'phase', 'message': f'并行下载不可用，自动改用官方单连接续传：{exc}'}, ensure_ascii=False), flush=True)
            result = original(location, partial, have, size, progress, cancel, timeout)
            discard_segments(Path(partial).with_name(Path(partial).name + '.ranges'))
            return result

    transfer._transfer = accelerated


def main():
    root = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(root))
    from strata_control import transfer
    install_transport(transfer)
    sys.argv = [str(root / 'strata.py'), *sys.argv[2:]]
    runpy.run_path(sys.argv[0], run_name='__main__')


if __name__ == '__main__':
    main()
