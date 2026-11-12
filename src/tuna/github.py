"""Read-only GitHub REST client with an on-disk response cache.

Every GET is cached under <cache>/<owner>__<repo>/<sha256 of the URL>.json and
never fetched again, so a rerun costs no requests. Unauthenticated by default
(60 core requests/hour, 10 search requests/minute per IP); pass a token only
through an environment variable the caller names explicitly.

When a rate limit runs out the client sleeps until its reset, unless that is
more than `max_wait` seconds away, in which case it raises RateLimited.
"""
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

API = "https://api.github.com"


class RateLimited(RuntimeError):
    pass


class GitHub:
    def __init__(self, cache_dir, token=None, max_wait=120, log=print):
        self.cache_dir, self.token, self.max_wait, self.log = cache_dir, token, max_wait, log
        self.requests = {"core": 0, "search": 0}   # network requests made by this process
        self.cache_hits = 0
        self.remaining = {}                          # resource -> (remaining, reset epoch)

    def _path(self, url):
        parts = urllib.parse.urlsplit(url).path.split("/")
        scope = "__".join(parts[2:4]).lower() if len(parts) > 3 and parts[1] == "repos" else "search"
        if parts[1:2] == ["search"]:
            q = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query).get("q", [""])[0]
            repo = next((t[5:] for t in q.split() if t.startswith("repo:")), "any")
            scope = "search__" + repo.replace("/", "__").lower()
        return os.path.join(self.cache_dir, scope, hashlib.sha256(url.encode()).hexdigest() + ".json")

    def get(self, path, **params):
        """JSON for GET path?params as {"status", "body"}; 404/410/422 are returned, not raised."""
        url = API + path + ("?" + urllib.parse.urlencode(params) if params else "")
        cpath = self._path(url)
        if os.path.exists(cpath):
            self.cache_hits += 1
            with open(cpath, encoding="utf-8") as fh:
                return json.load(fh)
        resource = "search" if path.startswith("/search/") else "core"
        for attempt in range(6):
            self._wait(resource)
            req = urllib.request.Request(url, headers={
                "Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "tuna-issues", **({"Authorization": f"Bearer {self.token}"} if self.token else {})})
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    status, headers, body = resp.status, resp.headers, json.load(resp)
            except urllib.error.HTTPError as e:
                status, headers = e.code, e.headers
                try:
                    body = json.loads(e.read() or b"null")
                except ValueError:
                    body = None
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                self.log(f"[github] {url}: {e}; retry {attempt + 1}")
                time.sleep(5 * (attempt + 1))
                continue
            self.requests[resource] += 1
            self._note(resource, headers)
            limited = status == 429 or (status == 403 and (headers.get("X-RateLimit-Remaining") == "0"
                                                            or "rate limit" in json.dumps(body).lower()))
            if limited:
                retry = headers.get("Retry-After")
                if retry:
                    self.remaining[resource] = (0, time.time() + int(retry))
                continue
            if status >= 500:
                time.sleep(5 * (attempt + 1))
                continue
            if status not in (200, 404, 410, 422):
                raise RuntimeError(f"GET {url}: HTTP {status}: {json.dumps(body)[:300]}")
            out = {"url": url, "status": status, "body": body,
                   "fetched": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            os.makedirs(os.path.dirname(cpath), exist_ok=True)
            with open(cpath + ".tmp", "w", encoding="utf-8") as fh:
                json.dump(out, fh, ensure_ascii=False)
            os.replace(cpath + ".tmp", cpath)
            return out
        raise RuntimeError(f"GET {url}: gave up after retries")

    def _note(self, resource, headers):
        rem, reset = headers.get("X-RateLimit-Remaining"), headers.get("X-RateLimit-Reset")
        if rem is not None and reset is not None:
            self.remaining[resource] = (int(rem), int(reset))

    def _wait(self, resource):
        rem, reset = self.remaining.get(resource, (1, 0))
        if rem > 0:
            return
        delay = reset - time.time() + 2
        if delay <= 0:
            return
        if delay > self.max_wait:
            raise RateLimited(f"{resource} rate limit exhausted; resets in {int(delay)}s "
                              f"(after {self.requests[resource]} {resource} requests this run)")
        self.log(f"[github] {resource} limit reached; sleeping {int(delay)}s")
        time.sleep(delay)

    def search_issues(self, q, created=None, sort="created", order="asc"):
        """Every result of an issue/PR search. GitHub returns at most 1000 per query,
        so a larger result set is split into created-date ranges until each fits."""
        lo, hi = created or ("2000-01-01", time.strftime("%Y-%m-%d", time.gmtime()))
        first = self.get("/search/issues", q=f"{q} created:{lo}..{hi}", sort=sort, order=order, per_page=100, page=1)
        if first["status"] != 200:
            raise RuntimeError(f"search {q!r}: HTTP {first['status']}: {first['body']}")
        total = first["body"]["total_count"]
        if total > 1000 and lo != hi:
            mid = _mid_date(lo, hi)
            return self.search_issues(q, (lo, mid), sort, order) + self.search_issues(q, (_next_day(mid), hi), sort, order)
        items, page = list(first["body"]["items"]), 1
        while len(items) < min(total, 1000):
            page += 1
            r = self.get("/search/issues", q=f"{q} created:{lo}..{hi}", sort=sort, order=order, per_page=100, page=page)
            if r["status"] != 200 or not r["body"]["items"]:
                break
            items += r["body"]["items"]
        if total > 1000:
            print(f"[github] warning: {total} results on {lo} alone for {q!r}; kept the first 1000", file=sys.stderr)
        return items


def _mid_date(lo, hi):
    a = time.mktime(time.strptime(lo, "%Y-%m-%d"))
    b = time.mktime(time.strptime(hi, "%Y-%m-%d"))
    return time.strftime("%Y-%m-%d", time.localtime((a + b) / 2))


def _next_day(d):
    return time.strftime("%Y-%m-%d", time.localtime(time.mktime(time.strptime(d, "%Y-%m-%d")) + 36 * 3600))
