"""Forward only to the configured origin, retaining the inspected request target."""

import time
from dataclasses import dataclass
from urllib.parse import urlsplit

import httpcore
from django.conf import settings
from django.http import HttpResponse

from .request import end_to_end


class UpstreamFailure(Exception):
    def __init__(self, reason, status=502):
        self.reason = reason
        self.status = status
        super().__init__(reason)


@dataclass
class UpstreamResponse:
    status: int
    headers: list[tuple[bytes, bytes]]
    body: bytes


class ProxyResponse(HttpResponse):
    """WSGI permits repeated fields, including independent Set-Cookie headers."""

    def __init__(self, upstream):
        super().__init__(upstream.body, status=upstream.status)
        del self.headers["Content-Type"]
        self.repeated_headers = []
        for name, value in end_to_end(upstream.headers):
            key, text = name.decode("ascii"), value.decode("latin-1")
            if key.lower().startswith("x-secbert-"):
                continue
            if key in self.headers:
                self.repeated_headers.append((key, text))
            else:
                self.headers[key] = text

    def items(self):
        return [*super().items(), *self.repeated_headers]


def forward(request):
    origin = urlsplit(settings.GATEWAY_UPSTREAM)
    host = origin.hostname.encode("idna")
    url = httpcore.URL(
        scheme=origin.scheme.encode("ascii"), host=host, port=origin.port, target=request.target
    )
    authority = b"[" + host + b"]" if b":" in host else host
    if origin.port is not None:
        authority += b":" + str(origin.port).encode("ascii")
    timeout = settings.GATEWAY_UPSTREAM_TIMEOUT
    deadline = time.monotonic() + timeout
    extensions = {"timeout": {name: timeout for name in ("connect", "read", "write", "pool")}}

    def check_deadline():
        if time.monotonic() >= deadline:
            raise UpstreamFailure("upstream_timeout", 504)

    try:
        with (
            httpcore.ConnectionPool(max_connections=1, retries=0) as pool,
            pool.stream(
                request.method,
                url,
                headers=request.upstream_headers(authority),
                content=request.body,
                extensions=extensions,
            ) as response,
        ):
            check_deadline()
            chunks, size = [], 0
            for chunk in response.iter_stream():
                check_deadline()
                size += len(chunk)
                if size > settings.GATEWAY_MAX_RESPONSE:
                    raise UpstreamFailure("upstream_response_too_large")
                chunks.append(chunk)
            # HEAD/204/empty responses may never yield a body chunk.
            check_deadline()
            return UpstreamResponse(response.status, response.headers, b"".join(chunks))
    except httpcore.TimeoutException as exc:
        raise UpstreamFailure("upstream_timeout", 504) from exc
    except (httpcore.NetworkError, httpcore.ProtocolError, ValueError) as exc:
        raise UpstreamFailure("upstream_error") from exc
