from __future__ import annotations

import re
from dataclasses import dataclass
from ipaddress import ip_address
from urllib.parse import quote

from django.conf import settings
from django.core.exceptions import RequestDataTooBig

HOP_HEADERS = {
    b"connection",
    b"keep-alive",
    b"proxy-authenticate",
    b"proxy-authorization",
    b"proxy-connection",
    b"te",
    b"trailer",
    b"transfer-encoding",
    b"upgrade",
}
REQUEST_ONLY = {b"content-length", b"expect", b"forwarded", b"x-real-ip"}
METHOD = re.compile(rb"[!#$%&'*+.^_`|~0-9A-Za-z-]{1,64}\Z")
TARGET = re.compile(rb"[!-~]+\Z")
HEADER_NAME = re.compile(rb"[!#$%&'*+.^_`|~0-9A-Za-z-]+\Z")


def from_django(request):
    """Capture the same target/body used by both inspectors and the upstream."""
    meta = request.META
    target = meta.get("RAW_URI") or meta.get("REQUEST_URI")
    if target is None:
        # runserver/test client fallback; Waitress supplies the original REQUEST_URI.
        path = quote(request.path, safe="/!$&'()*+,;=:@-._~")
        query = meta.get("QUERY_STRING", "")
        target = path + ("?" + query if query else "")
    headers = []
    for key, value in meta.items():
        if key.startswith("HTTP_"):
            headers.append((key[5:].replace("_", "-").encode("ascii"), value.encode("latin-1")))
        elif key in {"CONTENT_TYPE", "CONTENT_LENGTH"} and value != "":
            headers.append((key.replace("_", "-").encode("ascii"), str(value).encode("latin-1")))
    # Bounded reads also handle chunked requests with no original Content-Length.
    body = request.read(settings.GATEWAY_MAX_BODY + 1)
    if len(body) > settings.GATEWAY_MAX_BODY:
        raise RequestDataTooBig("Request body limit exceeded.")

    def address(host, port, default):
        try:
            number = int(meta.get(port, default))
        except (TypeError, ValueError):
            number = default
        try:
            ip = str(ip_address(meta.get(host) or "127.0.0.1"))
        except ValueError:
            ip = "127.0.0.1"
        return ip, max(0, min(number, 65535))

    return Request(
        method=request.method.encode("ascii"),
        target=target.encode("ascii"),
        version=meta.get("SERVER_PROTOCOL", "HTTP/1.1"),
        headers=tuple(headers),
        body=body,
        peer=address("REMOTE_ADDR", "REMOTE_PORT", 0),
        local=address("SERVER_NAME", "SERVER_PORT", 80),
    )


def end_to_end(headers):
    nominated = {
        token.strip().lower()
        for name, value in headers
        if name.lower() == b"connection"
        for token in value.split(b",")
    }
    return [(name, value) for name, value in headers if name.lower() not in HOP_HEADERS | nominated]


@dataclass(frozen=True)
class Request:
    method: bytes
    target: bytes
    version: str
    headers: tuple[tuple[bytes, bytes], ...]
    body: bytes
    peer: tuple[str, int]
    local: tuple[str, int]

    @property
    def display_path(self):
        if not self.target.startswith(b"/"):
            return "(unavailable)"
        return self.target.partition(b"?")[0][:320].decode("ascii", "replace")

    def model_fields(self, training_headers):
        values = {
            "method": self.method.decode("ascii"),
            "request_target": self.target.decode("ascii"),
            "protocol": self.version,
            "body": self.body.decode("utf-8", "replace"),
        }
        for field, header in training_headers.items():
            key = header.lower().encode("ascii")
            separator = b"; " if field == "cookie" else b", "
            values[field] = separator.join(v for k, v in self.headers if k.lower() == key).decode(
                "latin-1"
            )
        return values

    def rejection(self):
        names = [name.lower() for name, _ in self.headers]
        if (
            not METHOD.fullmatch(self.method)
            or not TARGET.fullmatch(self.target)
            or not self.target.startswith(b"/")
            or b"#" in self.target
            or self.version not in {"HTTP/1.0", "HTTP/1.1"}
        ):
            return 400, "invalid_request_target_or_line"
        if self.method.upper() == b"CONNECT" or b"upgrade" in names:
            return 400, "unsupported_tunnel_or_upgrade"
        if names.count(b"host") > 1 or names.count(b"content-length") > 1:
            return 400, "ambiguous_host_or_framing"
        if any(
            not HEADER_NAME.fullmatch(k) or any(c < 32 and c != 9 or c == 127 for c in v)
            for k, v in self.headers
        ):
            return 400, "invalid_header"
        for name, value in self.headers:
            if name.lower() == b"host" and (not value or b"," in value or b" " in value):
                return 400, "ambiguous_host_or_framing"
            if name.lower() == b"content-length" and (
                len(value) > 20 or not value.isdigit() or int(value) != len(self.body)
            ):
                return 400, "invalid_content_length"
        # Waitress decodes transfer framing before constructing the WSGI request.
        if b"transfer-encoding" in names:
            return 400, "unsupported_transfer_framing"
        if any(
            k.lower() == b"content-encoding" and v.strip().lower() != b"identity"
            for k, v in self.headers
        ):
            return 415, "unsupported_content_encoding"
        return None

    def upstream_headers(self, authority):
        result = [
            (name, value)
            for name, value in end_to_end(self.headers)
            if name.lower() not in REQUEST_ONLY and not name.lower().startswith(b"x-forwarded-")
        ]
        if not any(name.lower() == b"host" for name, _ in result):
            result.append((b"Host", authority))
        result.append((b"Content-Length", str(len(self.body)).encode("ascii")))
        return result
