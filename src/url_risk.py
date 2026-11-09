import ipaddress
from dataclasses import dataclass
from urllib.parse import urlsplit


WEB_SCHEMES = {"http", "https"}


@dataclass(frozen=True)
class UrlAssessment:
    is_web_url: bool
    host: str | None
    status: str
    warnings: tuple[str, ...]


def _looks_like_web_address(payload: str) -> bool:
    if "://" in payload or urlsplit(payload).scheme or payload.lower().startswith("www."):
        return True
    return not any(character.isspace() for character in payload) and "." in payload


def analyze_url(payload: str) -> UrlAssessment:
    """Report simple URL warning signals without making any network requests."""
    candidate = payload.strip()
    if not candidate or not _looks_like_web_address(candidate):
        return UrlAssessment(
            is_web_url=False,
            host=None,
            status="Not a web URL",
            warnings=(),
        )

    try:
        parsed = urlsplit(candidate)
        if not parsed.scheme:
            candidate = f"https://{candidate}"
            parsed = urlsplit(candidate)
        hostname = parsed.hostname
    except ValueError:
        return UrlAssessment(
            is_web_url=True,
            host=None,
            status="Review recommended",
            warnings=("URL could not be parsed reliably.",),
        )

    scheme = parsed.scheme.lower()
    if scheme not in WEB_SCHEMES:
        return UrlAssessment(
            is_web_url=False,
            host=hostname,
            status="Non-web URI; do not open automatically",
            warnings=(f"URI scheme is {scheme or 'missing'}, not HTTP or HTTPS.",),
        )
    if not hostname:
        return UrlAssessment(
            is_web_url=True,
            host=None,
            status="Review recommended",
            warnings=("HTTP(S) URL has no hostname.",),
        )

    warnings: list[str] = []
    if parsed.username is not None or parsed.password is not None:
        warnings.append("URL contains embedded username or password information.")
    if scheme == "http":
        warnings.append("Connection uses HTTP rather than encrypted HTTPS.")

    normalized_host = hostname.rstrip(".").lower()
    try:
        ip_address = ipaddress.ip_address(normalized_host)
    except ValueError:
        ip_address = None

    if ip_address is not None:
        warnings.append("Destination uses an IP address instead of a domain name.")
        if not ip_address.is_global:
            warnings.append("IP address is non-public or reserved.")

    try:
        ascii_host = normalized_host.encode("idna").decode("ascii").lower()
    except UnicodeError:
        ascii_host = normalized_host
        warnings.append("Hostname contains characters that could not be normalized.")
    if "xn--" in ascii_host:
        warnings.append("Hostname uses internationalized (punycode) domain labels.")

    try:
        port = parsed.port
    except ValueError:
        port = None
        warnings.append("URL contains an invalid port.")
    if port is not None and port not in {80, 443}:
        warnings.append(f"URL uses a non-standard web port ({port}).")

    status = "Review recommended" if warnings else "No heuristic warnings; not verified safe"
    return UrlAssessment(
        is_web_url=True,
        host=hostname,
        status=status,
        warnings=tuple(warnings),
    )
