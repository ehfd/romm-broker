"""Save-state frame capture over the selkies screenshot API.

Selkies answers `GET /api/screenshot` with a PNG of the primary display, the
cursor drawn in, taken by pixelflux from the live capture. The route is on
the same server as the token endpoint, so it is reached through the same
candidate bases (`SELKIES_SCREENSHOT_URLS`) with the same master token, and
the base that last answered is tried first. The frame is scaled down here to
the thumbnail RomM files beside a state.
"""

import io
import logging
from typing import Optional

import httpx
from PIL import Image, UnidentifiedImageError

from . import settings

log = logging.getLogger(__name__)

SCREENSHOT_TIMEOUT = 5.0
"""Seconds a screenshot request gets; on Wayland it forces a one-frame GPU readback."""

_active_url: Optional[str] = None
"""Endpoint that last answered with a frame, tried first on the next capture."""


def _screenshot_urls() -> list[str]:
    """Return the candidate screenshot endpoints, last-known-good first.

    Returns:
        The configured endpoints, with `_active_url` moved to the front when it
        is one of them.
    """
    urls = settings.SELKIES_SCREENSHOT_URLS
    if _active_url in urls:
        return [_active_url] + [u for u in urls if u != _active_url]
    return list(urls)


def capture_frame() -> Optional[bytes]:
    """Grab the streamed desktop as it is right now, scaled for a state thumbnail.

    Never raises: the frame is taken on the way into a save, and a missing
    preview must not read as a failed save.

    Returns:
        PNG bytes bounded to `STATE_SCREENSHOT_SIZE` on the longest side, or
        None when no endpoint is reachable or every one answers with anything
        that does not decode as an image.
    """
    global _active_url
    headers = {"Authorization": f"Bearer {settings.SELKIES_MASTER_TOKEN}"}
    for url in _screenshot_urls():
        try:
            response = httpx.get(url, headers=headers, timeout=SCREENSHOT_TIMEOUT)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            log.warning("state frame capture failed at %s: %s", url, exc)
            continue
        frame = _thumbnail(response.content)
        if frame is not None:
            _active_url = url
            return frame
    return None


def _thumbnail(raw: bytes) -> Optional[bytes]:
    """Scale a captured frame to the thumbnail size and re-encode it as PNG.

    Args:
        raw: The full-size frame as the endpoint returned it.

    Returns:
        PNG bytes bounded to `STATE_SCREENSHOT_SIZE` on the longest side, or
        None when `raw` is not a decodable image.
    """
    size = settings.STATE_SCREENSHOT_SIZE
    try:
        with Image.open(io.BytesIO(raw)) as image:
            frame = image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        log.warning("state frame is not a decodable image: %s", exc)
        return None
    frame.thumbnail((size, size))
    out = io.BytesIO()
    frame.save(out, format="PNG", optimize=True)
    log.debug("state frame captured: %dx%d, %d bytes", frame.width, frame.height, out.tell())
    return out.getvalue()
