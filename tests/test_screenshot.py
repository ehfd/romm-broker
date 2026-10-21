"""Capturing the streamed desktop for a state's thumbnail."""

import io
from collections.abc import Iterator
from typing import Callable

import httpx
import pytest
from PIL import Image

from webstation_broker import screenshot, settings

Getter = Callable[..., httpx.Response]
"""The shape of the `httpx.get` a test stands in for the real one."""


@pytest.fixture(autouse=True)
def no_frame_capture(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Override the shared stub: these tests drive the real capture against fake endpoints.

    Pins the candidate endpoints and the master token, and clears the
    last-known-good endpoint on both sides, since it is module-global.
    """
    monkeypatch.setattr(settings, "SELKIES_SCREENSHOT_URLS", ["http://a/api/screenshot", "http://b/api/screenshot"])
    monkeypatch.setattr(settings, "SELKIES_MASTER_TOKEN", "master")
    monkeypatch.setattr(screenshot, "_active_url", None)
    yield
    screenshot._active_url = None


def _png(width: int, height: int) -> bytes:
    """Encode a solid image of the given size as PNG.

    Args:
        width: Pixel width.
        height: Pixel height.

    Returns:
        The PNG bytes.
    """
    out = io.BytesIO()
    Image.new("RGBA", (width, height), (10, 200, 30, 255)).save(out, format="PNG")
    return out.getvalue()


def _size(png: bytes) -> tuple[int, int]:
    """Read the pixel size back out of a PNG.

    Args:
        png: The image to measure.

    Returns:
        `(width, height)`.
    """
    with Image.open(io.BytesIO(png)) as image:
        return image.size


def _answer(
    monkeypatch: pytest.MonkeyPatch, answers: dict[str, tuple[int, bytes]]
) -> list[tuple[str, dict[str, str]]]:
    """Make each endpoint answer with a status and body, recording every request.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        answers: Endpoint URL to the `(status, body)` it answers with; an
            endpoint not listed refuses the connection.

    Returns:
        The list every request's `(url, headers)` is appended to.
    """
    requests: list[tuple[str, dict[str, str]]] = []

    def get(url: str, headers: dict[str, str], timeout: float) -> httpx.Response:
        requests.append((url, headers))
        if url not in answers:
            raise httpx.ConnectError("connection refused")
        status, body = answers[url]
        return httpx.Response(status, content=body, request=httpx.Request("GET", url))

    monkeypatch.setattr(screenshot.httpx, "get", get)
    return requests


def test_the_frame_is_scaled_to_the_thumbnail_size(monkeypatch: pytest.MonkeyPatch) -> None:
    """A full-size capture comes back bounded on its longest side, aspect kept."""
    monkeypatch.setattr(settings, "STATE_SCREENSHOT_SIZE", 320)
    requests = _answer(monkeypatch, {"http://a/api/screenshot": (200, _png(1024, 768))})

    frame = screenshot.capture_frame()

    assert requests == [("http://a/api/screenshot", {"Authorization": "Bearer master"})]
    assert frame is not None
    assert frame.startswith(b"\x89PNG")
    assert _size(frame) == (320, 240)


def test_a_frame_already_small_enough_is_not_upscaled(monkeypatch: pytest.MonkeyPatch) -> None:
    """The thumbnail size is a ceiling, not a target."""
    monkeypatch.setattr(settings, "STATE_SCREENSHOT_SIZE", 640)
    _answer(monkeypatch, {"http://a/api/screenshot": (200, _png(200, 100))})

    frame = screenshot.capture_frame()

    assert frame is not None
    assert _size(frame) == (200, 100)


def test_an_unreachable_endpoint_falls_through_to_the_next(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each known image's endpoint is tried in turn, as the token pushes do."""
    requests = _answer(monkeypatch, {"http://b/api/screenshot": (200, _png(64, 64))})

    assert screenshot.capture_frame() is not None
    assert [url for url, _ in requests] == ["http://a/api/screenshot", "http://b/api/screenshot"]


def test_the_endpoint_that_answered_is_tried_first_next_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """A steady-state capture does not re-probe the other image's endpoint."""
    requests = _answer(monkeypatch, {"http://b/api/screenshot": (200, _png(64, 64))})
    screenshot.capture_frame()
    requests.clear()

    assert screenshot.capture_frame() is not None
    assert [url for url, _ in requests] == ["http://b/api/screenshot"]


def test_no_reachable_endpoint_yields_no_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    """A container without the API saves fine, just without a picture."""
    _answer(monkeypatch, {})

    assert screenshot.capture_frame() is None


@pytest.mark.parametrize("status", [401, 404, 501])
def test_an_error_status_yields_no_frame(monkeypatch: pytest.MonkeyPatch, status: int) -> None:
    """A refused token, an unknown display, or a build without screenshots reads as an absent endpoint."""
    _answer(
        monkeypatch, {"http://a/api/screenshot": (status, b"no"), "http://b/api/screenshot": (status, b"no")}
    )

    assert screenshot.capture_frame() is None


@pytest.mark.parametrize("body", [b"", b"not a png", b"<html>"])
def test_an_answer_that_is_not_an_image_yields_no_frame(monkeypatch: pytest.MonkeyPatch, body: bytes) -> None:
    """Anything but a decodable image is dropped rather than served."""
    _answer(monkeypatch, {"http://a/api/screenshot": (200, body), "http://b/api/screenshot": (200, body)})

    assert screenshot.capture_frame() is None

