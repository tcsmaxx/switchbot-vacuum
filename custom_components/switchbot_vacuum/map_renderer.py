"""Render SwitchBot K10+ cloud map data as PNG."""
from __future__ import annotations

import base64
import io
import json
from typing import Any

from PIL import Image, ImageDraw, ImageFont

SCALE = 4
ROOM_COLORS = [
    (125, 195, 245, 210),
    (160, 175, 245, 210),
    (255, 215, 110, 210),
    (170, 230, 215, 210),
    (240, 165, 140, 210),
]


def _lz4_block_decompress(data: bytes, max_output_size: int) -> bytes:
    """Decompress a raw LZ4 block without requiring the external lz4 package.

    The K10+ map payload uses the standard LZ4 block format.  This compact
    decoder intentionally implements only block decoding; frames are not used
    by the robot payload.
    """
    src = memoryview(data)
    pos = 0
    out = bytearray()
    size = len(src)

    while pos < size:
        token = src[pos]
        pos += 1

        literal_len = token >> 4
        if literal_len == 15:
            while True:
                if pos >= size:
                    raise ValueError("Invalid LZ4 block: truncated literal length")
                extra = src[pos]
                pos += 1
                literal_len += extra
                if extra != 255:
                    break

        if pos + literal_len > size:
            raise ValueError("Invalid LZ4 block: truncated literals")
        if len(out) + literal_len > max_output_size:
            raise ValueError("LZ4 map exceeds expected maximum size")
        out.extend(src[pos : pos + literal_len])
        pos += literal_len

        # A block may legally end after the final literal run.
        if pos >= size:
            break

        if pos + 2 > size:
            raise ValueError("Invalid LZ4 block: missing match offset")
        offset = int(src[pos]) | (int(src[pos + 1]) << 8)
        pos += 2
        if offset <= 0 or offset > len(out):
            raise ValueError("Invalid LZ4 block: bad match offset")

        match_len = token & 0x0F
        if match_len == 15:
            while True:
                if pos >= size:
                    raise ValueError("Invalid LZ4 block: truncated match length")
                extra = src[pos]
                pos += 1
                match_len += extra
                if extra != 255:
                    break
        match_len += 4

        if len(out) + match_len > max_output_size:
            raise ValueError("LZ4 map exceeds expected maximum size")

        match_start = len(out) - offset
        for _ in range(match_len):
            out.append(out[match_start])
            match_start += 1

    return bytes(out)


def _decode_name(value: Any) -> str:
    """Decode Base64 room names used by the K10 map payload."""
    if not isinstance(value, str):
        return ""
    try:
        return base64.b64decode(value).decode("utf-8", errors="ignore").strip() or value
    except Exception:
        return value


def _smart_areas(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Return smartArea entries from dict or JSON-string payloads."""
    smart = data.get("smartArea", {})
    if isinstance(smart, str):
        try:
            smart = json.loads(smart)
        except (TypeError, ValueError, json.JSONDecodeError):
            return []
    if isinstance(smart, dict):
        value = smart.get("value", [])
        return value if isinstance(value, list) else []
    return smart if isinstance(smart, list) else []


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "Arial.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_map_png(data: dict[str, Any], scale: int = SCALE) -> bytes:
    """Render a K10+ map payload to PNG bytes."""
    if not isinstance(data, dict):
        raise ValueError("Map payload is not an object")

    map_str = data.get("map")
    if not isinstance(map_str, str) or not map_str:
        raise ValueError("Map payload contains no raster data")

    width = int(data.get("width") or 0)
    if width <= 0:
        raise ValueError("Map width is invalid")

    resolution = float(data.get("resolution", 0.05))
    x_min = float(data.get("x_min", 0.0))
    y_min = float(data.get("y_min", 0.0))

    raw = base64.b64decode(map_str)
    declared_height = int(data.get("height") or 0)
    max_output = width * declared_height if declared_height > 0 else width * 1500
    decoded = _lz4_block_decompress(raw, max_output)
    height = len(decoded) // width
    if height <= 0:
        raise ValueError("Decoded map height is invalid")
    decoded = decoded[: width * height]

    def to_px(x_mm: float, y_mm: float) -> tuple[int, int]:
        x = int(((float(x_mm) / 1000.0) - x_min) / resolution)
        y = int(((float(y_mm) / 1000.0) - y_min) / resolution)
        return max(0, min(width - 1, x)), max(0, min(height - 1, y))

    # Pixel classes used by the current K10 payload: 0 wall, 127 unknown, 255 floor.
    img = Image.new("RGBA", (width, height), (252, 252, 252, 255))
    pixels = img.load()
    for y in range(height):
        row = y * width
        for x in range(width):
            value = decoded[row + x]
            if value == 0:
                pixels[x, y] = (40, 45, 60, 255)
            elif value == 127:
                pixels[x, y] = (245, 246, 248, 255)

    rooms = _smart_areas(data)
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    room_centres: list[tuple[int, int, str]] = []

    for index, room in enumerate(rooms):
        if not isinstance(room, dict):
            continue
        vertices = room.get("vertexs", [])
        points = []
        if isinstance(vertices, list):
            for vertex in vertices:
                if isinstance(vertex, (list, tuple)) and len(vertex) >= 2:
                    points.append(to_px(vertex[0], vertex[1]))
        if len(points) < 3:
            continue
        draw.polygon(points, fill=ROOM_COLORS[index % len(ROOM_COLORS)])
        cx = sum(p[0] for p in points) // len(points)
        cy = sum(p[1] for p in points) // len(points)
        room_centres.append((cx, cy, _decode_name(room.get("name", ""))))

    img = Image.alpha_composite(img, overlay)

    # Forbidden areas / virtual walls.
    areas = data.get("area", [])
    if isinstance(areas, list):
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        area_draw = ImageDraw.Draw(layer)
        for area in areas:
            if not isinstance(area, dict):
                continue
            vertices = area.get("vertexs", [])
            points = [to_px(v[0], v[1]) for v in vertices if isinstance(v, (list, tuple)) and len(v) >= 2]
            if len(points) >= 3 and area.get("active") == "forbid":
                area_draw.polygon(points, fill=(235, 80, 80, 55), outline=(170, 40, 40, 220))
            elif len(points) >= 2:
                area_draw.line(points, fill=(170, 40, 40, 220), width=1)
        img = Image.alpha_composite(img, layer)

    # Match the orientation used by the existing standalone renderer.
    img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM).transpose(Image.Transpose.ROTATE_180)
    img = img.resize((width * scale, height * scale), Image.Resampling.NEAREST)

    def transform(x: int, y: int) -> tuple[int, int]:
        # flip_tb followed by rot180 is equivalent to a horizontal flip.
        return (width - 1 - x) * scale, y * scale

    top = Image.new("RGBA", img.size, (0, 0, 0, 0))
    top_draw = ImageDraw.Draw(top)

    # Cleaning route.
    route = data.get("posArray", [])
    if isinstance(route, list):
        route_points = []
        for point in route:
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                route_points.append(transform(*to_px(point[0], point[1])))
        if len(route_points) >= 2:
            top_draw.line(route_points, fill=(255, 255, 255, 220), width=max(2, scale))

    # Room labels.
    font = _font(max(12, int(5.5 * scale)))
    for cx, cy, name in room_centres:
        if not name:
            continue
        x, y = transform(cx, cy)
        bbox = top_draw.textbbox((0, 0), name, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        px, py = 4 * scale, 2 * scale
        top_draw.rounded_rectangle(
            (x - tw // 2 - px, y - th // 2 - py, x + tw // 2 + px, y + th // 2 + py),
            radius=4 * scale,
            fill=(45, 45, 45, 150),
        )
        top_draw.text((x, y), name, font=font, fill=(255, 255, 255, 245), anchor="mm")

    # Charging station.
    charger = data.get("chargeHandlePos")
    if isinstance(charger, (list, tuple)) and len(charger) >= 2:
        x, y = transform(*to_px(charger[0], charger[1]))
        radius = 6 * scale
        top_draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=(160, 32, 240, 220),
            outline=(255, 255, 255, 255),
            width=max(1, scale),
        )

    img = Image.alpha_composite(img, top)
    output = io.BytesIO()
    img.convert("RGB").save(output, format="PNG", optimize=True)
    return output.getvalue()
