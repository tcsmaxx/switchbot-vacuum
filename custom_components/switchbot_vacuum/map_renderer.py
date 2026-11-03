"""Render SwitchBot K10+/K10+ Pro cloud map data as PNG."""
from __future__ import annotations

import base64
import io
import json
from typing import Any

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

SCALE = 4
ORIENTATION_OPS = ("flip_tb", "rot180")
ROOM_COLORS = [
    (125, 195, 245, 200),
    (160, 175, 245, 200),
    (255, 215, 110, 200),
    (170, 230, 215, 200),
    (240, 165, 140, 200),
]


def _lz4_block_decompress(data: bytes, max_output_size: int) -> bytes:
    """Decompress the raw LZ4 block used by the vacuum map."""
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


def _normalize_payload(data: dict[str, Any]) -> dict[str, Any]:
    if "data" in data and isinstance(data["data"], dict):
        return data["data"]
    return data


def _decode_name(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    try:
        return base64.b64decode(value).decode("utf-8", errors="ignore").strip() or value
    except Exception:
        return value


def _smart_areas(data: dict[str, Any]) -> list[dict[str, Any]]:
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
    for name in ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf", "tahoma.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _centroid(points: list[tuple[int, int]]) -> tuple[int, int]:
    """Polygon centroid matching the standalone renderer without numpy."""
    if not points:
        return 0, 0
    if len(points) < 3:
        return points[0]

    twice_area = 0.0
    cx = 0.0
    cy = 0.0
    for idx, (x1, y1) in enumerate(points):
        x2, y2 = points[(idx + 1) % len(points)]
        cross = x1 * y2 - x2 * y1
        twice_area += cross
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross

    if abs(twice_area) < 1e-6:
        return (
            sum(p[0] for p in points) // len(points),
            sum(p[1] for p in points) // len(points),
        )

    return int(cx / (3.0 * twice_area)), int(cy / (3.0 * twice_area))


def _apply_orientation(img: Image.Image) -> Image.Image:
    for op in ORIENTATION_OPS:
        if op == "flip_tb":
            img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        elif op == "flip_lr":
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        elif op == "rot180":
            img = img.transpose(Image.Transpose.ROTATE_180)
    return img


def _transform_point(x: int, y: int, width: int, height: int) -> tuple[int, int]:
    for op in ORIENTATION_OPS:
        if op == "flip_tb":
            y = (height - 1) - y
        elif op == "flip_lr":
            x = (width - 1) - x
        elif op == "rot180":
            x, y = (width - 1) - x, (height - 1) - y
    return x, y


def _draw_path_soft(
    base: Image.Image,
    points: list[tuple[int, int]],
    width_px: int = 2,
    color: tuple[int, int, int, int] = (255, 255, 255, 210),
    aa: int = 3,
    blur: float = 0.8,
) -> Image.Image:
    if len(points) < 2:
        return base
    width, height = base.size
    hi = Image.new("RGBA", (width * aa, height * aa), (0, 0, 0, 0))
    ImageDraw.Draw(hi).line(
        [(int(x * aa), int(y * aa)) for x, y in points],
        fill=color,
        width=max(1, int(width_px * aa)),
    )
    if blur > 0:
        hi = hi.filter(ImageFilter.GaussianBlur(radius=blur * aa / 3.0))
    return Image.alpha_composite(base, hi.resize((width, height), Image.Resampling.LANCZOS))


def _draw_pill(draw: ImageDraw.ImageDraw, cx: int, cy: int, text: str, scale: int) -> None:
    font = _font(int(5.5 * scale))
    pad_x = int(4.0 * scale)
    pad_y = int(2.5 * scale)
    radius = int(4.0 * scale)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    pill_w = tw + 2 * pad_x
    pill_h = th + 2 * pad_y
    x0 = cx - pill_w // 2
    y0 = cy - pill_h // 2
    draw.rounded_rectangle(
        (x0, y0, x0 + pill_w, y0 + pill_h),
        radius=radius,
        fill=(45, 45, 45, 140),
    )
    draw.text((cx, cy), text, font=font, fill=(255, 255, 255, 245), anchor="mm")


def render_map_png(data: dict[str, Any], scale: int = SCALE) -> bytes:
    """Render the cloud map using the same look as the original standalone renderer."""
    if not isinstance(data, dict):
        raise ValueError("Map payload is not an object")
    data = _normalize_payload(data)

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
        px = int(((float(x_mm) / 1000.0) - x_min) / resolution)
        py = int(((float(y_mm) / 1000.0) - y_min) / resolution)
        return max(0, min(width - 1, px)), max(0, min(height - 1, py))

    # Original renderer background: white with a very light unknown-space overlay.
    base = Image.new("RGBA", (width, height), (252, 252, 252, 255))
    unknown = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    unknown_px = unknown.load()
    floor_mask = Image.new("L", (width, height), 0)
    floor_px = floor_mask.load()

    for y in range(height):
        row = y * width
        for x in range(width):
            value = decoded[row + x]
            if value == 127:
                unknown_px[x, y] = (245, 246, 248, 255)
            elif value == 255:
                floor_px[x, y] = 255
    base = Image.alpha_composite(base, unknown)

    # Rooms: restrict the polygon overlay to real floor pixels exactly like the old script.
    rooms = _smart_areas(data)
    room_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    room_centres: list[tuple[int, int, str]] = []

    for index, room in enumerate(rooms):
        if not isinstance(room, dict):
            continue
        vertices = room.get("vertexs", [])
        points = [
            to_px(v[0], v[1])
            for v in vertices
            if isinstance(v, (list, tuple)) and len(v) >= 2
        ]
        if len(points) < 3:
            continue

        polygon_mask = Image.new("L", (width, height), 0)
        ImageDraw.Draw(polygon_mask).polygon(points, fill=255)
        clipped_mask = ImageChops.multiply(polygon_mask, floor_mask)
        room_color = Image.new("RGBA", (width, height), ROOM_COLORS[index % len(ROOM_COLORS)])
        room_layer = Image.composite(room_color, room_layer, clipped_mask)
        cx, cy = _centroid(points)
        room_centres.append((cx, cy, _decode_name(room.get("name", ""))))

    base = Image.alpha_composite(base, room_layer)

    # Thin wall outline: draw only boundary pixels of wall regions, not the whole wall mask.
    wall_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    wall_px = wall_layer.load()
    for y in range(height):
        row = y * width
        for x in range(width):
            if decoded[row + x] != 0:
                continue
            interior = True
            for nx, ny in ((x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)):
                if nx < 0 or ny < 0 or nx >= width or ny >= height:
                    interior = False
                    break
                if decoded[ny * width + nx] != 0:
                    interior = False
                    break
            if not interior:
                wall_px[x, y] = (40, 45, 60, 255)
    base = Image.alpha_composite(base, wall_layer)

    # Forbidden areas / virtual walls.
    forbid = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    forbid_draw = ImageDraw.Draw(forbid)
    areas = data.get("area", [])
    if isinstance(areas, list):
        for area in areas:
            if not isinstance(area, dict):
                continue
            vertices = area.get("vertexs", [])
            points = [
                to_px(v[0], v[1])
                for v in vertices
                if isinstance(v, (list, tuple)) and len(v) >= 2
            ]
            if len(points) >= 3 and area.get("active") == "forbid":
                forbid_draw.polygon(points, fill=(235, 80, 80, 60), outline=(170, 40, 40, 200))
            elif len(points) >= 2:
                forbid_draw.line(points, fill=(170, 40, 40, 200), width=1)
    base = Image.alpha_composite(base, forbid)

    # Same orientation and pixel-perfect nearest-neighbour scaling as the old renderer.
    base = _apply_orientation(base)
    map_scaled = base.resize((width * scale, height * scale), Image.Resampling.NEAREST)

    # Soft antialiased cleaning route.
    route = data.get("posArray", [])
    route_points: list[tuple[int, int]] = []
    if isinstance(route, list):
        for point in route:
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                tx, ty = _transform_point(*to_px(point[0], point[1]), width, height)
                route_points.append((tx * scale, ty * scale))
    map_scaled = _draw_path_soft(map_scaled, route_points)

    # Labels + charging station on the final top layer.
    top = Image.new("RGBA", map_scaled.size, (0, 0, 0, 0))
    top_draw = ImageDraw.Draw(top)

    for cx, cy, name in room_centres:
        if not name:
            continue
        tx, ty = _transform_point(cx, cy, width, height)
        _draw_pill(top_draw, tx * scale, ty * scale, name, scale)

    charger = data.get("chargeHandlePos")
    if isinstance(charger, (list, tuple)) and len(charger) >= 2:
        tx, ty = _transform_point(*to_px(charger[0], charger[1]), width, height)
        radius = 6 * scale
        cx = tx * scale
        cy = ty * scale
        top_draw.ellipse(
            (cx - radius, cy - radius, cx + radius, cy + radius),
            fill=(160, 32, 240, 200),
            outline=(255, 255, 255, 255),
            width=max(1, int(1.5 * scale)),
        )

    image = Image.alpha_composite(map_scaled, top)
    output = io.BytesIO()
    image.convert("RGB").save(output, format="PNG", optimize=True)
    return output.getvalue()
