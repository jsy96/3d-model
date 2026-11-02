#!/usr/bin/env python3
"""Serve a local, manual checkerboard-box annotation page."""

from __future__ import annotations

import json
import mimetypes
import io
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse


ROOT = Path(__file__).resolve().parent
PROJECT_DIR = ROOT / "生产项目"
DATA_DIR = PROJECT_DIR / "data"
IMAGE_DIR = DATA_DIR / "images"
ANNOTATIONS_DIR = DATA_DIR / "annotations"
BOXES_PATH = ANNOTATIONS_DIR / "board_boxes.json"
ARTIFACTS_DIR = DATA_DIR / "calibration_artifacts"
HTML_PATH = ROOT / "calibration_labeler.html"
HOST = "127.0.0.1"
PORT = 8765
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
SAVE_LOCK = threading.Lock()
BOARD_INNER_CORNER_SIZES = ((8, 11), (11, 8))


def list_images() -> list[Path]:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(
        (
            path for path in IMAGE_DIR.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ),
        key=lambda path: path.name.casefold(),
    )


def image_names() -> set[str]:
    return {path.name for path in list_images()}


def read_annotations() -> dict:
    if not BOXES_PATH.is_file():
        return {"version": 1, "coordinate_system": "normalized_image", "image_sizes": {}, "boxes": {}}
    try:
        value = json.loads(BOXES_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"version": 1, "coordinate_system": "normalized_image", "image_sizes": {}, "boxes": {}}
    if not isinstance(value, dict) or not isinstance(value.get("boxes"), dict):
        return {"version": 1, "coordinate_system": "normalized_image", "image_sizes": {}, "boxes": {}}
    value.setdefault("version", 1)
    value.setdefault("coordinate_system", "normalized_image")
    value.setdefault("image_sizes", {})
    return value


def save_annotations(value: dict) -> None:
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    temporary = BOXES_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(BOXES_PATH)


def resolve_image(name: str) -> Path:
    if not name or Path(name).name != name or "/" in name or "\\" in name:
        raise ValueError("图片文件名无效。")
    path = IMAGE_DIR / name
    if not path.is_file() or path.resolve().parent != IMAGE_DIR.resolve():
        raise ValueError("找不到这张图片。")
    return path


def save_uploaded_image(name: str, body: bytes) -> str:
    if not name or Path(name).name != name or "/" in name or "\\" in name:
        raise ValueError("上传文件名无效。")
    suffix = Path(name).suffix.lower()
    if suffix not in IMAGE_SUFFIXES:
        raise ValueError("只支持 JPG、PNG、BMP、TIF 或 WEBP 图片。")
    if not body or len(body) > MAX_UPLOAD_BYTES:
        raise ValueError("图片为空或超过 100 MB。")
    try:
        from PIL import Image

        with Image.open(io.BytesIO(body)) as image:
            image.verify()
    except Exception as exc:
        raise ValueError("上传内容不是可读取的图片。") from exc

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(name).stem
    with SAVE_LOCK:
        candidate = name
        version = 2
        while (IMAGE_DIR / candidate).exists():
            candidate = f"{stem} ({version}){suffix}"
            version += 1
        temporary = IMAGE_DIR / f".{candidate}.uploading"
        temporary.write_bytes(body)
        temporary.replace(IMAGE_DIR / candidate)
    return candidate


def validate_boxes(boxes: object) -> list[dict]:
    if not isinstance(boxes, list) or len(boxes) > 1000:
        raise ValueError("框选数据格式不正确。")
    validated = []
    for box in boxes:
        if not isinstance(box, dict):
            raise ValueError("框选数据格式不正确。")
        if isinstance(box.get("points"), list):
            if len(box["points"]) != 4:
                raise ValueError("四边形必须包含四个顶点。")
            points = []
            for point in box["points"]:
                if not isinstance(point, dict):
                    raise ValueError("四边形顶点格式不正确。")
                try:
                    x, y = float(point["x"]), float(point["y"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError("四边形顶点坐标格式不正确。") from exc
                if not 0 <= x <= 1 or not 0 <= y <= 1:
                    raise ValueError("四边形顶点必须在图片范围内。")
                points.append({"x": x, "y": y})
            item = {"points": points}
            seed_points = box.get("seed_points")
            if seed_points is not None:
                if not isinstance(seed_points, list) or len(seed_points) != 4:
                    raise ValueError("原始框选区域必须包含四个顶点。")
                validated_seed = []
                for point in seed_points:
                    if not isinstance(point, dict):
                        raise ValueError("原始框选顶点格式不正确。")
                    try:
                        x, y = float(point["x"]), float(point["y"])
                    except (KeyError, TypeError, ValueError) as exc:
                        raise ValueError("原始框选顶点坐标格式不正确。") from exc
                    if not 0 <= x <= 1 or not 0 <= y <= 1:
                        raise ValueError("原始框选顶点必须在图片范围内。")
                    validated_seed.append({"x": x, "y": y})
                item["seed_points"] = validated_seed
            detected_squares = box.get("detected_squares")
            if (
                isinstance(detected_squares, list)
                and len(detected_squares) == 2
                and all(isinstance(value, int) and 2 <= value <= 100 for value in detected_squares)
            ):
                item["detected_squares"] = detected_squares
            validated.append(item)
            continue
        try:
            x, y, width, height = (float(box[key]) for key in ("x", "y", "w", "h"))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("框选坐标格式不正确。") from exc
        values = (x, y, width, height)
        if any(not 0 <= item <= 1 for item in values):
            raise ValueError("框选坐标必须在图片范围内。")
        if width <= 0 or height <= 0 or x + width > 1.000001 or y + height > 1.000001:
            raise ValueError("框选范围必须位于图片内。")
        validated.append({"x": x, "y": y, "w": width, "h": height})
    return validated


def refine_checkerboard(image_path: Path, rough_points: object) -> dict:
    """Find and validate one complete 9x12-square board around a seed region."""
    try:
        import cv2
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"缺少图像处理依赖：{exc.name}") from exc

    if not isinstance(rough_points, list) or len(rough_points) != 4:
        raise ValueError("请先画一个矩形框。")
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("图片无法读取。")
    image_height, image_width = image.shape[:2]
    points = []
    for point in rough_points:
        if not isinstance(point, dict):
            raise ValueError("矩形框顶点格式不正确。")
        try:
            x, y = float(point["x"]), float(point["y"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("矩形框顶点坐标格式不正确。") from exc
        if not 0 <= x <= 1 or not 0 <= y <= 1:
            raise ValueError("矩形框顶点必须在图片范围内。")
        points.append((x * image_width, y * image_height))

    seed = np.asarray(points, dtype=np.float32)
    seed_min = np.min(seed, axis=0)
    seed_max = np.max(seed, axis=0)
    seed_center = np.mean(seed, axis=0)
    seed_width, seed_height = np.maximum(seed_max - seed_min, 16)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY

    def validate_candidate(corners: object, pattern_size: tuple[int, int]) -> dict | None:
        inner_columns, inner_rows = pattern_size
        corner_points = np.asarray(corners, dtype=np.float32).reshape((-1, 2))
        if corner_points.shape[0] != inner_columns * inner_rows:
            return None
        grid = np.asarray(
            [(column, row) for row in range(inner_rows) for column in range(inner_columns)],
            dtype=np.float32,
        )
        homography, _ = cv2.findHomography(grid, corner_points, method=0)
        if homography is None or not np.isfinite(homography).all():
            return None

        projected = cv2.perspectiveTransform(grid.reshape((1, -1, 2)), homography).reshape((-1, 2))
        residuals = np.linalg.norm(projected - corner_points, axis=1)
        neighbor_steps = []
        lattice = corner_points.reshape((inner_rows, inner_columns, 2))
        if inner_columns > 1:
            neighbor_steps.extend(np.linalg.norm(np.diff(lattice, axis=1), axis=2).reshape(-1).tolist())
        if inner_rows > 1:
            neighbor_steps.extend(np.linalg.norm(np.diff(lattice, axis=0), axis=2).reshape(-1).tolist())
        pitch = float(np.median(neighbor_steps)) if neighbor_steps else 0.0
        if pitch <= 1 or float(np.median(residuals)) / pitch > 0.18:
            return None

        outer_grid = np.asarray(
            [
                [-1, -1],
                [inner_columns, -1],
                [inner_columns, inner_rows],
                [-1, inner_rows],
            ],
            dtype=np.float32,
        ).reshape((1, 4, 2))
        quad = cv2.perspectiveTransform(outer_grid, homography).reshape((4, 2))
        if not np.isfinite(quad).all():
            return None
        if (
            np.any(quad[:, 0] < 0)
            or np.any(quad[:, 0] >= image_width)
            or np.any(quad[:, 1] < 0)
            or np.any(quad[:, 1] >= image_height)
        ):
            return None
        edges = np.roll(quad, -1, axis=0) - quad
        turns = edges[:, 0] * np.roll(edges[:, 1], -1) - edges[:, 1] * np.roll(edges[:, 0], -1)
        if not (np.all(turns > 0) or np.all(turns < 0)):
            return None
        contour = quad.reshape((-1, 1, 2))
        if abs(float(cv2.contourArea(contour))) < pitch * pitch * 8:
            return None
        margin = float(cv2.pointPolygonTest(contour, (float(seed_center[0]), float(seed_center[1])), True))
        if margin < 0:
            return None

        # Check the alternating light/dark square centers across the complete board.
        cell_centers = np.asarray(
            [
                (column - 0.5, row - 0.5)
                for row in range(inner_rows + 1)
                for column in range(inner_columns + 1)
            ],
            dtype=np.float32,
        )
        offsets = np.asarray(
            [(-0.12, -0.12), (-0.12, 0.0), (-0.12, 0.12),
             (0.0, -0.12), (0.0, 0.0), (0.0, 0.12),
             (0.12, -0.12), (0.12, 0.0), (0.12, 0.12)],
            dtype=np.float32,
        )
        sample_grid = cell_centers[:, None, :] + offsets[None, :, :]
        sample_image = cv2.perspectiveTransform(sample_grid.reshape((1, -1, 2)), homography).reshape((-1, 9, 2))
        sample_xy = np.rint(sample_image).astype(np.int32)
        if (
            np.any(sample_xy[:, :, 0] < 0)
            or np.any(sample_xy[:, :, 0] >= image_width)
            or np.any(sample_xy[:, :, 1] < 0)
            or np.any(sample_xy[:, :, 1] >= image_height)
        ):
            return None
        cell_values = np.median(gray[sample_xy[:, :, 1], sample_xy[:, :, 0]], axis=1)
        parity = np.asarray(
            [((column + row) % 2) == 0 for row in range(inner_rows + 1) for column in range(inner_columns + 1)],
            dtype=bool,
        )
        parity_a = float(np.median(cell_values[parity]))
        parity_b = float(np.median(cell_values[~parity]))
        contrast = abs(parity_a - parity_b)
        if contrast < 28:
            return None
        midpoint = (parity_a + parity_b) / 2.0
        if parity_a < parity_b:
            consistency = (float(np.mean(cell_values[parity] < midpoint)) + float(np.mean(cell_values[~parity] > midpoint))) / 2
        else:
            consistency = (float(np.mean(cell_values[parity] > midpoint)) + float(np.mean(cell_values[~parity] < midpoint))) / 2
        if consistency < 0.72:
            return None

        quality = consistency + min(contrast / 160.0, 1.0) * 0.25 + min(margin / max(pitch, 1.0), 3.0) * 0.03
        quality -= float(np.median(residuals)) / pitch
        return {
            "quad": quad,
            "squares": (inner_columns + 1, inner_rows + 1),
            "quality": quality,
        }

    best = None
    seen_regions: set[tuple[int, int, int, int]] = set()
    for expansion in (1.25, 1.5, 2.0, 2.75, 3.75, 5.0, 6.75, 9.0, 12.0, 16.0, 22.0, 32.0):
        half_width = max(float(seed_width) * expansion / 2, 48)
        half_height = max(float(seed_height) * expansion / 2, 48)
        x0 = max(0, int(np.floor(seed_center[0] - half_width)))
        y0 = max(0, int(np.floor(seed_center[1] - half_height)))
        x1 = min(image_width, int(np.ceil(seed_center[0] + half_width)))
        y1 = min(image_height, int(np.ceil(seed_center[1] + half_height)))
        region = (x0, y0, x1, y1)
        if region in seen_regions or x1 - x0 < 100 or y1 - y0 < 100:
            continue
        seen_regions.add(region)
        crop = gray[y0:y1, x0:x1]
        candidates = []
        for pattern_size in BOARD_INNER_CORNER_SIZES:
            try:
                found, corners = cv2.findChessboardCornersSB(crop, pattern_size, flags)
            except cv2.error:
                continue
            if not found or corners is None:
                continue
            full_image_corners = np.asarray(corners, dtype=np.float32).reshape((-1, 2))
            full_image_corners += np.asarray([x0, y0], dtype=np.float32)
            candidate = validate_candidate(full_image_corners, pattern_size)
            if candidate is not None:
                candidates.append(candidate)
        if candidates:
            candidates.sort(key=lambda item: item["quality"], reverse=True)
            best = candidates[0]
            break

    if best is None:
        raise ValueError(
            "没有找到通过完整尺寸和黑白格校验的 9×12 棋盘。"
            "请确认棋盘完整出现在画面中，并把框画在目标棋盘内部几格。"
        )

    normalized = best["quad"] / np.asarray([image_width, image_height], dtype=np.float32)
    normalized_seed = seed / np.asarray([image_width, image_height], dtype=np.float32)
    return {
        "points": [{"x": float(x), "y": float(y)} for x, y in normalized],
        "seed_points": [{"x": float(x), "y": float(y)} for x, y in normalized_seed],
        "detected_squares": list(best["squares"]),
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "ManualBoardBoxTool/1.0"

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"[{self.log_date_time_string()}] {fmt % args}")

    def send_json(self, value: dict, status: int = 200, download: bool = False) -> None:
        body = json.dumps(value, ensure_ascii=False, indent=2 if download else None).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if download:
            self.send_header("Content-Disposition", 'attachment; filename="board_boxes.json"')
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path: Path, content_type: str | None = None) -> None:
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        route = urlparse(self.path).path
        try:
            if route in {"/", "/index.html"}:
                self.send_file(HTML_PATH, "text/html; charset=utf-8")
            elif route == "/api/data":
                annotations = read_annotations()
                names = image_names()
                boxes = {
                    name: value for name, value in annotations["boxes"].items()
                    if name in names and isinstance(value, list)
                }
                self.send_json({
                    "project": PROJECT_DIR.name,
                    "data_directory": DATA_DIR.name,
                    "images": [
                        {"name": path.name, "url": "/images/" + quote(path.name)}
                        for path in list_images()
                    ],
                    "boxes": boxes,
                    "image_sizes": annotations.get("image_sizes", {}),
                })
            elif route == "/api/export.json":
                self.send_json(read_annotations(), download=True)
            elif route.startswith("/images/"):
                name = unquote(route.removeprefix("/images/"))
                try:
                    path = resolve_image(name)
                except ValueError:
                    self.send_error(404)
                else:
                    self.send_file(path)
            else:
                self.send_error(404)
        except (OSError, ValueError) as exc:
            self.send_json({"error": str(exc)}, 500)

    def do_POST(self) -> None:  # noqa: N802
        route = urlparse(self.path).path
        if route == "/api/upload":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > MAX_UPLOAD_BYTES:
                    raise ValueError("图片为空或超过 100 MB。")
                query = parse_qs(urlparse(self.path).query)
                name = query.get("name", [""])[0]
                saved_name = save_uploaded_image(name, self.rfile.read(length))
                self.send_json({"ok": True, "name": saved_name})
            except (OSError, ValueError) as exc:
                self.send_json({"error": str(exc)}, 400)
            return
        if route == "/api/refine":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 100_000:
                    raise ValueError("请求内容为空或过大。")
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("请求数据格式不正确。")
                name = payload.get("name")
                if not isinstance(name, str) or name not in image_names():
                    raise ValueError("找不到这张图片。")
                result = refine_checkerboard(resolve_image(name), payload.get("points"))
                self.send_json({"ok": True, **result})
            except RuntimeError as exc:
                self.send_json({"error": str(exc)}, 500)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                self.send_json({"error": str(exc)}, 400)
            return
        if route != "/api/boxes":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 2_000_000:
                raise ValueError("请求内容为空或过大。")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("请求数据格式不正确。")
            name = payload.get("name")
            if not isinstance(name, str) or name not in image_names():
                raise ValueError("找不到这张图片。")
            boxes = validate_boxes(payload.get("boxes"))
            try:
                width = int(payload.get("image_width"))
                height = int(payload.get("image_height"))
            except (TypeError, ValueError) as exc:
                raise ValueError("图片尺寸无效。") from exc
            if width <= 0 or height <= 0:
                raise ValueError("图片尺寸无效。")
            with SAVE_LOCK:
                annotations = read_annotations()
                annotations["boxes"][name] = boxes
                annotations.setdefault("image_sizes", {})[name] = {"width": width, "height": height}
                save_annotations(annotations)
            self.send_json({"ok": True, "name": name, "box_count": len(boxes)})
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            self.send_json({"error": str(exc)}, 400)


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    url = f"http://{HOST}:{PORT}"
    print(f"手动画框网页已启动：{url}")
    threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n网页服务已停止。")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
