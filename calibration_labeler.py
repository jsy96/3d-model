#!/usr/bin/env python3
"""Serve a local checkerboard detection and annotation page."""

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
            if not 3 <= len(box["points"]) <= 8:
                raise ValueError("棋盘轮廓必须包含 3 到 8 个顶点。")
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
                if not isinstance(seed_points, list) or not 3 <= len(seed_points) <= 8:
                    raise ValueError("原始框选区域必须包含 3 到 8 个顶点。")
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
            for field in ("partial", "needs_review"):
                if field in box:
                    if not isinstance(box[field], bool):
                        raise ValueError("棋盘检测状态格式不正确。")
                    item[field] = box[field]
            if "confidence" in box:
                try:
                    value = float(box["confidence"])
                except (TypeError, ValueError) as exc:
                    raise ValueError("置信度格式不正确。") from exc
                if not 0 <= value <= 1:
                    raise ValueError("置信度必须在 0 到 1 之间。")
                item["confidence"] = value
            if box.get("source") in ("automatic", "refined"):
                item["source"] = box["source"]
            corners, indices = box.get("corners"), box.get("grid_indices")
            if corners is not None or indices is not None:
                if not isinstance(corners, list) or not isinstance(indices, list) or not 12 <= len(corners) <= 88 or len(corners) != len(indices):
                    raise ValueError("角点与网格坐标数量不一致。")
                for corner, index in zip(corners, indices):
                    if not isinstance(corner, list) or len(corner) != 2 or any(not isinstance(v, (int, float)) or not 0 <= v <= 1 for v in corner):
                        raise ValueError("角点坐标格式不正确。")
                    if not isinstance(index, list) or len(index) != 2 or any(type(v) is not int or not 0 <= v <= 11 for v in index):
                        raise ValueError("网格坐标格式不正确。")
                item["corners"], item["grid_indices"] = corners, indices
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


DETECTION_LOCK = threading.Lock()
DETECTION_CACHE: dict[tuple, dict] = {}


def detect_checkerboards(image_path: Path) -> dict:
    try:
        from checkerboard_detector import detect_image
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"缺少图像处理依赖：{exc.name}") from exc
    stat = image_path.stat()
    key = (str(image_path.resolve()), stat.st_mtime_ns, stat.st_size)
    with DETECTION_LOCK:
        if key not in DETECTION_CACHE:
            result = detect_image(image_path)
            while len(DETECTION_CACHE) >= 4:
                DETECTION_CACHE.pop(next(iter(DETECTION_CACHE)))
            DETECTION_CACHE[key] = result
        return DETECTION_CACHE[key]


def refine_checkerboard(image_path: Path, rough_points: object) -> dict:
    """Select the automatically detected board containing the original seed."""
    if not isinstance(rough_points, list) or not 3 <= len(rough_points) <= 8:
        raise ValueError("请先画一个矩形框。")
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
        points.append({"x": x, "y": y})
    result = detect_checkerboards(image_path)
    import cv2
    import numpy as np

    center = tuple(np.mean([[p["x"], p["y"]] for p in points], axis=0).tolist())
    candidates = []
    for box in result["boxes"]:
        contour = np.array([[p["x"], p["y"]] for p in box["points"]], np.float32)
        if cv2.pointPolygonTest(contour, center, False) >= 0:
            candidates.append(box)
    if not candidates:
        # One focused retry retains a useful role for manual seeds when global
        # thresholding misses a small or poorly lit board. No expansion loop.
        from checkerboard_detector import find_boards

        gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        height, width = gray.shape
        seed = np.array([[p["x"] * width, p["y"] * height] for p in points])
        half_size = np.maximum(np.ptp(seed, axis=0) * 2, [width * 0.18, height * 0.18])
        pixel_center = np.array(center) * [width, height]
        x0, y0 = np.maximum(pixel_center - half_size, 0).astype(int)
        x1, y1 = np.minimum(pixel_center + half_size, [width, height]).astype(int)
        local_result = find_boards(gray[y0:y1, x0:x1])
        for local_box in local_result["boxes"]:
            contour = np.array([[p["x"], p["y"]] for p in local_box["points"]], np.float32)
            contour = (contour * [x1 - x0, y1 - y0] + [x0, y0]) / [width, height]
            if cv2.pointPolygonTest(contour.astype(np.float32), center, False) < 0:
                continue
            candidates.append({
                **local_box,
                "points": [{"x": float(x), "y": float(y)} for x, y in contour],
                "corners": ((np.array(local_box["corners"]) * [x1 - x0, y1 - y0] + [x0, y0]) / [width, height]).tolist(),
            })
    if not candidates:
        raise ValueError("此处没有检测到可靠的棋盘网格，请把框画在可见黑白格内部，或使用“自动检测本图”查看候选")
    best = max(candidates, key=lambda box: (len(box.get("corners", [])), box.get("confidence", 0)))
    return {**best, "seed_points": points, "source": "refined"}


class Handler(BaseHTTPRequestHandler):
    server_version = "CheckerboardTool/2.0"

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
        if route in {"/api/refine", "/api/detect"}:
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
                result = (detect_checkerboards(resolve_image(name)) if route == "/api/detect"
                          else refine_checkerboard(resolve_image(name), payload.get("points")))
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
    print(f"棋盘格检测网页已启动：{url}")
    threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n网页服务已停止。")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
