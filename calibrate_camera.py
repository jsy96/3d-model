#!/usr/bin/env python3
"""Detect the printed checkerboards, then calibrate a camera from labeled views.

Each board contains 9 x 12 squares, hence 8 x 11 inner corners. The user
labels the physical square pitch for each selected board detection; multiple
boards in one source image may be used as separate calibration views.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from calibration_grouping import update_group_metadata as shared_update_group_metadata


PATTERN_SIZES = ((8, 11), (11, 8))  # inner corners: columns, rows
DETECT_FLAGS = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
PROJECT_DATA = Path(__file__).resolve().parent / "生产项目" / "data"
DEFAULT_INPUT = str(PROJECT_DATA / "images")
DEFAULT_OUTPUT = str(PROJECT_DATA / "calibration_artifacts")
DEFAULT_QUICK_OUTPUT = str(PROJECT_DATA / "calibration_artifacts_review")
GROUP_EXPECTED_BOARDS = {"10": 4, "15": 2}


def image_files(folder: Path) -> list[Path]:
    return sorted(
        (p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda p: p.name.casefold(),
    )


def axis_starts(length: int, window: int, step: int) -> list[int]:
    if length <= window:
        return [0]
    starts = list(range(0, length - window + 1, step))
    final = length - window
    if starts[-1] != final:
        starts.append(final)
    return starts


def board_area(corners: np.ndarray) -> float:
    hull = cv2.convexHull(corners.astype(np.float32))
    return float(cv2.contourArea(hull))


def partial_pattern_sizes() -> list[tuple[int, int]]:
    """Return every contiguous sub-grid of the 8 x 11 inner-corner lattice."""
    sizes = {
        (cols, rows)
        for cols in range(4, 9)
        for rows in range(4, 12)
    }
    sizes |= {(rows, cols) for cols, rows in tuple(sizes)}
    return sorted(sizes, key=lambda size: (-(size[0] * size[1]), -max(size), -min(size)))


def make_detection(points: np.ndarray, pattern_size: tuple[int, int]) -> dict:
    points = points.reshape(-1, 2).astype(np.float32)
    return {
        "center": points.mean(axis=0),
        "corners": points,
        "pattern_size": [int(pattern_size[0]), int(pattern_size[1])],
        "area_px2": board_area(points),
    }


def detection_center(detection: dict) -> np.ndarray:
    if "center" in detection:
        return np.asarray(detection["center"], dtype=np.float64)
    return np.asarray([
        float(detection["center_x_px"]),
        float(detection["center_y_px"]),
    ], dtype=np.float64)


def detection_pattern_size(detection: dict) -> tuple[int, int]:
    if "pattern_size" in detection:
        return tuple(map(int, detection["pattern_size"]))
    return int(detection["pattern_columns"]), int(detection["pattern_rows"])


def grid_pitch_px(detection: dict) -> float:
    """Estimate the image-space square pitch from a detected corner lattice."""
    points = np.asarray(detection["corners"], dtype=np.float32).reshape(-1, 2)
    cols, rows = detection_pattern_size(detection)
    lattice = points.reshape(rows, cols, 2)
    steps = []
    if cols > 1:
        steps.extend(np.linalg.norm(lattice[:, 1:] - lattice[:, :-1], axis=2).reshape(-1).tolist())
    if rows > 1:
        steps.extend(np.linalg.norm(lattice[1:] - lattice[:-1], axis=2).reshape(-1).tolist())
    return float(np.median(steps)) if steps else 0.0


def group_rectangle_score(items: list[dict], indices: tuple[int, ...]) -> float | None:
    """Score how well four board centers form a 2x2 paper layout.

    The exact paper margins and inter-board spacing are intentionally not used.
    Only the repeated 2x2 arrangement, similar board scale, and non-collinearity
    are used as auxiliary evidence for grouping.
    """
    if len(indices) != 4:
        return None
    centers = np.asarray([detection_center(items[i]) for i in indices], dtype=np.float64)
    pitches = np.asarray([grid_pitch_px(items[i]) for i in indices], dtype=np.float64)
    pitch_mean = float(np.mean(pitches))
    if pitch_mean <= 0:
        return None
    pitch_cv = float(np.std(pitches) / pitch_mean)
    hull = cv2.convexHull(centers.astype(np.float32))
    hull_area = float(cv2.contourArea(hull))
    x0, y0 = centers.min(axis=0)
    x1, y1 = centers.max(axis=0)
    bbox_area = max(1.0, float((x1 - x0) * (y1 - y0)))
    # A 2x2 paper layout must have visible area; this rejects four nearly
    # collinear boards that happen to be close together.
    if hull_area / bbox_area < 0.16:
        return None

    scale = max(pitch_mean, 1.0)
    best = float("inf")
    for order in itertools.permutations(range(4)):
        p0, p1, p2, p3 = (centers[index] for index in order)
        edge_a = float(np.linalg.norm(p1 - p0))
        edge_b = float(np.linalg.norm(p2 - p0))
        if min(edge_a, edge_b) < 4.0 * scale:
            continue
        diagonal = max(edge_a + edge_b, scale)
        parallelogram_error = float(np.linalg.norm(p3 - (p1 + p2 - p0)) / diagonal)
        edge_balance = abs(math.log(max(edge_a, 1e-9) / max(edge_b, 1e-9)))
        best = min(best, 3.0 * parallelogram_error + 0.25 * edge_balance)
    if not math.isfinite(best):
        return None
    return best + pitch_cv


def infer_groups_for_image(detections: list[dict]) -> list[dict]:
    """Infer same-paper groups from already labeled single-board detections.

    A 1 cm paper is expected to contain four boards; a 1.5 cm paper contains
    two. The groups are auxiliary metadata. Calibration still consumes each
    board's own corner lattice and square pitch independently.
    """
    groups: list[dict] = []
    for spacing, expected in GROUP_EXPECTED_BOARDS.items():
        eligible = [
            index for index, item in enumerate(detections)
            if str(item.get("choice", "")) == spacing
        ]
        if len(eligible) < expected:
            continue
        candidates: list[tuple[float, tuple[int, ...]]] = []
        if expected == 4:
            for combo in itertools.combinations(eligible, expected):
                score = group_rectangle_score(detections, combo)
                if score is not None:
                    candidates.append((score, combo))
        else:
            # For the two-board target, use the closest compatible pair. The
            # exact inter-board gap is deliberately not part of the model.
            for first, second in itertools.combinations(eligible, 2):
                first_pitch = grid_pitch_px(detections[first])
                second_pitch = grid_pitch_px(detections[second])
                pitch = max((first_pitch + second_pitch) / 2.0, 1.0)
                distance = float(np.linalg.norm(
                    detection_center(detections[first]) - detection_center(detections[second])
                ))
                ratio = distance / pitch
                if ratio <= 36.0:
                    pitch_error = abs(first_pitch - second_pitch) / pitch
                    candidates.append((ratio / 36.0 + pitch_error, (first, second)))
        candidates.sort(key=lambda entry: (entry[0], tuple(detections[index]["board_id"] for index in entry[1])))
        used: set[int] = set()
        for score, indices in candidates:
            if any(index in used for index in indices):
                continue
            # For four-board groups, avoid accepting a geometrically weak
            # cross-paper combination when a better layout is available.
            if expected == 4 and score > 1.0:
                continue
            used.update(indices)
            board_ids = [detections[index]["board_id"] for index in indices]
            groups.append({
                "grid_spacing_mm": int(spacing),
                "expected_boards": expected,
                "board_ids": board_ids,
                "score": float(score),
                "confidence": float(max(0.0, min(1.0, math.exp(-score)))),
            })
    groups.sort(key=lambda group: (group["grid_spacing_mm"], group["board_ids"]))
    return groups


def update_group_metadata(manifest: dict, choices: dict[str, str]) -> None:
    """Attach paper-group metadata to a manifest in place."""
    for image_index, image in enumerate(manifest.get("images", []), start=1):
        detections = image.get("detections", [])
        for item in detections:
            key = f"{image['image']}\t{item['board_id']}"
            item["choice"] = choices.get(key, "")
            for field in ("group_id", "group_spacing_mm", "group_expected_boards", "group_rank", "group_confidence"):
                item.pop(field, None)
        groups = infer_groups_for_image(detections)
        serialized_groups = []
        for group_index, group in enumerate(groups, start=1):
            group_id = f"G{image_index:02d}-{group['grid_spacing_mm']}cm-{group_index:02d}"
            group["group_id"] = group_id
            for rank, board_id in enumerate(group["board_ids"], start=1):
                item = next(item for item in detections if item["board_id"] == board_id)
                item["group_id"] = group_id
                item["group_spacing_mm"] = group["grid_spacing_mm"]
                item["group_expected_boards"] = group["expected_boards"]
                item["group_rank"] = rank
                item["group_confidence"] = round(group["confidence"], 4)
            serialized_groups.append({
                "group_id": group_id,
                "grid_spacing_mm": group["grid_spacing_mm"],
                "expected_boards": group["expected_boards"],
                "board_ids": group["board_ids"],
                "confidence": round(group["confidence"], 4),
            })
        image["groups"] = serialized_groups


def shared_corner_count(first: np.ndarray, second: np.ndarray, tolerance_px: float = 10.0) -> int:
    delta = first[:, None, :] - second[None, :, :]
    distances_squared = np.einsum("ijk,ijk->ij", delta, delta)
    return int(np.count_nonzero(np.min(distances_squared, axis=1) <= tolerance_px * tolerance_px))


def deduplicate_detections(detections: list[dict]) -> list[dict]:
    """Merge overlapping crop hits while keeping the largest contiguous grid."""
    ranked = sorted(
        detections,
        key=lambda item: (len(item["corners"]), item["area_px2"]),
        reverse=True,
    )
    kept: list[dict] = []
    for candidate in ranked:
        duplicate = False
        for existing in kept:
            smaller_count = min(len(candidate["corners"]), len(existing["corners"]))
            required_overlap = max(6, int(math.ceil(smaller_count * 0.45)))
            if shared_corner_count(candidate["corners"], existing["corners"]) >= required_overlap:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    kept.sort(key=lambda item: (float(item["center"][1]), float(item["center"][0])))
    return kept


def tile_starts(length: int, window: int, step: int) -> list[int]:
    if length <= window:
        return [0]
    starts = list(range(0, length - window + 1, step))
    final = length - window
    if starts[-1] != final:
        starts.append(final)
    return starts


def partial_tile_hits(gray: np.ndarray, full_only: bool = False) -> list[dict]:
    """Search overlapping crops for full boards and, optionally, partial grids."""
    height, width = gray.shape[:2]
    pattern_sizes = list(PATTERN_SIZES) if full_only else partial_pattern_sizes()
    hits: list[dict] = []
    # The smaller tiles separate adjacent boards that compete in a large crop.
    for window in (1400, 1000, 700):
        window_w, window_h = min(window, width), min(window, height)
        step = max(1, round(window * 0.55))
        for y0 in tile_starts(height, window_h, step):
            for x0 in tile_starts(width, window_w, step):
                crop = gray[y0 : y0 + window_h, x0 : x0 + window_w]
                if not full_only:
                    strong_corners = cv2.goodFeaturesToTrack(
                        crop,
                        maxCorners=700,
                        qualityLevel=0.01,
                        minDistance=5,
                        blockSize=3,
                    )
                    if strong_corners is None or len(strong_corners) < 12:
                        continue

                for cols, rows in pattern_sizes:
                    found, corners = cv2.findChessboardCornersSB(
                        crop,
                        (cols, rows),
                        flags=cv2.CALIB_CB_NORMALIZE_IMAGE,
                    )
                    if not found or corners is None or len(corners) != cols * rows:
                        continue
                    points = corners.reshape(-1, 2).astype(np.float32)
                    points += np.array([x0, y0], dtype=np.float32)
                    hits.append(make_detection(points, (cols, rows)))
                    # Keep the largest pattern found in this crop. Neighboring
                    # overlapping crops reveal other boards and larger patches.
                    break
    return hits


def detect_boards(gray: np.ndarray, full_only: bool = False) -> list[dict]:
    """Find full boards and, unless requested otherwise, contiguous partial grids."""
    height, width = gray.shape[:2]
    window_w, window_h = min(2200, width), min(1500, height)
    xs = axis_starts(width, window_w, 1000)
    ys = axis_starts(height, window_h, 700)
    hits: list[dict] = []

    for y0 in ys:
        for x0 in xs:
            crop = gray[y0 : y0 + window_h, x0 : x0 + window_w]
            for cols, rows in PATTERN_SIZES:
                found, corners = cv2.findChessboardCornersSB(crop, (cols, rows), flags=DETECT_FLAGS)
                if not found or corners is None or len(corners) != cols * rows:
                    continue
                points = corners.reshape(-1, 2).astype(np.float32)
                points += np.array([x0, y0], dtype=np.float32)
                hits.append(make_detection(points, (cols, rows)))
                break

    hits.extend(partial_tile_hits(gray, full_only=full_only))
    detections = deduplicate_detections(hits)
    if full_only:
        detections = [item for item in detections if len(item["corners"]) == 8 * 11]
    return detections


def serialized_detection_to_candidate(detection: dict) -> dict:
    points = np.asarray(detection["corners"], dtype=np.float32).reshape(-1, 2)
    return {
        "center": points.mean(axis=0),
        "corners": points,
        "pattern_size": [int(detection["pattern_columns"]), int(detection["pattern_rows"])],
        "area_px2": float(detection["board_area_px2"]),
        "board_id": detection["board_id"],
    }


def preserve_ids_and_candidates(
    detections: list[dict],
    previous: list[dict],
    next_id: int,
    keep_missing_previous: bool = True,
) -> tuple[list[dict], int]:
    """Keep old IDs and detections, matching new hits by shared corner pixels."""
    detections = deduplicate_detections(detections)
    old_candidates = [serialized_detection_to_candidate(item) for item in previous]
    used_old: set[int] = set()
    results: list[dict] = []

    for item in detections:
        matches: list[tuple[int, int]] = []
        for old_index, old in enumerate(old_candidates):
            if old_index in used_old:
                continue
            smaller_count = min(len(item["corners"]), len(old["corners"]))
            required_overlap = max(8, int(math.ceil(smaller_count * 0.65)))
            overlap = shared_corner_count(item["corners"], old["corners"])
            if overlap >= required_overlap:
                matches.append((overlap, old_index))

        if matches:
            _, old_index = max(matches)
            used_old.add(old_index)
            old = old_candidates[old_index]
            item["board_id"] = old["board_id"]
            if len(old["corners"]) > len(item["corners"]):
                item = old
        else:
            item["board_id"] = f"B{next_id:03d}"
            next_id += 1
        results.append(item)

    if keep_missing_previous:
        for old_index, old in enumerate(old_candidates):
            if old_index not in used_old:
                results.append(old)

    results.sort(key=lambda item: (float(item["center"][1]), float(item["center"][0])))
    return results, next_id


def raw_to_display(points: np.ndarray, raw_shape: tuple[int, int], orientation: int) -> np.ndarray:
    """Map raw sensor pixels to EXIF display pixels for human-readable previews."""
    height, width = raw_shape
    p = points.copy().astype(np.float32)
    x, y = p[:, 0].copy(), p[:, 1].copy()
    if orientation == 3:
        p[:, 0], p[:, 1] = width - 1 - x, height - 1 - y
    elif orientation == 6:  # rotate 90 degrees clockwise
        p[:, 0], p[:, 1] = height - 1 - y, x
    elif orientation == 8:  # rotate 90 degrees counter-clockwise
        p[:, 0], p[:, 1] = y, width - 1 - x
    elif orientation == 2:
        p[:, 0] = width - 1 - x
    elif orientation == 4:
        p[:, 1] = height - 1 - y
    elif orientation == 5:
        p[:, 0], p[:, 1] = y, x
    elif orientation == 7:
        p[:, 0], p[:, 1] = height - 1 - y, width - 1 - x
    return p


def exif_orientation(path: Path) -> int:
    try:
        with Image.open(path) as image:
            return int(image.getexif().get(274, 1))
    except Exception:
        return 1


def preview_image(path: Path, detections: list[dict], raw_shape: tuple[int, int], out_path: Path) -> Image.Image:
    with Image.open(path) as image:
        display = ImageOps.exif_transpose(image).convert("RGB")
    canvas = cv2.cvtColor(np.asarray(display), cv2.COLOR_RGB2BGR)
    orientation = exif_orientation(path)
    for item in detections:
        pts = raw_to_display(np.asarray(item["corners"]), raw_shape, orientation)
        hull = cv2.convexHull(pts.astype(np.float32)).reshape(-1, 2)
        polygon = np.round(hull).astype(np.int32).reshape(-1, 1, 2)
        width = max(8, round(min(display.size) / 380)) if item.get("recommended") else 5
        color = (20, 20, 230) if item.get("recommended") else (20, 150, 245)
        cv2.polylines(canvas, [polygon], True, color, width, cv2.LINE_AA)
        cx, cy = raw_to_display(np.asarray([item["center"]]), raw_shape, orientation)[0]
        label = str(item["board_id"])
        cx, cy = int(round(cx)), int(round(cy))
        cv2.rectangle(canvas, (cx - 140, cy - 58), (cx + 140, cy + 58), (255, 255, 255), -1)
        cv2.rectangle(canvas, (cx - 140, cy - 58), (cx + 140, cy + 58), color, 5)
        cv2.putText(canvas, label, (cx - 112, cy + 30), cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 6, cv2.LINE_AA)
    display = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    display.save(out_path, quality=88)
    return display


def choose_recommended(detections: list[dict]) -> int:
    """Choose the detection covering the largest clear grid as a visual guide."""
    return max(range(len(detections)), key=lambda index: detections[index]["area_px2"])


def load_previous_choices(output: Path, previous_manifest: dict | None) -> dict[str, str]:
    """Read existing labels before prepare rewrites the detection artifacts."""
    choices: dict[str, str] = {}
    annotations_path = output / "calibration_annotations.csv"
    if annotations_path.exists():
        with annotations_path.open(newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                key = f"{row.get('image', '')}\t{row.get('board_id', '')}"
                value = (row.get("grid_spacing_mm") or "").strip()
                if value in {"10", "15"}:
                    choices[key] = value

    state_path = output / "labeler_state.json"
    manifest_path = output / "detections.json"
    if state_path.exists() and previous_manifest is not None and manifest_path.exists():
        try:
            saved = json.loads(state_path.read_text(encoding="utf-8"))
            expected_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            if saved.get("manifest_sha256") == expected_hash:
                for key, value in saved.get("choices", {}).items():
                    if value in {"10", "15", "skip"}:
                        choices[key] = value
        except (OSError, json.JSONDecodeError, AttributeError):
            pass
    return choices


def prepare(args: argparse.Namespace) -> int:
    source = Path(args.input).resolve()
    output = Path(args.output).resolve()
    quick_mode = bool(getattr(args, "quick", False))
    output.mkdir(parents=True, exist_ok=True)
    overlays_dir = output / "previews"
    overlays_dir.mkdir(parents=True, exist_ok=True)

    previous_manifest_path = output / "detections.json"
    previous_manifest = None
    if previous_manifest_path.exists():
        try:
            previous_manifest = json.loads(previous_manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous_manifest = None
    previous_choices = load_previous_choices(output, previous_manifest)
    previous_images = {
        item["image"]: item["detections"]
        for item in (previous_manifest or {}).get("images", [])
    }
    existing_ids = [
        int(board["board_id"][1:])
        for image in (previous_manifest or {}).get("images", [])
        for board in image["detections"]
        if board.get("board_id", "").startswith("B") and board["board_id"][1:].isdigit()
    ]
    next_id = max(existing_ids, default=0) + 1

    files = image_files(source)
    if not files:
        print(f"No supported images found in {source}", file=sys.stderr)
        return 2

    font = ImageFont.load_default()
    manifest = {"pattern_squares": [9, 12], "sensor_image_size_px": None, "images": []}
    all_rows: list[dict] = []
    contact_items: list[tuple[Image.Image, str]] = []

    for image_index, path in enumerate(files, start=1):
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE | cv2.IMREAD_IGNORE_ORIENTATION)
        if gray is None:
            print(f"Unreadable: {path.name}", file=sys.stderr)
            continue
        image_size = [int(gray.shape[1]), int(gray.shape[0])]
        if manifest["sensor_image_size_px"] is None:
            manifest["sensor_image_size_px"] = image_size

        found_detections = detect_boards(gray, full_only=quick_mode)
        if quick_mode:
            detections, next_id = preserve_ids_and_candidates(
                found_detections,
                previous_images.get(path.name, []),
                next_id,
                keep_missing_previous=False,
            )
        else:
            detections, next_id = preserve_ids_and_candidates(
                found_detections,
                previous_images.get(path.name, []),
                next_id,
            )
        for item in detections:
            item["recommended"] = quick_mode
        recommended_index = choose_recommended(detections) if detections else None
        if recommended_index is not None and not quick_mode:
            detections[recommended_index]["recommended"] = True

        serializable = []
        for item in detections:
            center_x, center_y = map(float, item["center"])
            row = {
                "image": path.name,
                "board_id": item["board_id"],
                "recommended": "yes" if item.get("recommended") else "",
                "center_x_px": f"{center_x:.1f}",
                "center_y_px": f"{center_y:.1f}",
                "board_area_px2": f"{item['area_px2']:.0f}",
                "pattern_columns": item["pattern_size"][0],
                "pattern_rows": item["pattern_size"][1],
            }
            all_rows.append(row)
            serializable.append({
                **row,
                "corners": np.asarray(item["corners"], dtype=float).tolist(),
            })

        manifest["images"].append({
            "image": path.name,
            "image_size_px": image_size,
            "exif_orientation": exif_orientation(path),
            "detections": serializable,
        })

        preview_path = overlays_dir / f"{image_index:02d}.jpg"
        display = preview_image(path, detections, tuple(gray.shape[:2]), preview_path)
        display.thumbnail((460, 340))
        if quick_mode:
            caption = f"Frame {image_index:02d}  {len(detections)} full board(s), auto 1 cm"
            status = f", auto-labeled {len(detections)} full board(s) at 1 cm"
        else:
            selected_label = detections[recommended_index]["board_id"] if recommended_index is not None else "no board"
            caption = f"Frame {image_index:02d}  selected {selected_label}"
            status = f", guide {selected_label}" if recommended_index is not None else ", no board"
        contact_items.append((display.copy(), caption))
        print(f"{path.name}: {len(detections)} grid candidate(s){status}", flush=True)

    valid_keys = {
        f"{image['image']}\t{board['board_id']}"
        for image in manifest["images"]
        for board in image["detections"]
    }
    if quick_mode:
        choices = {
            key: previous_choices.get(key, "10")
            if previous_choices.get(key) in {"10", "skip"}
            else "10"
            for key in valid_keys
        }
    else:
        choices = {key: value for key, value in previous_choices.items() if key in valid_keys}
    shared_update_group_metadata(manifest, choices)
    manifest_path = output / "detections.json"
    manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2)
    manifest_path.write_text(manifest_text, encoding="utf-8")
    detection_rows = []
    for image in manifest["images"]:
        for board in image["detections"]:
            detection_rows.append({
                "image": image["image"],
                "board_id": board["board_id"],
                "recommended": board.get("recommended", ""),
                "center_x_px": board["center_x_px"],
                "center_y_px": board["center_y_px"],
                "board_area_px2": board["board_area_px2"],
                "pattern_columns": board["pattern_columns"],
                "pattern_rows": board["pattern_rows"],
                "choice": board.get("choice", ""),
                "group_id": board.get("group_id", ""),
                "group_spacing_mm": board.get("group_spacing_mm", ""),
                "group_expected_boards": board.get("group_expected_boards", ""),
                "group_rank": board.get("group_rank", ""),
                "group_confidence": board.get("group_confidence", ""),
            })
    with (output / "all_board_detections.csv").open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = [
            "image", "board_id", "recommended", "center_x_px", "center_y_px",
            "board_area_px2", "pattern_columns", "pattern_rows", "choice",
            "group_id", "group_spacing_mm", "group_expected_boards", "group_rank",
            "group_confidence",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(detection_rows)
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    state_path = output / "labeler_state.json"
    temp_state = state_path.with_suffix(".json.tmp")
    temp_state.write_text(json.dumps({
        "version": 1,
        "manifest_sha256": manifest_hash,
        "choices": choices,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_state.replace(state_path)

    annotation_rows = []
    for image in manifest["images"]:
        for board in image["detections"]:
            key = f"{image['image']}\t{board['board_id']}"
            choice = choices.get(key)
            if choice in {"10", "15"}:
                annotation_rows.append({
                    "image": image["image"],
                    "board_id": board["board_id"],
                    "grid_spacing_mm": choice,
                    "note": (
                        "Automatically assigned 10 mm in quick mode"
                        if quick_mode
                        else "Preserved from previous annotation" if key in previous_choices else ""
                    ),
                })
    with (output / "calibration_annotations.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "board_id", "grid_spacing_mm", "note"])
        writer.writeheader()
        writer.writerows(annotation_rows)

    cols = 3
    cell_w, cell_h = 480, 380
    rows = math.ceil(len(contact_items) / cols)
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), (235, 235, 235))
    draw = ImageDraw.Draw(sheet)
    for index, (thumb, caption) in enumerate(contact_items):
        x, y = (index % cols) * cell_w, (index // cols) * cell_h
        sheet.paste(thumb, (x + (cell_w - thumb.width) // 2, y + 28))
        draw.text((x + 8, y + 6), caption, fill=(20, 20, 20), font=font)
    sheet.save(output / "board_selection_contact_sheet.jpg", quality=90)

    new_count = sum(
        1
        for image in manifest["images"]
        for board in image["detections"]
        if board["board_id"] not in {
            old_board["board_id"]
            for old_board in previous_images.get(image["image"], [])
        }
    )
    print(f"\nPrepared {len(all_rows)} grid candidate(s) from {len(files)} image(s).")
    if quick_mode:
        print(f"Automatically labeled {len(annotation_rows)} complete 12 x 9-square board(s) as 10 mm.")
    else:
        print(f"Preserved {len(choices)} existing size choice(s); {new_count} new candidate(s) need review.")
        print(f"Edit: {output / 'calibration_annotations.csv'}")
    print(f"Review: {output / 'board_selection_contact_sheet.jpg'}")
    usable_images = len({row["image"] for row in annotation_rows})
    if quick_mode and usable_images < 10:
        print(
            f"Warning: only {usable_images} photos contain a complete board; at least 10 are required.",
            file=sys.stderr,
        )
    elif not quick_mode and len(annotation_rows) < 10:
        print("Warning: fewer than 10 labeled board views were found.", file=sys.stderr)
    return 0


def object_points_for(cols: int, rows: int, square_mm: float) -> np.ndarray:
    grid = np.zeros((rows * cols, 3), dtype=np.float32)
    grid[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2).astype(np.float32)
    grid[:, :2] *= np.float32(square_mm)
    return grid


def residuals_for_view(obj: np.ndarray, img: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, camera: np.ndarray, dist: np.ndarray) -> np.ndarray:
    projected, _ = cv2.projectPoints(obj, rvec, tvec, camera, dist)
    delta = projected.reshape(-1, 2) - img.reshape(-1, 2)
    return np.linalg.norm(delta, axis=1)


def summarize_errors(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean_px": float(array.mean()),
        "median_px": float(np.median(array)),
        "p95_px": float(np.percentile(array, 95)),
        "max_px": float(array.max()),
        "point_rmse_px": float(np.sqrt(np.mean(array ** 2))),
    }


def cross_validate(views: list[dict], image_size: tuple[int, int], folds: int = 5) -> dict:
    if len(views) < 10:
        return {"folds": 0, "status": "not_run: at least 10 views are required"}
    folds = min(folds, len(views))
    errors: list[float] = []
    per_view: list[dict] = []
    test_folds = np.array_split(np.arange(len(views)), folds)
    try:
        for fold, test_group in enumerate(test_folds):
            test_indices = [int(i) for i in test_group]
            test_set = set(test_indices)
            train_indices = [i for i in range(len(views)) if i not in test_set]
            obj_train = [views[i]["object_points"] for i in train_indices]
            img_train = [views[i]["image_points"] for i in train_indices]
            _, camera, dist, _, _ = cv2.calibrateCamera(obj_train, img_train, image_size, None, None)
            for i in test_indices:
                view = views[i]
                ok, rvec, tvec = cv2.solvePnP(view["object_points"], view["image_points"], camera, dist)
                if not ok:
                    continue
                point_errors = residuals_for_view(view["object_points"], view["image_points"], rvec, tvec, camera, dist)
                errors.extend(point_errors.tolist())
                per_view.append({"image": view["image"], "board_id": view["board_id"], "fold": fold + 1, "rmse_px": float(np.sqrt(np.mean(point_errors ** 2)))})
        if not errors:
            return {"folds": folds, "status": "failed: no held-out errors"}
        return {"folds": folds, "status": "ok", "metrics": summarize_errors(errors), "per_view": per_view}
    except cv2.error as exc:
        return {"folds": folds, "status": f"failed: {exc}"}


def calibrate(args: argparse.Namespace) -> int:
    output = Path(args.output).resolve()
    manifest_path = output / "detections.json"
    annotations_path = Path(args.annotations).resolve() if args.annotations else output / "calibration_annotations.csv"
    if not manifest_path.exists() or not annotations_path.exists():
        print("Run the prepare command first to create detections and an annotation CSV.", file=sys.stderr)
        return 2

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    lookup: dict[tuple[str, str], dict] = {}
    for image in manifest["images"]:
        for detection in image["detections"]:
            lookup[(image["image"], detection["board_id"])] = detection

    views = []
    missing = []
    with annotations_path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            value = (row.get("grid_spacing_mm") or "").strip()
            if not value:
                missing.append(row.get("board_id", "(unknown)"))
                continue
            try:
                spacing = float(value)
            except ValueError:
                print(f"Invalid grid spacing for {row.get('board_id')}: {value}", file=sys.stderr)
                return 2
            if not math.isclose(spacing, 10.0, abs_tol=1e-6) and not math.isclose(spacing, 15.0, abs_tol=1e-6):
                print(f"Grid spacing must be 10 or 15 mm; got {value} for {row.get('board_id')}.", file=sys.stderr)
                return 2
            key = (row["image"], row["board_id"])
            detection = lookup.get(key)
            if detection is None:
                print(f"Unknown image/board pair in annotation CSV: {key}", file=sys.stderr)
                return 2
            cols = int(detection["pattern_columns"])
            rows = int(detection["pattern_rows"])
            corners = np.asarray(detection["corners"], dtype=np.float32).reshape(-1, 1, 2)
            views.append({
                "image": row["image"],
                "board_id": row["board_id"],
                "spacing_mm": spacing,
                "pattern_size": [cols, rows],
                "object_points": object_points_for(cols, rows, spacing),
                "image_points": corners,
            })

    if missing:
        print("Fill grid_spacing_mm with 10 or 15 for every selected board before calibration.", file=sys.stderr)
        print("Unlabeled IDs: " + ", ".join(missing), file=sys.stderr)
        return 2
    if len(views) < 10:
        print(f"Only {len(views)} labeled views; at least 10 are required for this report.", file=sys.stderr)
        return 2
    image_sizes = {tuple(manifest_image["image_size_px"]) for manifest_image in manifest["images"]}
    if len(image_sizes) != 1:
        print(f"Images do not share one raw sensor size: {sorted(image_sizes)}", file=sys.stderr)
        return 2
    image_size = next(iter(image_sizes))  # OpenCV expects width, height

    rms, camera, dist, rvecs, tvecs = cv2.calibrateCamera(
        [view["object_points"] for view in views],
        [view["image_points"] for view in views],
        image_size,
        None,
        None,
    )
    per_view = []
    point_errors: list[float] = []
    for view, rvec, tvec in zip(views, rvecs, tvecs):
        errors = residuals_for_view(view["object_points"], view["image_points"], rvec, tvec, camera, dist)
        point_errors.extend(errors.tolist())
        per_view.append({
            "image": view["image"],
            "board_id": view["board_id"],
            "grid_spacing_mm": view["spacing_mm"],
            "corners": int(len(errors)),
            "rmse_px": float(np.sqrt(np.mean(errors ** 2))),
            "median_px": float(np.median(errors)),
            "p95_px": float(np.percentile(errors, 95)),
        })
    fit_metrics = summarize_errors(point_errors)
    cv_metrics = cross_validate(views, image_size)

    result = {
        "model": "OpenCV pinhole, 5 distortion coefficients",
        "image_size_px": {"width": image_size[0], "height": image_size[1]},
        "views": len(views),
        "corners": int(sum(item["corners"] for item in per_view)),
        "rms_reprojection_px_opencv": float(rms),
        "fit_point_error": fit_metrics,
        "cross_validation_5_fold": cv_metrics,
        "camera_matrix": camera.tolist(),
        "intrinsics_px": {"fx": float(camera[0, 0]), "fy": float(camera[1, 1]), "cx": float(camera[0, 2]), "cy": float(camera[1, 2])},
        "distortion_order": ["k1", "k2", "p1", "p2", "k3"],
        "distortion_coefficients": dist.reshape(-1).tolist(),
        "per_view": per_view,
        "coordinate_unit": "millimeter",
    }
    (output / "camera_calibration.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    sample_path = Path(args.input).resolve() / views[0]["image"]
    sample = cv2.imread(str(sample_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if sample is not None:
        new_camera, _ = cv2.getOptimalNewCameraMatrix(camera, dist, image_size, 0)
        undistorted = cv2.undistort(sample, camera, dist, None, new_camera)
        cv2.imwrite(str(output / "undistorted_sample.jpg"), undistorted, [cv2.IMWRITE_JPEG_QUALITY, 92])

    report_lines = [
        "# Camera calibration and accuracy report",
        "",
        f"- Source images: {len(manifest['images'])}",
        f"- Labeled calibration views: {len(views)}",
        f"- Inner corners: 8 x 11 per view; total {result['corners']}",
        f"- Image size used: {image_size[0]} x {image_size[1]} px (EXIF rotation ignored for a consistent sensor coordinate frame)",
        f"- Square sizes: {', '.join(f'{v:g} mm' for v in sorted({view['spacing_mm'] for view in views}))}",
        "",
        "## Intrinsic parameters",
        "",
        f"- fx = {camera[0, 0]:.6f} px; fy = {camera[1, 1]:.6f} px",
        f"- cx = {camera[0, 2]:.6f} px; cy = {camera[1, 2]:.6f} px",
        f"- Distortion `[k1, k2, p1, p2, k3]` = `{np.array2string(dist.reshape(-1), precision=10, separator=', ')}`",
        "",
        "## Accuracy",
        "",
        f"- OpenCV calibration RMS: {rms:.6f} px",
        f"- Reprojection point RMSE: {fit_metrics['point_rmse_px']:.6f} px",
        f"- Reprojection median / 95th percentile / max: {fit_metrics['median_px']:.6f} / {fit_metrics['p95_px']:.6f} / {fit_metrics['max_px']:.6f} px",
    ]
    if cv_metrics.get("status") == "ok":
        m = cv_metrics["metrics"]
        report_lines.extend([
            f"- {cv_metrics['folds']}-fold held-out reprojection RMSE: {m['point_rmse_px']:.6f} px",
            f"- Held-out median / 95th percentile / max: {m['median_px']:.6f} / {m['p95_px']:.6f} / {m['max_px']:.6f} px",
        ])
    else:
        report_lines.append(f"- Cross-validation: {cv_metrics.get('status', 'not available')}")
    report_lines.extend([
        "",
        "Per-view residuals are listed in `camera_calibration.json`. The fit residual describes how closely the model explains the calibration observations; the held-out result is a more useful check of generalization. Neither metric alone guarantees measurement accuracy in the scene.",
        "",
        "## Per-view reprojection RMSE",
        "",
        "| Image | Board | Pitch (mm) | RMSE (px) | 95th percentile (px) |",
        "|---|---:|---:|---:|---:|",
    ])
    for item in per_view:
        report_lines.append(f"| {item['image']} | {item['board_id']} | {item['grid_spacing_mm']:g} | {item['rmse_px']:.5f} | {item['p95_px']:.5f} |")
    report_lines.append("")
    (output / "calibration_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Calibration saved to {output / 'camera_calibration.json'}")
    print(f"Accuracy report saved to {output / 'calibration_report.md'}")
    print(f"OpenCV RMS: {rms:.6f} px; held-out RMSE: {cv_metrics.get('metrics', {}).get('point_rmse_px', float('nan')):.6f} px")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect checkerboards and calibrate a camera from labeled views.")
    parser.add_argument("command", choices=("prepare", "calibrate"), help="prepare detections or calibrate from labeled views")
    parser.add_argument("--input", default=DEFAULT_INPUT, help=f"folder containing source photos (default: {DEFAULT_INPUT})")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help=f"output folder (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--annotations", default=None, help="annotation CSV path for calibrate (default: output/calibration_annotations.csv)")
    args = parser.parse_args()
    args.quick = False
    return prepare(args) if args.command == "prepare" else calibrate(args)


if __name__ == "__main__":
    raise SystemExit(main())
