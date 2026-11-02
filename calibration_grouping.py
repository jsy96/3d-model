"""Infer same-paper checkerboard groups from labeled board detections.

This module deliberately uses only the Python standard library. It is used by
the labeler as well as the OpenCV detector, so changing a board's size choice
can refresh group metadata without running image detection again.
"""

from __future__ import annotations

import itertools
import math


EXPECTED_BOARDS = {"10": 4, "15": 2}


def _center(item: dict) -> tuple[float, float]:
    if "center" in item:
        center = item["center"]
        return float(center[0]), float(center[1])
    return float(item["center_x_px"]), float(item["center_y_px"])


def _points(item: dict) -> list[tuple[float, float]]:
    return [(float(point[0]), float(point[1])) for point in item.get("corners", [])]


def _pattern_size(item: dict) -> tuple[int, int]:
    if "pattern_size" in item:
        return int(item["pattern_size"][0]), int(item["pattern_size"][1])
    return int(item["pattern_columns"]), int(item["pattern_rows"])


def _pitch(item: dict) -> float:
    points = _points(item)
    cols, rows = _pattern_size(item)
    if len(points) != cols * rows:
        return 0.0
    steps: list[float] = []
    for row in range(rows):
        for col in range(cols - 1):
            steps.append(math.dist(points[row * cols + col], points[row * cols + col + 1]))
    for row in range(rows - 1):
        for col in range(cols):
            steps.append(math.dist(points[row * cols + col], points[(row + 1) * cols + col]))
    if not steps:
        return 0.0
    steps.sort()
    middle = len(steps) // 2
    return steps[middle] if len(steps) % 2 else (steps[middle - 1] + steps[middle]) / 2.0


def _cross(origin: tuple[float, float], first: tuple[float, float], second: tuple[float, float]) -> float:
    return (first[0] - origin[0]) * (second[1] - origin[1]) - (first[1] - origin[1]) * (second[0] - origin[0])


def _convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    ordered = sorted(set(points))
    if len(ordered) <= 2:
        return ordered
    lower: list[tuple[float, float]] = []
    for point in ordered:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(ordered):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def _polygon_area(points: list[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    return abs(sum(
        points[index][0] * points[(index + 1) % len(points)][1]
        - points[(index + 1) % len(points)][0] * points[index][1]
        for index in range(len(points))
    ) / 2.0)


def _rectangle_score(items: list[dict], indices: tuple[int, ...]) -> float | None:
    if len(indices) not in {3, 4}:
        return None
    centers = [_center(items[index]) for index in indices]
    pitches = [_pitch(items[index]) for index in indices]
    pitch_mean = sum(pitches) / len(pitches)
    if pitch_mean <= 0:
        return None
    pitch_cv = math.sqrt(sum((pitch - pitch_mean) ** 2 for pitch in pitches) / len(pitches)) / pitch_mean
    min_x = min(point[0] for point in centers)
    max_x = max(point[0] for point in centers)
    min_y = min(point[1] for point in centers)
    max_y = max(point[1] for point in centers)
    bbox_area = max(1.0, (max_x - min_x) * (max_y - min_y))
    if _polygon_area(_convex_hull(centers)) / bbox_area < (0.10 if len(indices) == 3 else 0.16):
        return None
    scale = max(pitch_mean, 1.0)
    best = float("inf")
    for order in itertools.permutations(range(len(indices))):
        p0, p1, p2 = (centers[index] for index in order[:3])
        edge_a = math.dist(p1, p0)
        edge_b = math.dist(p2, p0)
        if min(edge_a, edge_b) < 4.0 * scale:
            continue
        diagonal = max(edge_a + edge_b, scale)
        if len(indices) == 4:
            p3 = centers[order[3]]
            expected_p3 = (p1[0] + p2[0] - p0[0], p1[1] + p2[1] - p0[1])
            parallelogram_error = math.dist(p3, expected_p3) / diagonal
        else:
            # Three corners of a 2x2 layout should contain one approximately
            # right angle. Perspective permits a generous tolerance here.
            dot = (p1[0] - p0[0]) * (p2[0] - p0[0]) + (p1[1] - p0[1]) * (p2[1] - p0[1])
            parallelogram_error = abs(dot) / max(edge_a * edge_b, 1.0)
        edge_balance = abs(math.log(max(edge_a, 1e-9) / max(edge_b, 1e-9)))
        best = min(best, (3.0 if len(indices) == 4 else 1.8) * parallelogram_error + 0.25 * edge_balance)
    if not math.isfinite(best):
        return None
    return best + pitch_cv


def infer_groups_for_image(detections: list[dict]) -> list[dict]:
    """Return auxiliary groups for the labeled detections in one photo."""
    groups: list[dict] = []
    for spacing, expected in EXPECTED_BOARDS.items():
        eligible = [
            index for index, item in enumerate(detections)
            if str(item.get("choice", "")) == spacing
        ]
        minimum_for_hint = 3 if expected == 4 else expected
        if len(eligible) < minimum_for_hint:
            continue
        candidates: list[tuple[float, tuple[int, ...]]] = []
        if expected == 4:
            # Prefer a complete 2x2 group, but retain a geometrically strong
            # three-board group to expose a likely missing board in the UI.
            combo_sizes = (4, 3) if len(eligible) >= 4 else (3,)
            for combo_size in combo_sizes:
                for combo in itertools.combinations(eligible, combo_size):
                    score = _rectangle_score(detections, combo)
                    if score is not None:
                        candidates.append((score + (0.0 if combo_size == 4 else 0.18), combo))
        else:
            for first, second in itertools.combinations(eligible, 2):
                first_pitch = _pitch(detections[first])
                second_pitch = _pitch(detections[second])
                pitch_mean = max((first_pitch + second_pitch) / 2.0, 1.0)
                distance = math.dist(_center(detections[first]), _center(detections[second]))
                ratio = distance / pitch_mean
                if ratio <= 36.0:
                    pitch_error = abs(first_pitch - second_pitch) / pitch_mean
                    candidates.append((ratio / 36.0 + pitch_error, (first, second)))
        candidates.sort(key=lambda entry: (entry[0], tuple(detections[index]["board_id"] for index in entry[1])))
        used: set[int] = set()
        for score, indices in candidates:
            if any(index in used for index in indices):
                continue
            if expected == 4 and score > 1.0:
                continue
            used.update(indices)
            groups.append({
                "grid_spacing_mm": int(spacing),
                "expected_boards": expected,
                "board_ids": [detections[index]["board_id"] for index in indices],
                "score": float(score),
                "confidence": float(max(0.0, min(1.0, math.exp(-score)))),
            })
    groups.sort(key=lambda group: (group["grid_spacing_mm"], group["board_ids"]))
    return groups


def update_group_metadata(manifest: dict, choices: dict[str, str]) -> None:
    """Update per-board and per-image group fields in a manifest in place."""
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

