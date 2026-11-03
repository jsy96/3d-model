"""Find multiple complete or cropped checkerboards without annotation seeds.

Dark-cell components propose regions; OpenCV recovers a local corner lattice.
Only image-supported corners are returned. A cropped board keeps its local grid
coordinates: its unknown origin on the printed sheet is not extrapolated.
"""

from __future__ import annotations

from pathlib import Path
import time

import cv2
import numpy as np


FULL_PATTERN = (8, 11)
MAX_REGIONS = 100


def _project(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(np.asarray(points, np.float32).reshape(1, -1, 2), matrix).reshape(-1, 2)


def _clip_polygon(points: np.ndarray, width: int, height: int) -> np.ndarray:
    """Intersect with the image; clamping vertices would change slanted edges."""
    frame = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], np.float32)
    _, clipped = cv2.intersectConvexConvex(np.asarray(points, np.float32), frame)
    return clipped.reshape(-1, 2) if clipped is not None else np.empty((0, 2), np.float32)


def _regions(gray: np.ndarray) -> tuple[list[dict], bool]:
    scale = min(1.0, 1800 / max(gray.shape))
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    _, binary = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # Separate black cells that touch diagonally, including blurred junctions.
    binary = cv2.erode(binary, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cells = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 30 or area > small.size * 0.015:
            continue
        quad = cv2.approxPolyDP(contour, 0.045 * cv2.arcLength(contour, True), True).reshape(-1, 2).astype(np.float32)
        if len(quad) != 4 or not cv2.isContourConvex(quad) or cv2.contourArea(quad) < 0.8 * area:
            continue
        edges = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)
        if min(edges) < 4 or max(edges) / min(edges) > 8:
            continue
        cells.append(quad)
    # Bound work on images with dense unrelated texture.
    limited = len(cells) > 4000
    cells = sorted(cells, key=cv2.contourArea, reverse=True)[:4000]
    if not cells:
        return [], limited
    quads = np.asarray(cells)
    centers = quads.mean(axis=1)
    radii = np.sqrt([cv2.contourArea(quad) for quad in quads])
    parent = list(range(len(cells)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for first in range(len(cells)):
        distances = np.linalg.norm(centers[first + 1:] - centers[first], axis=1)
        neighbors = np.flatnonzero(distances < 2.5 * radii[first]) + first + 1
        for second in neighbors:
            if not 0.4 < radii[second] / radii[first] < 2.5:
                continue
            gap = np.linalg.norm(quads[first][:, None] - quads[second][None, :], axis=2).min()
            if gap < max(5.0, 0.42 * min(radii[first], radii[second])):
                parent[find(int(second))] = find(first)
    groups: dict[int, list[int]] = {}
    for index in range(len(cells)):
        groups.setdefault(find(index), []).append(index)
    regions = []
    for indices in groups.values():
        if len(indices) < 8:
            continue
        points = quads[indices].reshape(-1, 2)
        pad = float(np.median(radii[indices])) * 1.5
        low = np.maximum(points.min(axis=0) - pad, 0) / scale
        high = np.minimum(points.max(axis=0) + pad, [small.shape[1], small.shape[0]]) / scale
        regions.append({
            "rect": tuple(np.r_[low, high].astype(int)),
            "cell_centers": (centers[indices] + 0.5) / scale - 0.5,
            "cell_quads": (quads[indices] + 0.5) / scale - 0.5,
            "cell_count": len(indices),
        })
    regions.sort(key=lambda item: item["cell_count"], reverse=True)
    return regions[:MAX_REGIONS], limited or len(regions) > MAX_REGIONS


def _cell_samples(gray: np.ndarray, matrix: np.ndarray, cells: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.array([[-0.12, -0.12], [-0.12, 0.12], [0, 0], [0.12, -0.12], [0.12, 0.12]], np.float32)
    xy = _project((cells[:, None] + 0.5 + offsets[None]).reshape(-1, 2), matrix).reshape(-1, 5, 2)
    finite = np.isfinite(xy).all(axis=(1, 2))
    xy = np.rint(np.nan_to_num(xy, nan=-1, posinf=-1, neginf=-1)).astype(np.int32)
    valid = finite & (xy[:, :, 0] >= 0).all(axis=1) & (xy[:, :, 0] < gray.shape[1]).all(axis=1)
    valid &= (xy[:, :, 1] >= 0).all(axis=1) & (xy[:, :, 1] < gray.shape[0]).all(axis=1)
    values = np.zeros(len(cells), np.float32)
    values[valid] = np.median(gray[xy[valid, :, 1], xy[valid, :, 0]], axis=1)
    return values, valid


def _photometric_score(gray: np.ndarray, matrix: np.ndarray, low: np.ndarray, high: np.ndarray) -> tuple[float, float]:
    cells = np.array([(x, y) for y in range(int(low[1]), int(high[1])) for x in range(int(low[0]), int(high[0]))], np.float32)
    if len(cells) < 12:
        return 0.0, 0.0
    values, valid = _cell_samples(gray, matrix, cells)
    parity = ((cells[:, 0] + cells[:, 1]).astype(int) % 2) == 0
    first, second = values[valid & parity], values[valid & ~parity]
    if min(len(first), len(second)) < 4:
        return 0.0, 0.0
    a, b = float(np.median(first)), float(np.median(second))
    middle = (a + b) / 2
    consistency = (np.mean(first < middle) + np.mean(second > middle)) / 2 if a < b else (np.mean(first > middle) + np.mean(second < middle)) / 2
    return float(consistency), abs(a - b)


def _boundary_score(gray: np.ndarray, matrix: np.ndarray, low: np.ndarray, high: np.ndarray, contrast: float) -> float:
    """Prefer a board extent whose visible outside cells stop alternating."""
    cells: list[tuple[int, int]] = []
    for y in range(int(low[1]), int(high[1])):
        cells.extend(((int(low[0]) - 1, y), (int(high[0]), y)))
    for x in range(int(low[0]), int(high[0])):
        cells.extend(((x, int(low[1]) - 1), (x, int(high[1]))))
    if not cells:
        return 0.0
    values, valid = _cell_samples(gray, matrix, np.asarray(cells, np.float32))
    groups: dict[tuple[str, int], list[float]] = {}
    for cell, value, is_valid in zip(cells, values, valid):
        if not is_valid:
            continue
        x, y = cell
        if x < low[0]:
            groups.setdefault(("left", y), []).append(float(value))
        elif x >= high[0]:
            groups.setdefault(("right", y), []).append(float(value))
        elif y < low[1]:
            groups.setdefault(("top", x), []).append(float(value))
        else:
            groups.setdefault(("bottom", x), []).append(float(value))
    scores = []
    for side in ("left", "right", "top", "bottom"):
        stripe = [(key[1], samples[0]) for key, samples in groups.items() if key[0] == side]
        if len(stripe) < 4:
            continue
        stripe.sort()
        even = [value for index, value in stripe if index % 2 == 0]
        odd = [value for index, value in stripe if index % 2 != 0]
        if not even or not odd:
            continue
        alternation = abs(float(np.median(even)) - float(np.median(odd))) / max(contrast, 1.0)
        scores.append(1.0 - min(1.0, alternation))
    return float(np.mean(scores)) if scores else 0.0


def _full_extent(gray: np.ndarray, matrix: np.ndarray, cols: int, rows: int, region: dict) -> tuple[np.ndarray, np.ndarray, float, float] | None:
    """Fit a detected sub-grid into the known 9 x 12 square board plane."""
    inverse = np.linalg.inv(matrix)
    local_centers = _project(region["cell_centers"], inverse)
    cell_indices = np.rint(local_centers - 0.5).astype(int)
    aligned = np.linalg.norm(local_centers - cell_indices - 0.5, axis=1) < 0.32
    if np.count_nonzero(aligned) < max(8, int(region["cell_count"] * 0.5)):
        return None

    options = []
    for width, height in ((9, 12), (12, 9)):
        if cols > width - 1 or rows > height - 1:
            continue
        for x0 in range(cols - width, 0):
            for y0 in range(rows - height, 0):
                low = np.array([x0, y0], np.float32)
                high = low + [width, height]
                score, contrast = _photometric_score(gray, matrix, low, high)
                if score < 0.83 or contrast < 28:
                    continue
                inside = (cell_indices[:, 0] >= x0) & (cell_indices[:, 0] < x0 + width)
                inside &= (cell_indices[:, 1] >= y0) & (cell_indices[:, 1] < y0 + height)
                support = float(np.mean(inside[aligned]))
                if support < 0.9:
                    continue
                boundary = _boundary_score(gray, matrix, low, high, contrast)
                rank = score + 0.05 * boundary + 0.01 * min(contrast / 200, 1)
                options.append((rank, score, contrast, low, high))
    if not options:
        return None
    _, score, contrast, low, high = max(options, key=lambda item: item[0])
    return low, high, score, contrast


def _saddles(gray: np.ndarray, points: np.ndarray, matrix: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Require the two alternating diagonal pairs around each measured corner."""
    inverse = np.linalg.inv(matrix)
    local = _project(points, inverse)
    offsets = np.array([[-0.22, -0.22], [0.22, -0.22], [0.22, 0.22], [-0.22, 0.22]], np.float32)
    xy = _project((local[:, None] + offsets[None]).reshape(-1, 2), matrix).reshape(-1, 4, 2)
    xy = np.rint(xy).astype(np.int32)
    inside = (xy[:, :, 0] >= 0).all(axis=1) & (xy[:, :, 0] < gray.shape[1]).all(axis=1)
    inside &= (xy[:, :, 1] >= 0).all(axis=1) & (xy[:, :, 1] < gray.shape[0]).all(axis=1)
    values = np.zeros((len(points), 4), np.float32)
    values[inside] = gray[xy[inside, :, 1], xy[inside, :, 0]]
    contrast = np.abs((values[:, 0] + values[:, 2] - values[:, 1] - values[:, 3]) / 2)
    mismatch = (np.abs(values[:, 0] - values[:, 2]) + np.abs(values[:, 1] - values[:, 3])) / 2
    displacement = np.linalg.norm(local - grid, axis=1)
    return inside & (contrast >= 25) & (mismatch < contrast * 0.85) & (displacement < 0.4)


def _find_lattice(small: np.ndarray) -> tuple[bool, np.ndarray | None, int, int]:
    flags = cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCornersSB(small, FULL_PATTERN, flags)
    if found and corners is not None:
        return True, corners, FULL_PATTERN[0], FULL_PATTERN[1]

    found, corners, meta = cv2.findChessboardCornersSBWithMeta(
        small, (4, 4), flags | cv2.CALIB_CB_LARGER
    )
    if found and corners is not None and meta is not None:
        rows, cols = meta.shape
        if min(cols, rows) >= 3 and max(cols, rows) <= 11:
            return False, corners, cols, rows

    # A low-contrast or clipped border can make the larger-pattern search fail
    # even when OpenCV can still fit a smaller contiguous corner patch.
    patterns = {
        (cols, rows)
        for cols in range(3, 9)
        for rows in range(3, 12)
        if cols * rows >= 16
    }
    patterns |= {(rows, cols) for cols, rows in tuple(patterns)}
    variants = [small]
    if min(small.shape) >= 24:
        variants.append(cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(small))
    for variant in variants:
        for cols, rows in sorted(patterns, key=lambda size: size[0] * size[1], reverse=True):
            found, corners = cv2.findChessboardCornersSB(variant, (cols, rows), flags)
            if found and corners is not None:
                return False, corners, cols, rows
    return False, None, 0, 0


def _lattice_hypotheses(small: np.ndarray):
    """Yield the first OpenCV result, then alternate grid sizes as needed."""
    first = _find_lattice(small)
    yield first

    tried = {(first[2], first[3])} if first[1] is not None else set()
    patterns = {
        (cols, rows)
        for cols in range(3, 9)
        for rows in range(3, 12)
        if cols * rows >= 16
    }
    patterns |= {(rows, cols) for cols, rows in tuple(patterns)}
    flags = cv2.CALIB_CB_NORMALIZE_IMAGE
    variants = [small]
    if min(small.shape) >= 24:
        variants.append(cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(small))
    for variant in variants:
        for cols, rows in sorted(patterns, key=lambda size: size[0] * size[1], reverse=True):
            if (cols, rows) in tried:
                continue
            tried.add((cols, rows))
            found, corners = cv2.findChessboardCornersSB(variant, (cols, rows), flags)
            if found and corners is not None:
                yield False, corners, cols, rows


def _candidate_for_lattice(gray: np.ndarray, region: dict, lattice: tuple[bool, np.ndarray | None, int, int]) -> dict | None:
    x0, y0, x1, y1 = region["rect"]
    crop = gray[y0:y1, x0:x1]
    if min(crop.shape) < 35:
        return None
    scale = min(1.0, 1000 / max(crop.shape))
    small = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    complete, corners, cols, rows = lattice
    if corners is None or min(cols, rows) < 3 or max(cols, rows) > 11 or min(cols, rows) > 8:
        return None
    points = (corners.reshape(-1, 2) + 0.5) / scale - 0.5 + np.array([x0, y0], np.float32)
    grid = np.array([(x, y) for y in range(rows) for x in range(cols)], np.float32)
    if len(points) != len(grid):
        return None
    # Rectification can place a proposed corner beyond the original image.
    # Reject that lattice before OpenCV's subpixel refinement asserts; later
    # hypotheses may still explain the visible portion of the same board.
    if not np.isfinite(points).all() or np.any(points < 0) or np.any(points > [gray.shape[1] - 1, gray.shape[0] - 1]):
        return None
    lattice = points.reshape(rows, cols, 2)
    pitch = float(np.median(np.r_[np.linalg.norm(np.diff(lattice, axis=1), axis=2).ravel(), np.linalg.norm(np.diff(lattice, axis=0), axis=2).ravel()]))
    if pitch < 3:
        return None
    window = max(2, min(12, round(pitch * 0.2)))
    cv2.cornerSubPix(gray, points.reshape(-1, 1, 2), (window, window), (-1, -1), (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 25, 0.01))
    matrix, _ = cv2.findHomography(grid, points, 0)
    if matrix is None or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-8:
        return None
    residual = float(np.median(np.linalg.norm(_project(grid, matrix) - points, axis=1))) / pitch
    if residual > 0.18:
        return None
    low, high = np.array([-1, -1]), np.array([cols, rows])
    consistency, contrast = _photometric_score(gray, matrix, low, high)
    if consistency < 0.83 or contrast < 28:
        return None
    if not complete:
        # Recover the visible extent beyond OpenCV's rectangular sub-grid using
        # the original dark-cell component. This also handles one missing corner.
        local_centers = _project(region["cell_centers"], np.linalg.inv(matrix))
        cells = np.rint(local_centers - 0.5).astype(int)
        valid = np.linalg.norm(local_centers - cells - 0.5, axis=1) < 0.32
        # A diagonal or skipped-cell lattice can also produce periodic samples.
        # Its axes must agree with the actual square edges, and its cell centers
        # must explain the proposing component, before it is accepted.
        basis_x = _project(local_centers + [0.5, 0], matrix) - _project(local_centers - [0.5, 0], matrix)
        basis_y = _project(local_centers + [0, 0.5], matrix) - _project(local_centers - [0, 0.5], matrix)
        basis = np.stack([basis_x, basis_y], axis=1)
        basis /= np.maximum(np.linalg.norm(basis, axis=2, keepdims=True), 1e-6)
        edges = np.roll(region["cell_quads"], -1, axis=1) - region["cell_quads"]
        edges /= np.maximum(np.linalg.norm(edges, axis=2, keepdims=True), 1e-6)
        alignment = np.abs(np.einsum("nei,nbi->neb", edges, basis)).max(axis=2).mean(axis=1)
        supported = valid & (alignment >= 0.9)
        if np.count_nonzero(supported) < 8:
            return None
        # Cell components from adjacent boards can join into one region. Use
        # only the components explained by this lattice when checking extent.
        support_region = {
            **region,
            "cell_centers": region["cell_centers"][supported],
            "cell_quads": region["cell_quads"][supported],
            "cell_count": int(np.count_nonzero(supported)),
        }
        full_extent = _full_extent(gray, matrix, cols, rows, support_region)
        if full_extent is not None:
            low, high, consistency, contrast = full_extent
        elif np.count_nonzero(supported) >= 8:
            extent_low, extent_high = cells[supported].min(axis=0), cells[supported].max(axis=0) + 1
            extent_low, extent_high = np.minimum(low, extent_low), np.maximum(high, extent_high)
            span = extent_high - extent_low
            if min(span) <= 9 and max(span) <= 12:
                score, difference = _photometric_score(gray, matrix, extent_low, extent_high)
                if score >= 0.83 and difference >= 28:
                    low, high, consistency, contrast = extent_low, extent_high, score, difference
        # Try additional interior corners, but save each only after a local
        # saddle check. Projected positions alone are never observations.
        extended_grid = np.array([(x, y) for y in range(int(low[1] + 1), int(high[1])) for x in range(int(low[0] + 1), int(high[0]))], np.float32)
        extended_points = _project(extended_grid, matrix)
        safe = (extended_points[:, 0] > window) & (extended_points[:, 0] < gray.shape[1] - window - 1)
        safe &= (extended_points[:, 1] > window) & (extended_points[:, 1] < gray.shape[0] - window - 1)
        extended_grid, extended_points = extended_grid[safe], extended_points[safe]
        if len(extended_points):
            cv2.cornerSubPix(gray, extended_points.reshape(-1, 1, 2), (window, window), (-1, -1), (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 25, 0.01))
            original = {(int(x), int(y)): point for (x, y), point in zip(grid, points)}
            for index, coordinate in enumerate(extended_grid):
                if tuple(map(int, coordinate)) in original:
                    extended_points[index] = original[tuple(map(int, coordinate))]
            points, grid = extended_points, extended_grid
    valid = _saddles(gray, points, matrix, grid)
    points, grid = points[valid], grid[valid]
    if len(points) < 12 or len(np.unique(grid[:, 0])) < 3 or len(np.unique(grid[:, 1])) < 3:
        return None
    quad = _project(np.array([low, [high[0], low[1]], high, [low[0], high[1]]], np.float32), matrix)
    if not np.isfinite(quad).all() or not cv2.isContourConvex(quad):
        return None
    height, width = gray.shape
    polygon = _clip_polygon(quad, width, height)
    if len(polygon) < 3 or cv2.contourArea(polygon) < pitch * pitch * 6:
        return None
    clipped = bool(np.any(quad < 0) or np.any(quad[:, 0] >= width) or np.any(quad[:, 1] >= height))
    span = tuple(map(int, high - low))
    partial = clipped
    confidence = max(0, min(1, consistency - residual + min(contrast / 200, 1) * 0.05))
    return {
        "polygon": polygon,
        "corners": points,
        "grid_indices": (grid - low - 1).astype(int),
        "detected_squares": list(span),
        "partial": partial,
        "needs_review": partial or len(points) < 88 or confidence < 0.9,
        "confidence": round(float(confidence), 3),
        "quality": len(points) + confidence,
    }


def _rectified_lattice(gray: np.ndarray, region: dict) -> tuple[bool, np.ndarray, int, int] | None:
    """Make one search with the observed cell axes closer to orthogonal.

    Strongly slanted cells can make SB return a diagonal or skipped-cell
    lattice. Rectification only proposes corners; validation and subpixel
    refinement still use the original image.
    """
    x0, y0, x1, y1 = region["rect"]
    quad = max(region["cell_quads"], key=cv2.contourArea)
    basis = np.stack([
        (quad[1] - quad[0] + quad[2] - quad[3]) / 2,
        (quad[3] - quad[0] + quad[2] - quad[1]) / 2,
    ], axis=1)
    if abs(np.linalg.det(basis)) < 1e-6:
        return None
    matrix = np.eye(3)
    matrix[:2, :2] = 40 * np.linalg.inv(basis)
    bounds = _project(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32), matrix)
    low, high = np.floor(bounds.min(axis=0)), np.ceil(bounds.max(axis=0))
    # Keep the additional search within the same size budget as the raw crop.
    resize = min(1.0, 1000 / max(high - low))
    matrix[:2, :2] *= resize
    matrix[:2, 2] = -low * resize
    size = tuple(np.ceil((high - low) * resize).astype(int))
    rectified = cv2.warpPerspective(gray, matrix, size, borderValue=float(np.median(gray[y0:y1, x0:x1])))
    found, corners, meta = cv2.findChessboardCornersSBWithMeta(
        rectified, (4, 4), cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_LARGER
    )
    if not found or corners is None or meta is None:
        return None
    rows, cols = meta.shape
    # _candidate_for_lattice expects coordinates on the resized original crop.
    scale = min(1.0, 1000 / max(y1 - y0, x1 - x0))
    original = _project(corners, np.linalg.inv(matrix)) - np.array([x0, y0], np.float32)
    corners = ((original + 0.5) * scale - 0.5).astype(np.float32).reshape(-1, 1, 2)
    return False, corners, cols, rows


def _candidate(gray: np.ndarray, region: dict) -> dict | None:
    """Try alternate lattice hypotheses when the first one fails validation."""
    x0, y0, x1, y1 = region["rect"]
    crop = gray[y0:y1, x0:x1]
    if min(crop.shape) < 35:
        return None
    scale = min(1.0, 1000 / max(crop.shape))
    small = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    raw_lattices = _lattice_hypotheses(small)

    def hypotheses():
        yield next(raw_lattices)
        try:
            rectified = _rectified_lattice(gray, region)
        except (cv2.error, np.linalg.LinAlgError):
            rectified = None
        if rectified is not None:
            yield rectified
        yield from raw_lattices

    best = None
    for lattice in hypotheses():
        try:
            candidate = _candidate_for_lattice(gray, region, lattice)
        except (cv2.error, np.linalg.LinAlgError):
            continue
        if candidate is None:
            continue
        if best is None or candidate["quality"] > best["quality"]:
            best = candidate
        # 80+ image-supported corners with strong checker contrast is already
        # a reliable near-complete board; avoid scanning smaller patterns.
        if candidate["quality"] >= 80:
            return candidate
    return best


def _overlap(first: np.ndarray, second: np.ndarray) -> float:
    area, _ = cv2.intersectConvexConvex(first.astype(np.float32), second.astype(np.float32))
    return float(area) / max(1.0, min(cv2.contourArea(first), cv2.contourArea(second)))


def find_boards(gray: np.ndarray) -> dict:
    started = time.perf_counter()
    regions, limited = _regions(gray)
    candidates = []
    for region in regions:
        try:
            candidate = _candidate(gray, region)
        except (cv2.error, np.linalg.LinAlgError):
            candidate = None
        if candidate is not None:
            candidates.append(candidate)
    candidates.sort(key=lambda item: item["quality"], reverse=True)
    kept = []
    for candidate in candidates:
        if any(_overlap(candidate["polygon"], old["polygon"]) > 0.45 for old in kept):
            continue
        kept.append(candidate)
    kept.sort(key=lambda item: tuple(item["polygon"].mean(axis=0)[::-1]))
    height, width = gray.shape
    dimensions = np.array([width, height], np.float32)
    boxes = []
    for item in kept:
        normalized = np.clip(item["polygon"] / dimensions, 0, 1)
        boxes.append({
            "points": [{"x": float(x), "y": float(y)} for x, y in normalized],
            "detected_squares": item["detected_squares"],
            "partial": bool(item["partial"]),
            "needs_review": bool(item["needs_review"]),
            "confidence": item["confidence"],
            "source": "automatic",
            "corners": (item["corners"] / dimensions).astype(float).tolist(),
            "grid_indices": item["grid_indices"].tolist(),
        })
    return {"boxes": boxes, "image_width": width, "image_height": height, "elapsed_seconds": round(time.perf_counter() - started, 2), "limited": limited}


def detect_image(image_path: Path) -> dict:
    # Match the orientation displayed by browsers, as the existing labeler does.
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError("图片无法读取。")
    return find_boards(gray)
