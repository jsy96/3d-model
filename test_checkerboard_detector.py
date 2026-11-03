"""Behavior checks with generated images and a read-only production regression case."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.request import Request, urlopen

import cv2
import numpy as np

import calibration_labeler as labeler
from checkerboard_detector import find_boards


def checkerboard() -> np.ndarray:
    image = np.full((600, 480), 245, np.uint8)
    for row in range(12):
        for col in range(9):
            if (row + col) % 2 == 0:
                image[60 + row * 40:100 + row * 40, 60 + col * 40:100 + col * 40] = 15
    return image


class DetectionTests(unittest.TestCase):
    def assert_coordinates(self, box: dict) -> None:
        self.assertEqual(len(box["corners"]), len(box["grid_indices"]))
        self.assertEqual(len(box["grid_indices"]), len(set(map(tuple, box["grid_indices"]))))
        self.assertTrue(all(0 <= value <= 1 for point in box["corners"] for value in point))
        self.assertTrue(all(0 <= point[axis] <= 1 for point in box["points"] for axis in ("x", "y")))

    def test_complete_and_multiple_boards(self) -> None:
        image = checkerboard()
        two = np.concatenate([image, np.full((600, 50), 245, np.uint8), image], axis=1)
        for source, expected in ((image, 1), (two, 2)):
            with self.subTest(expected=expected):
                boxes = find_boards(source)["boxes"]
                self.assertEqual(len(boxes), expected)
                for box in boxes:
                    self.assertFalse(box["partial"])
                    self.assertEqual(len(box["corners"]), 88)
                    self.assert_coordinates(box)

    def test_low_contrast_and_blurred_board(self) -> None:
        image = checkerboard()
        low_contrast = np.clip(image.astype(np.float32) * 0.19 + 108, 0, 255).astype(np.uint8)
        for name, source in (
            ("low contrast", low_contrast),
            ("blurred", cv2.GaussianBlur(image, (5, 5), 1.2)),
            ("low contrast and blurred", cv2.GaussianBlur(low_contrast, (5, 5), 1.2)),
        ):
            with self.subTest(name=name):
                boxes = find_boards(source)["boxes"]
                self.assertEqual(len(boxes), 1)
                self.assertEqual(set(boxes[0]["detected_squares"]), {9, 12})
                self.assertEqual(len(boxes[0]["corners"]), 88)
                self.assertFalse(boxes[0]["partial"])

    def test_cropped_edge_corner_and_missing_corner(self) -> None:
        image = checkerboard()
        notch = image.copy()
        cv2.fillConvexPoly(notch, np.array([[60, 60], [145, 60], [60, 145]]), 245)
        cases = (
            ("edge", image[105:], True),
            ("corner", image[105:, 105:], True),
            ("missing corner", notch, False),
        )
        for name, source, clipped in cases:
            with self.subTest(name=name):
                boxes = find_boards(source)["boxes"]
                self.assertEqual(len(boxes), 1)
                # The detector infers the complete 9 x 12 board extent from
                # the visible plane; only contact with the image edge makes
                # the board itself partial.
                self.assertEqual(set(boxes[0]["detected_squares"]), {9, 12})
                self.assertEqual(boxes[0]["partial"], clipped)
                self.assertTrue(boxes[0]["needs_review"])
                self.assertGreaterEqual(len(boxes[0]["corners"]), 12)
                self.assertLess(len(boxes[0]["corners"]), 88)
                self.assert_coordinates(boxes[0])

    def test_slanted_crop_uses_polygon_intersection(self) -> None:
        image = checkerboard()
        source = np.float32([[0, 0], [479, 0], [479, 599], [0, 599]])
        target = np.float32([[140, 15], [480, 120], [590, 630], [0, 550]])
        view = cv2.warpPerspective(image, cv2.getPerspectiveTransform(source, target), (610, 650), borderValue=245)
        box = find_boards(view[100:, 80:])["boxes"][0]
        self.assertTrue(box["partial"])
        self.assertEqual(len(box["points"]), 5)
        self.assert_coordinates(box)
        # The persisted representation retains the measured corner/index pairs.
        self.assertEqual(labeler.validate_boxes([box]), [box])

    def test_plain_image_has_no_candidates(self) -> None:
        self.assertEqual(find_boards(np.full((600, 480), 200, np.uint8))["boxes"], [])

    def test_merged_regions_and_edge_board_refine(self) -> None:
        root = Path(__file__).resolve().parent
        image_path = root / "生产项目/data/images/微信图片_20261102105444_200_5.jpg"
        annotations_path = root / "生产项目/data/annotations/board_boxes.json"
        if not image_path.is_file() or not annotations_path.is_file():
            self.skipTest("Production image fixture is not present")

        image_name = image_path.name
        boxes = json.loads(annotations_path.read_text(encoding="utf-8"))["boxes"][image_name]
        legacy_box = boxes[19]
        seed = legacy_box["seed_points"]

        result = labeler.refine_checkerboard(image_path, seed)
        self.assertGreaterEqual(len(result["corners"]), 80)
        self.assertEqual(len(result["corners"]), len(result["grid_indices"]))
        self.assertTrue(result["partial"], "The detected board reaches the image's bottom edge")
        center = tuple(np.mean([[point["x"], point["y"]] for point in seed], axis=0))
        polygon = np.array([[point["x"], point["y"]] for point in result["points"]], np.float32)
        self.assertGreaterEqual(cv2.pointPolygonTest(polygon, center, False), 0)

    def test_partial_production_board_recovers_full_extent(self) -> None:
        root = Path(__file__).resolve().parent
        image_path = root / "生产项目/data/images/微信图片_20261102105444_201_5.jpg"
        annotations_path = root / "生产项目/data/annotations/board_boxes.json"
        if not image_path.is_file() or not annotations_path.is_file():
            self.skipTest("Production image fixture is not present")

        annotations = json.loads(annotations_path.read_text(encoding="utf-8"))["boxes"]
        image_names = sorted(annotations)
        self.assertEqual(image_names[7], image_path.name)
        legacy_box = annotations[image_path.name][23]
        seed = legacy_box["seed_points"]

        result = labeler.refine_checkerboard(image_path, seed)

        self.assertEqual(set(result["detected_squares"]), {9, 12})
        self.assertTrue(result["partial"])
        self.assertGreaterEqual(len(result["corners"]), 80)
        self.assert_coordinates(result)
        center = tuple(np.mean([[point["x"], point["y"]] for point in seed], axis=0))
        polygon = np.array([[point["x"], point["y"]] for point in result["points"]], np.float32)
        self.assertGreaterEqual(cv2.pointPolygonTest(polygon, center, False), 0)
        # Dimensions alone also accept diagonal/shifted grids. Check the real
        # visible perimeter, including the intersection with the left edge.
        expected = np.array([[0, 1157], [0, 713], [22, 632], [482, 952], [387, 1349]], np.float32)
        polygon = polygon * [4032, 3024]
        intersection, _ = cv2.intersectConvexConvex(polygon.astype(np.float32), expected)
        union = cv2.contourArea(polygon.astype(np.float32)) + cv2.contourArea(expected) - intersection
        self.assertGreater(intersection / union, 0.97)

    def test_api_preview_save_export_and_partial_refine(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cv2.imwrite(str(root / "cropped.png"), checkerboard()[105:, 105:])
            with patch.multiple(labeler, IMAGE_DIR=root, ANNOTATIONS_DIR=root, BOXES_PATH=root / "boxes.json", DETECTION_CACHE={}):
                with patch.object(labeler.Handler, "log_message", lambda *args: None):
                    server = labeler.ThreadingHTTPServer(("127.0.0.1", 0), labeler.Handler)
                    worker = threading.Thread(target=server.serve_forever, daemon=True)
                    worker.start()
                    base = f"http://127.0.0.1:{server.server_port}"

                    def request(route: str, payload: dict | None = None) -> dict:
                        data = None if payload is None else json.dumps(payload).encode()
                        with urlopen(Request(base + route, data=data, headers={"Content-Type": "application/json"}), timeout=15) as response:
                            return json.load(response)

                    try:
                        preview = request("/api/detect", {"name": "cropped.png"})
                        self.assertEqual(len(preview["boxes"]), 1)
                        self.assertFalse(labeler.BOXES_PATH.exists(), "Preview must not persist candidates")
                        box = preview["boxes"][0]
                        refined = request("/api/refine", {"name": "cropped.png", "points": box["points"]})
                        self.assertTrue(refined["partial"])
                        self.assertFalse(labeler.BOXES_PATH.exists())
                        old = {"x": 0.01, "y": 0.01, "w": 0.05, "h": 0.05}
                        request("/api/boxes", {"name": "cropped.png", "boxes": [old, box], "image_width": preview["image_width"], "image_height": preview["image_height"]})
                        exported = request("/api/export.json")
                        self.assertEqual(exported["boxes"]["cropped.png"], [old, box])
                        self.assertEqual(request("/api/data")["boxes"], exported["boxes"])
                    finally:
                        server.shutdown()
                        server.server_close()
                        worker.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
