# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for ``rfdetr.utilities.video.predict_video``."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import supervision as sv

import rfdetr
import rfdetr.utilities.video as video
from rfdetr import RFDETRNano, RFDETRSegNano
from rfdetr.utilities.video import _BoxPropagator, predict_video

cv2 = pytest.importorskip("cv2")

#: Side of each textured square in pixels.
_SQUARE_SIZE = 40
#: Class id and confidence the recording detector reports for its first square.
_FIRST_CLASS_ID, _FIRST_CONFIDENCE = 7, 0.9


def moving_squares_scene(
    num_frames: int,
    steps: tuple[tuple[int, int], ...] = ((2, 0),),
    height: int = 128,
    width: int = 192,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Build RGB frames of textured squares sliding over a textured background.

    Args:
        num_frames: Number of frames.
        steps: Per square, the ``(dx, dy)`` pixels it moves between two frames.
        height: Frame height in pixels.
        width: Frame width in pixels.

    Returns:
        The frames and the true ``xyxy`` boxes, shape ``(num_frames, len(steps), 4)``.

    Examples:
        >>> frames, boxes = moving_squares_scene(3, steps=((2, 0), (0, -3)))
        >>> len(frames), boxes.shape, frames[0].shape
        (3, (3, 2, 4), (128, 192, 3))
        >>> float(boxes[1, 0, 0] - boxes[0, 0, 0]), float(boxes[1, 1, 1] - boxes[0, 1, 1])
        (2.0, -3.0)
    """
    rng = np.random.default_rng(0)
    background = cv2.GaussianBlur(rng.integers(0, 255, (height, width, 3), dtype=np.uint8), (5, 5), 0)
    squares = [
        cv2.GaussianBlur(rng.integers(0, 255, (_SQUARE_SIZE, _SQUARE_SIZE, 3), dtype=np.uint8), (5, 5), 0)
        for _ in steps
    ]
    frames, boxes = [], []
    for index in range(num_frames):
        frame = background.copy()
        frame_boxes = []
        for number, ((dx, dy), square) in enumerate(zip(steps, squares)):
            left = 20 + 90 * number + dx * index
            top = 40 + dy * index
            frame[top : top + _SQUARE_SIZE, left : left + _SQUARE_SIZE] = square
            frame_boxes.append([left, top, left + _SQUARE_SIZE, top + _SQUARE_SIZE])
        frames.append(frame)
        boxes.append(frame_boxes)
    return frames, np.asarray(boxes, dtype=np.float64)


class RecordingDetector:
    """Stands in for an RF-DETR model: returns the true boxes of a frame and records which frames it was given.

    Examples:
        >>> frames, boxes = moving_squares_scene(2)
        >>> detector = RecordingDetector(frames, boxes)
        >>> len(detector.predict(frames[1], threshold=0.5, include_source_image=False))
        1
        >>> detector.seen
        [1]
    """

    def __init__(self, frames: list[np.ndarray], boxes: np.ndarray) -> None:
        self._index_by_id = {id(frame): index for index, frame in enumerate(frames)}
        self._boxes = boxes
        self.seen: list[int] = []
        self.kwargs: list[dict[str, Any]] = []

    def predict(self, image: np.ndarray, threshold: float = 0.5, **kwargs: Any) -> sv.Detections:
        """Return the squares' true boxes for ``image``, with ``class_name`` data like the real model."""
        index = self._index_by_id[id(image)]
        self.seen.append(index)
        self.kwargs.append({"threshold": threshold, **kwargs})
        boxes = self._boxes[index]
        return sv.Detections(
            xyxy=boxes.copy(),
            confidence=np.full(len(boxes), _FIRST_CONFIDENCE - 0.01 * index),
            class_id=np.arange(len(boxes)) + _FIRST_CLASS_ID,
            data={"class_name": np.array([f"square-{number}" for number in range(len(boxes))])},
        )


class FixedDetector:
    """Stands in for a model whose ``predict`` always returns ``result``.

    Examples:
        >>> FixedDetector(sv.Detections.empty()).predict(np.zeros((2, 2, 3), np.uint8)) is not None
        True
    """

    def __init__(self, result: object) -> None:
        self._result = result

    def predict(self, image: np.ndarray, threshold: float = 0.5, **kwargs: Any) -> Any:
        """Return the fixed result."""
        return self._result


def one_buffer(frames: list[np.ndarray]) -> Iterator[np.ndarray]:
    """Yield every frame through the same array, like a reader that decodes into one preallocated buffer.

    Examples:
        >>> frames = [np.zeros((2, 2, 3), np.uint8), np.ones((2, 2, 3), np.uint8)]
        >>> [int(frame.max()) for frame in one_buffer(frames)], len({id(frame) for frame in one_buffer(frames)})
        ([0, 1], 1)
    """
    buffer = np.empty_like(frames[0])
    for frame in frames:
        buffer[...] = frame
        yield buffer


def unrelated_frames(shape: tuple[int, ...], count: int) -> list[np.ndarray]:
    """Return ``count`` identical noise frames that share nothing with a ``moving_squares_scene`` frame.

    Examples:
        >>> frames = unrelated_frames((8, 8, 3), 2)
        >>> len(frames), bool((frames[0] == frames[1]).all()), frames[0].shape
        (2, True, (8, 8, 3))
    """
    noise = np.random.default_rng(1).integers(0, 255, shape, dtype=np.uint8)
    return [noise.copy() for _ in range(count)]


def field_of(detections: sv.Detections, field: str) -> np.ndarray:
    """Return the array behind ``field``: a ``data`` entry such as ``class_name``, else the attribute of that name.

    Examples:
        >>> detections = sv.Detections(
        ...     xyxy=np.zeros((1, 4)), class_id=np.array([3]), data={"class_name": np.array(["cat"])}
        ... )
        >>> field_of(detections, "class_name").tolist(), field_of(detections, "class_id").tolist()
        (['cat'], [3])
    """
    return detections.data[field] if field in detections.data else getattr(detections, field)


class TestDetectorSchedule:
    def test_detector_runs_on_every_nth_frame(self) -> None:
        frames, boxes = moving_squares_scene(7)
        detector = RecordingDetector(frames, boxes)
        list(predict_video(detector, frames, detect_every=3))
        assert detector.seen == [0, 3, 6]

    def test_detect_every_one_never_builds_a_propagator(self, monkeypatch: pytest.MonkeyPatch) -> None:
        frames, boxes = moving_squares_scene(3)
        monkeypatch.setattr(video, "_BoxPropagator", None)  # calling it would raise
        assert len(list(predict_video(RecordingDetector(frames, boxes), frames, detect_every=1))) == 3

    def test_detect_every_one_runs_the_detector_on_every_frame(self) -> None:
        frames, boxes = moving_squares_scene(4)
        detector = RecordingDetector(frames, boxes)
        list(predict_video(detector, frames, detect_every=1))
        assert detector.seen == [0, 1, 2, 3]

    def test_one_result_per_frame(self) -> None:
        frames, boxes = moving_squares_scene(5)
        results = list(predict_video(RecordingDetector(frames, boxes), frames, detect_every=4))
        assert len(results) == 5

    def test_numpy_integer_detect_every_is_accepted(self) -> None:
        frames, boxes = moving_squares_scene(3)
        detector = RecordingDetector(frames, boxes)
        list(predict_video(detector, frames, detect_every=np.int64(3)))
        assert detector.seen == [0]

    def test_threshold_and_keywords_reach_the_model(self) -> None:
        frames, boxes = moving_squares_scene(1)
        detector = RecordingDetector(frames, boxes)
        list(predict_video(detector, frames, threshold=0.3, shape=(64, 64)))
        assert detector.kwargs == [{"threshold": 0.3, "include_source_image": False, "shape": (64, 64)}]

    def test_frame_size_change_with_the_same_content_starts_a_new_detector_frame(self) -> None:
        small, small_boxes = moving_squares_scene(2)
        large = [cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR) for frame in small]
        detector = RecordingDetector(small + large, np.concatenate([small_boxes, small_boxes * 2]))
        list(predict_video(detector, small + large, detect_every=10))
        assert detector.seen == [0, 2]

    def test_scene_cut_runs_the_detector_before_the_next_scheduled_frame(self) -> None:
        scene, scene_boxes = moving_squares_scene(3)
        frames = scene + unrelated_frames(scene[0].shape, 2)
        boxes = np.concatenate([scene_boxes, np.tile(scene_boxes[:1], (2, 1, 1))])
        detector = RecordingDetector(frames, boxes)
        list(predict_video(detector, frames, detect_every=10))
        assert detector.seen == [0, 3]

    @pytest.mark.parametrize("step", [18, -18])
    def test_gradual_brightness_change_is_not_a_scene_cut(self, step: int) -> None:
        scene, scene_boxes = moving_squares_scene(4)
        # Start mid-range so that neither direction clips at 0 or 255 within four frames.
        frames = [
            np.clip(frame.astype(np.int16) // 2 + 90 + step * index, 0, 255).astype(np.uint8)
            for index, frame in enumerate(scene)
        ]
        detector = RecordingDetector(frames, scene_boxes)
        list(predict_video(detector, frames, detect_every=10))
        assert detector.seen == [0]

    def test_schedule_restarts_after_a_scene_cut(self) -> None:
        scene, scene_boxes = moving_squares_scene(2)
        frames = scene + unrelated_frames(scene[0].shape, 4)
        boxes = np.concatenate([scene_boxes, np.tile(scene_boxes[:1], (4, 1, 1))])
        detector = RecordingDetector(frames, boxes)
        list(predict_video(detector, frames, detect_every=3))
        assert detector.seen == [0, 2, 5]


class TestBoxPropagation:
    @pytest.mark.parametrize(
        "steps",
        [
            pytest.param(((2, 0),), id="horizontal"),
            pytest.param(((0, -3),), id="vertical"),
            pytest.param(((2, 0), (-2, 2)), id="two-boxes-with-their-own-motion"),
        ],
    )
    def test_propagated_boxes_follow_the_motion(self, steps: tuple[tuple[int, int], ...]) -> None:
        frames, boxes = moving_squares_scene(3, steps=steps)
        results = list(predict_video(RecordingDetector(frames, boxes), frames, detect_every=3))
        assert np.abs(results[2].xyxy - boxes[2]).max() < 1.0

    def test_box_follows_motion_in_a_frame_larger_than_the_flow_image(self) -> None:
        small, small_boxes = moving_squares_scene(3)
        frames = [cv2.resize(frame, None, fx=6, fy=6, interpolation=cv2.INTER_LINEAR) for frame in small]
        boxes = small_boxes * 6
        results = list(predict_video(RecordingDetector(frames, boxes), frames, detect_every=3))
        assert np.abs(results[2].xyxy - boxes[2]).max() < 3.0

    @pytest.mark.parametrize(
        ("field", "expected"),
        [("class_id", _FIRST_CLASS_ID), ("confidence", _FIRST_CONFIDENCE), ("class_name", "square-0")],
    )
    def test_propagated_frame_keeps_the_fields_of_the_detector_frame(self, field: str, expected: Any) -> None:
        frames, boxes = moving_squares_scene(3)
        results = list(predict_video(RecordingDetector(frames, boxes), frames, detect_every=3))
        assert field_of(results[2], field).tolist() == [expected]

    @pytest.mark.parametrize("field", ["confidence", "class_id", "class_name"])
    def test_editing_a_propagated_frame_leaves_the_others_alone(self, field: str) -> None:
        frames, boxes = moving_squares_scene(4)
        results = list(predict_video(RecordingDetector(frames, boxes), frames, detect_every=4))
        before = field_of(results[2], field).copy()
        field_of(results[1], field)[:] = 0
        assert field_of(results[2], field).tolist() == before.tolist()

    def test_box_follows_motion_when_the_reader_reuses_one_buffer(self) -> None:
        frames, boxes = moving_squares_scene(2, steps=((3, 0),))
        first = sv.Detections(xyxy=boxes[0].copy(), confidence=np.array([0.9]), class_id=np.array([0]))
        results = [d.xyxy.copy() for d in predict_video(FixedDetector(first), one_buffer(frames), detect_every=2)]
        assert np.abs(results[1] - boxes[1]).max() < 1.0

    def test_drawing_on_the_detector_frame_does_not_disturb_the_propagation(self) -> None:
        frames, boxes = moving_squares_scene(2, steps=((3, 0),))
        stream = predict_video(RecordingDetector(frames, boxes), frames, detect_every=2)
        next(stream)
        frames[0][40:80, 20:60] = 128  # a caller painting over the detected object
        assert np.abs(next(stream).xyxy - boxes[1]).max() < 1.0

    @pytest.mark.parametrize("field", ["xyxy", "confidence", "class_id", "class_name"])
    def test_editing_the_detector_frame_detections_leaves_the_propagated_frames_alone(self, field: str) -> None:
        frames, boxes = moving_squares_scene(2)
        untouched = list(predict_video(RecordingDetector(frames, boxes), frames, detect_every=2))[1]
        stream = predict_video(RecordingDetector(frames, boxes), frames, detect_every=2)
        field_of(next(stream), field)[...] = 0
        assert field_of(next(stream), field).tolist() == field_of(untouched, field).tolist()

    def test_detect_every_one_passes_detections_that_cannot_be_copied_through(self) -> None:
        frames, _ = moving_squares_scene(3)
        handle = np.empty(1, dtype=object)
        handle[0] = threading.Lock()
        detections = sv.Detections(
            xyxy=np.array([[1.0, 1.0, 5.0, 5.0]]),
            confidence=np.array([0.9]),
            class_id=np.array([0]),
            data={"handle": handle},
        )
        assert len(list(predict_video(FixedDetector(detections), frames, detect_every=1))) == 3

    def test_box_dtype_stays_that_of_the_detector_frame(self) -> None:
        frames, _ = moving_squares_scene(2)
        detections = sv.Detections(
            xyxy=np.array([[20.0, 40.0, 60.0, 80.0]], dtype=np.float32),
            confidence=np.array([0.9]),
            class_id=np.array([0]),
        )
        results = list(predict_video(FixedDetector(detections), frames, detect_every=2))
        assert results[1].xyxy.dtype == np.float32

    def test_no_detections_propagate_as_empty(self) -> None:
        frames, _ = moving_squares_scene(3)
        results = list(predict_video(FixedDetector(sv.Detections.empty()), frames, detect_every=3))
        assert [len(result) for result in results] == [0, 0, 0]

    @pytest.mark.parametrize(
        ("axis", "limit_index", "box"),
        [
            pytest.param(1, 2, [170.0, 40.0, 190.0, 80.0], id="right-edge"),
            pytest.param(0, 3, [60.0, 100.0, 100.0, 126.0], id="bottom-edge"),
        ],
    )
    def test_box_is_clipped_to_the_frame(self, axis: int, limit_index: int, box: list[float]) -> None:
        still, _ = moving_squares_scene(1)
        shifted = np.roll(still[0], 6, axis=axis)
        propagator = _BoxPropagator(still[0], np.array([box]))
        limit = still[0].shape[axis]
        assert propagator.update(shifted)[0, limit_index] == limit

    def test_box_follows_the_motion_of_most_of_its_points(self) -> None:
        frames, boxes = moving_squares_scene(2, steps=((6, 0),))
        wide_box = boxes[0] + np.array([[0.0, -30.0, 60.0, 30.0]])
        moved = _BoxPropagator(frames[0], wide_box).update(frames[1])
        assert abs(moved[0, 0] - wide_box[0, 0]) < 0.1

    def test_small_box_follows_the_motion_of_its_surroundings(self) -> None:
        still, _ = moving_squares_scene(1, height=1080, width=1920)
        propagator = _BoxPropagator(still[0], np.array([[600.0, 400.0, 608.0, 408.0]]))
        moved = propagator.update(np.roll(still[0], 20, axis=1))
        assert abs(moved[0, 0] - 620.0) < 3.0

    def test_box_around_a_moving_box_does_not_follow_it(self) -> None:
        frames, boxes = moving_squares_scene(2, steps=((3, 0),))
        inner = boxes[0, 0]
        outer = inner + np.array([-4.0, -4.0, 4.0, 4.0])
        moved = _BoxPropagator(frames[0], np.array([outer, inner])).update(frames[1])
        assert abs(moved[0, 0] - outer[0]) < 1.0

    @pytest.mark.parametrize(("tracked_points", "moves"), [(2, False), (3, True)])
    def test_box_moves_only_with_the_minimum_tracked_points(self, tracked_points: int, moves: bool) -> None:
        frames, boxes = moving_squares_scene(2, steps=((3, 0),))
        propagator = _BoxPropagator(frames[0], boxes[0])
        own_points = np.flatnonzero(propagator._owner == 0)
        propagator._tracked[:] = False
        propagator._tracked[own_points[:tracked_points]] = True
        moved = propagator.update(frames[1])
        assert (abs(moved[0, 0] - boxes[0, 0, 0]) > 1.0) == moves


class TestRefusals:
    @pytest.mark.parametrize("detect_every", [0, -1, 2.5, True, "2"])
    def test_detect_every_that_is_not_a_positive_integer_raises_when_called(self, detect_every: object) -> None:
        frames, boxes = moving_squares_scene(1)
        with pytest.raises(ValueError, match="detect_every"):
            predict_video(RecordingDetector(frames, boxes), frames, detect_every=detect_every)  # type: ignore[arg-type]

    def test_include_source_image_in_the_keywords_raises_when_called(self) -> None:
        frames, boxes = moving_squares_scene(1)
        with pytest.raises(ValueError, match="include_source_image"):
            predict_video(RecordingDetector(frames, boxes), frames, include_source_image=True)

    def test_missing_opencv_raises_with_install_hint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        frames, boxes = moving_squares_scene(1)
        monkeypatch.setattr(video, "_IS_OPENCV_AVAILABLE", False)
        with pytest.raises(ImportError, match="opencv-python-headless"):
            predict_video(RecordingDetector(frames, boxes), frames)

    @pytest.mark.parametrize(
        "frame",
        [
            pytest.param(np.zeros((8, 8, 3), np.float32), id="float"),
            pytest.param(np.zeros((8, 8, 4), np.uint8), id="four-channels"),
            pytest.param(np.zeros((8, 8), np.uint8), id="gray"),
            "not an array",
        ],
    )
    def test_frame_that_is_not_rgb_uint8_raises(self, frame: object) -> None:
        with pytest.raises(ValueError, match="uint8"):
            list(predict_video(FixedDetector(sv.Detections.empty()), [frame]))  # type: ignore[list-item]

    def test_masks_raise(self) -> None:
        frames, _ = moving_squares_scene(1)
        detections = sv.Detections(
            xyxy=np.array([[1.0, 1.0, 5.0, 5.0]]),
            confidence=np.array([0.9]),
            class_id=np.array([0]),
            mask=np.ones((1, 128, 192), dtype=bool),
        )
        with pytest.raises(ValueError, match="masks"):
            list(predict_video(FixedDetector(detections), frames))

    def test_non_detections_result_raises(self) -> None:
        frames, _ = moving_squares_scene(1)
        with pytest.raises(TypeError, match="sv.Detections"):
            list(predict_video(FixedDetector(object()), frames))


class TestWithoutOpenCV:
    def test_module_imports_and_names_the_install_command(self) -> None:
        child = "\n".join(
            [
                "import sys",
                "sys.modules['cv2'] = None",  # `import cv2` raises ModuleNotFoundError, as when it is not installed
                "from rfdetr.utilities.video import predict_video",
                "try:",
                "    predict_video(None, [])",
                "except ImportError as error:",
                "    assert 'opencv-python-headless' in str(error), error",
                "else:",
                "    raise SystemExit('no ImportError')",
            ]
        )
        source_root = str(Path(rfdetr.__file__).resolve().parents[1])
        env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [source_root, os.environ.get("PYTHONPATH")]))}
        result = subprocess.run([sys.executable, "-c", child], env=env, capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stderr


class TestRealModel:
    def test_detection_model_yields_one_detections_per_frame(self) -> None:
        frames, _ = moving_squares_scene(3)
        model = RFDETRNano(pretrain_weights=None)
        results = list(predict_video(model, frames, detect_every=2, threshold=0.0))
        assert [isinstance(result, sv.Detections) for result in results] == [True, True, True]

    def test_propagated_frame_keeps_the_detector_frame_confidences(self) -> None:
        frames, _ = moving_squares_scene(2)
        model = RFDETRNano(pretrain_weights=None)
        results = list(predict_video(model, frames, detect_every=2, threshold=0.0))
        assert results[1].confidence.tolist() == results[0].confidence.tolist()

    def test_segmentation_model_raises(self) -> None:
        frames, _ = moving_squares_scene(1)
        model = RFDETRSegNano(pretrain_weights=None)
        with pytest.raises(ValueError, match="masks"):
            list(predict_video(model, frames, threshold=0.0))
