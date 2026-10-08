# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Video inference that runs the detector on every ``detect_every``-th frame (issue #1555).

On the frames in between, the boxes of the last detector frame are moved with sparse Lucas-Kanade optical flow (OpenCV,
CPU): the median displacement of the corner points inside each box. Compared with a constant-velocity prediction this
follows camera motion and objects that change direction, and it needs no model change. Objects that appear between two
detector frames are found on the next detector frame.
"""

from __future__ import annotations

__all__ = ["predict_video"]

__doctest_requires__ = {("predict_video", "_BoxPropagator"): ["cv2"]}

import copy
import numbers
from collections.abc import Iterable, Iterator
from typing import Any, Protocol

import numpy as np
import supervision as sv

from rfdetr.utilities.package import is_installed

#: OpenCV is not a direct rfdetr dependency; ``predict_video`` raises an ``ImportError`` with the install command
#: without it.
_IS_OPENCV_AVAILABLE = is_installed("cv2")
if _IS_OPENCV_AVAILABLE:
    import cv2

#: Side length of the gray copy the flow runs on; flow cost does not grow with the camera resolution.
_FLOW_SIDE = 512
#: Corner budget per detector frame. The corners are found on the whole flow image and the ones outside every box are
#: discarded, so the background uses part of it.
_MAX_CORNERS = 1200
#: Corner points fewer than this inside a box are topped up with a 5 x 5 grid so that small boxes can still move.
_MIN_POINTS_PER_BOX = 6
#: A box moves only if at least this many of its points were tracked successfully.
_MIN_TRACKED_POINTS = 3
#: Mean absolute gray difference (0-255 scale) between two consecutive flow images above which the frame is a scene
#: cut and the detector runs on it. Measured on the 512-pixel flow image of supervision's video assets: consecutive
#: frames of real footage reached at most 0.055 of the range, and the first frames of two different assets differ by
#: at least 0.167.
_SCENE_CUT_THRESHOLD = 0.10 * 255
#: Lucas-Kanade search window in flow pixels; with the pyramid it sets how far a point may move between two frames.
_LK_WINDOW = (15, 15)
#: Lucas-Kanade pyramid levels above the full-size flow image.
_LK_PYRAMID_LEVELS = 2
#: Lucas-Kanade stopping rule: 20 iterations or a 0.03 pixel update, whichever comes first. The first entry is
#: ``cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT`` (2 | 1).
_LK_CRITERIA = (3, 20, 0.03)


class _Predictor(Protocol):
    """What ``predict_video`` needs from a model: RF-DETR's ``predict`` for one RGB image."""

    def predict(self, images: np.ndarray, threshold: float = ..., **kwargs: Any) -> Any:
        """Detect objects in one RGB image."""
        ...


class _BoxPropagator:
    """Moves the boxes of one detector frame through the following frames with a single flow call per frame.

    Args:
        frame: The detector frame, RGB ``uint8`` of shape ``(H, W, 3)``.
        xyxy: Boxes of the detector frame in pixels, shape ``(N, 4)``.

    Examples:
        >>> rng = np.random.default_rng(0)
        >>> canvas = rng.integers(0, 255, (96, 128, 3), dtype=np.uint8)
        >>> shifted = np.roll(canvas, 3, axis=1)
        >>> propagator = _BoxPropagator(canvas, np.array([[30.0, 20.0, 70.0, 60.0]]))
        >>> moved = propagator.update(shifted)
        >>> bool(abs(moved[0, 0] - 33.0) < 1.0)
        True
    """

    def __init__(self, frame: np.ndarray, xyxy: np.ndarray) -> None:
        self._height, self._width = frame.shape[:2]
        self._scale = _FLOW_SIDE / max(self._height, self._width)
        self._boxes: np.ndarray = np.asarray(xyxy, dtype=np.float64).reshape(-1, 4)
        self._previous = self._to_flow_gray(frame)
        self._origin, self._owner = self._sample_points(self._previous, self._boxes * self._scale)
        self._points = self._origin.copy()
        self._tracked = np.ones(len(self._origin), dtype=bool)
        self._pending: np.ndarray | None = None

    def _to_flow_gray(self, frame: np.ndarray) -> np.ndarray:
        """Return ``frame`` as a gray image scaled to ``_FLOW_SIDE`` pixels on its long side."""
        size = (max(1, round(self._width * self._scale)), max(1, round(self._height * self._scale)))
        return cv2.cvtColor(cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR), cv2.COLOR_RGB2GRAY)

    @staticmethod
    def _sample_points(gray: np.ndarray, boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Pick flow points: image corners owned by the smallest box containing them, plus a grid in sparse boxes."""
        corners = cv2.goodFeaturesToTrack(gray, _MAX_CORNERS, 0.005, 5)
        points = np.zeros((0, 2), np.float32) if corners is None else corners.reshape(-1, 2).astype(np.float32)
        owner = np.full(len(points), -1, dtype=np.int32)
        if len(boxes) and len(points):
            inside = (
                (points[:, None, 0] >= boxes[None, :, 0])
                & (points[:, None, 0] <= boxes[None, :, 2])
                & (points[:, None, 1] >= boxes[None, :, 1])
                & (points[:, None, 1] <= boxes[None, :, 3])
            )
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            smallest = np.where(inside, -area[None, :], -np.inf).argmax(axis=1)
            owner = np.where(inside.any(axis=1), smallest, -1).astype(np.int32)
        keep = owner >= 0
        parts, owners = [points[keep]], [owner[keep]]
        height, width = gray.shape
        for index, (x0, y0, x1, y1) in enumerate(boxes):
            if (owner == index).sum() >= _MIN_POINTS_PER_BOX:
                continue
            grid_x, grid_y = np.meshgrid(np.linspace(x0, x1, 5), np.linspace(y0, y1, 5))
            grid = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1).astype(np.float32)
            grid = grid[(grid[:, 0] >= 0) & (grid[:, 0] < width) & (grid[:, 1] >= 0) & (grid[:, 1] < height)]
            parts.append(grid)
            owners.append(np.full(len(grid), index, dtype=np.int32))
        return np.concatenate(parts), np.concatenate(owners)

    def is_scene_cut(self, frame: np.ndarray) -> bool:
        """Tell whether ``frame`` differs from the previous one by more than ``_SCENE_CUT_THRESHOLD``.

        The flow image of ``frame`` is kept for the next :meth:`update`, so each frame is converted once.

        Args:
            frame: The next frame, RGB ``uint8`` with the shape of the detector frame.

        Returns:
            ``True`` for a hard cut or any change as large as one.
        """
        self._pending = self._to_flow_gray(frame)
        return float(np.abs(self._pending.astype(np.int16) - self._previous.astype(np.int16)).mean()) > (
            _SCENE_CUT_THRESHOLD
        )

    def update(self, frame: np.ndarray) -> np.ndarray:
        """Track the points into ``frame`` and return every box moved by its points' median displacement.

        Args:
            frame: The next frame, RGB ``uint8`` with the shape of the detector frame.

        Returns:
            Boxes in pixels, shape ``(N, 4)``, clipped to the frame. A box with too few tracked points keeps the
            position it had on the detector frame.
        """
        gray = self._pending if self._pending is not None else self._to_flow_gray(frame)
        self._pending = None
        if len(self._points):
            start = self._points.reshape(-1, 1, 2)
            moved, status, _ = cv2.calcOpticalFlowPyrLK(
                self._previous,
                gray,
                start,
                np.empty_like(start),
                winSize=_LK_WINDOW,
                maxLevel=_LK_PYRAMID_LEVELS,
                criteria=_LK_CRITERIA,
            )
            tracked = (status.reshape(-1) == 1) & self._tracked
            self._points = np.where(tracked[:, None], moved.reshape(-1, 2), self._points)
            self._tracked = tracked
        self._previous = gray
        boxes: np.ndarray = self._boxes.copy()
        for index in range(len(boxes)):
            selected = (self._owner == index) & self._tracked
            if selected.sum() >= _MIN_TRACKED_POINTS:
                shift = np.median(self._points[selected] - self._origin[selected], axis=0) / self._scale
                boxes[index] += np.array([shift[0], shift[1], shift[0], shift[1]])
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, self._width)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, self._height)
        return np.asarray(boxes)


def predict_video(
    model: _Predictor,
    frames: Iterable[np.ndarray],
    *,
    detect_every: int = 2,
    threshold: float = 0.5,
    **predict_kwargs: Any,
) -> Iterator[sv.Detections]:
    """Detect objects in a video, running the model on every ``detect_every``-th frame only.

    The first frame and every ``detect_every``-th frame after it go through ``model.predict``. On the frames in
    between, the detections of the last detector frame are kept (class, confidence) and their boxes are moved with
    Lucas-Kanade optical flow, which runs on the CPU. An object that appears between two detector frames is reported
    from the next detector frame on. A frame whose mean absolute gray difference from the previous frame (on a
    512-pixel gray copy) exceeds 10 % of the range (most hard cuts) also runs the detector; a cut between two
    similar-looking or equally dark scenes can go
    unnoticed until the next scheduled detector frame. A propagated frame carries ``xyxy``, ``confidence``,
    ``class_id`` and ``data`` of the detector frame; ``tracker_id`` and ``metadata`` are not carried. Segmentation
    masks and keypoints are not propagated, so models that return them are refused. The flow cost grows with the
    number of boxes; with hundreds of boxes (a very low ``threshold`` or a crowded scene) it can exceed the cost of
    the detector call it replaces.

    Args:
        model: An RF-DETR detection model, e.g. ``RFDETRSmall()``.
        frames: Video frames as RGB ``uint8`` arrays of shape ``(H, W, 3)``, in order. A frame whose size differs
            from the previous one, or that follows a scene cut, runs the detector.
        detect_every: Run the detector every this many frames. ``1`` runs it on every frame.
        threshold: Confidence threshold passed to ``model.predict``.
        **predict_kwargs: Further keyword arguments for ``model.predict``, such as ``shape``. ``include_source_image``
            is refused: ``predict_video`` always passes ``False``.

    Yields:
        One ``sv.Detections`` per input frame.

    Raises:
        ImportError: If OpenCV is not installed (``pip install opencv-python-headless``).
        ValueError: If ``detect_every`` is not an integer of at least 1, or ``predict_kwargs`` contains
            ``include_source_image``. Raised when the function is called.
        ValueError: If a frame is not a ``(H, W, 3)`` ``uint8`` NumPy array, or the detections of a frame carry
            segmentation masks. Raised while iterating, together with anything ``model.predict`` raises.
        TypeError: If ``model.predict`` returns anything but ``sv.Detections`` for a frame (keypoint models). Raised
            while iterating.

    Examples:
        >>> class FixedBox:
        ...     def predict(self, image, threshold=0.5, **kwargs):
        ...         return sv.Detections(
        ...             xyxy=np.array([[10.0, 10.0, 40.0, 40.0]]), confidence=np.array([0.9]), class_id=np.array([0])
        ...         )
        >>> frames = [np.zeros((64, 64, 3), np.uint8) for _ in range(3)]
        >>> [len(d) for d in predict_video(FixedBox(), frames, detect_every=2)]
        [1, 1, 1]
    """
    if not _IS_OPENCV_AVAILABLE:
        raise ImportError(
            "predict_video needs OpenCV for optical flow. Install it with: pip install opencv-python-headless"
        )
    if isinstance(detect_every, bool) or not isinstance(detect_every, numbers.Integral) or detect_every < 1:
        raise ValueError(f"detect_every must be an integer of at least 1, got {detect_every!r}.")
    if "include_source_image" in predict_kwargs:
        raise ValueError("predict_video always passes include_source_image=False; remove it from the keywords.")
    return _predict_video(model, frames, detect_every, threshold, predict_kwargs)


def _predict_video(
    model: _Predictor,
    frames: Iterable[np.ndarray],
    detect_every: int,
    threshold: float,
    predict_kwargs: dict[str, Any],
) -> Iterator[sv.Detections]:
    """Generator behind :func:`predict_video`, so that its arguments are checked when it is called."""
    propagator: _BoxPropagator | None = None
    detector: sv.Detections | None = None
    shape: tuple[int, ...] | None = None
    frames_since_detector = 0
    for frame in frames:
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError(f"Frames must be RGB uint8 arrays of shape (H, W, 3), got {_describe(frame)}.")
        frames_since_detector += 1
        if (
            propagator is not None
            and detector is not None
            and frames_since_detector < detect_every
            and frame.shape == shape
            and not propagator.is_scene_cut(frame)
        ):
            yield sv.Detections(
                xyxy=propagator.update(frame).astype(detector.xyxy.dtype),
                confidence=copy.deepcopy(detector.confidence),
                class_id=copy.deepcopy(detector.class_id),
                data=copy.deepcopy(detector.data),
            )
            continue
        detections = model.predict(frame, threshold=threshold, include_source_image=False, **predict_kwargs)
        if not isinstance(detections, sv.Detections):
            raise TypeError(f"predict_video needs sv.Detections from model.predict, got {type(detections).__name__}.")
        if detections.mask is not None:
            raise ValueError("predict_video does not propagate segmentation masks; use a detection model.")
        shape, frames_since_detector = frame.shape, 0
        if detect_every > 1:
            # Snapshot before the yield: the caller may edit the detections in place, and a reader may reuse its
            # frame buffer for the next frame. With detect_every=1 no frame is propagated, so nothing is built.
            detector = copy.deepcopy(detections)
            propagator = _BoxPropagator(frame, detector.xyxy)
        yield detections


def _describe(frame: object) -> str:
    """Describe a rejected frame for an error message.

    Examples:
        >>> _describe(np.zeros((2, 2), np.float32))
        'float32 (2, 2)'
        >>> _describe("frame")
        'str'
    """
    return f"{frame.dtype} {frame.shape}" if isinstance(frame, np.ndarray) else type(frame).__name__
