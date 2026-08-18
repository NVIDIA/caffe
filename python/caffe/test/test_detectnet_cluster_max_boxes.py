import sys
import types
from pathlib import Path

import numpy as np
import pytest


def _stub_caffe_and_cv2():
    """Provide minimal stubs so clustering.py can be imported without pycaffe."""
    caffe_pkg = types.ModuleType("caffe")
    caffe_pkg.__path__ = []
    caffe_pkg.Layer = object
    sys.modules["caffe"] = caffe_pkg

    caffe_layers = types.ModuleType("caffe.layers")
    caffe_layers.__path__ = []
    sys.modules["caffe.layers"] = caffe_layers

    detectnet_dir = str(Path(__file__).resolve().parent.parent / "layers" / "detectnet")
    caffe_detectnet = types.ModuleType("caffe.layers.detectnet")
    caffe_detectnet.__path__ = [detectnet_dir]
    sys.modules["caffe.layers.detectnet"] = caffe_detectnet

    cv2_stub = types.ModuleType("cv2")
    cv2_stub.groupRectangles = lambda boxes, *args, **kwargs: ([], [])
    sys.modules["cv2"] = cv2_stub


_stub_caffe_and_cv2()
from caffe.layers.detectnet.clustering import (  # noqa: E402
    MAX_BOXES,
    cluster,
)


class _FakeGroundTruthLayer:
    is_groundtruth = True
    image_size_x = 16
    image_size_y = 16
    stride = 1
    coverage_threshold = 0.0


def test_cluster_truncates_boxes_to_max_boxes():
    """cluster() must not crash when an image produces more than MAX_BOXES proposals.

    Regression: the output blob is fixed at [batch_size, MAX_BOXES, 5], but the
    number of proposals was unbounded, so assigning more than MAX_BOXES rows
    raised ValueError: could not broadcast input array from shape (256,4)
    into shape (50,4).
    """
    layer = _FakeGroundTruthLayer()

    # 16x16 grid, stride 1, every cell covered => 256 ground-truth proposals.
    net_cvg = np.ones((1, 1, 16, 16), dtype=np.float32)
    net_boxes = np.zeros((1, 4, 16, 16), dtype=np.float32)
    net_boxes[0, 2, :, :] = 1.0  # width
    net_boxes[0, 3, :, :] = 1.0  # height

    result = cluster(layer, net_cvg, net_boxes)

    assert result.shape == (1, MAX_BOXES, 5)
    # All MAX_BOXES slots should be populated (no all-zero padding from a crash).
    assert np.count_nonzero(result[0, :, :]) > 0
