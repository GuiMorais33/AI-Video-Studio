import numpy as np

from studio.masks import contour, load_mask, overlay, save_mask


def test_save_and_load_round_trip(tmp_path):
    mask = np.zeros((20, 30), dtype=bool)
    mask[5:15, 10:20] = True
    path = tmp_path / "m" / "00000.png"
    save_mask(path, mask)
    assert np.array_equal(load_mask(path), mask)
    assert load_mask(tmp_path / "missing.png") is None


def test_contour_is_border_only():
    mask = np.zeros((20, 20), dtype=bool)
    mask[4:16, 4:16] = True
    edge = contour(mask, width=2)
    assert edge[4, 10] and edge[5, 10]
    assert not edge[6, 10]
    assert not edge[10, 10]
    assert not edge[0, 0]


def test_overlay_tints_only_masked_pixels():
    frame = np.full((20, 20, 3), 100, dtype=np.uint8)
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 5:15] = True
    out = overlay(frame, [(mask, (0, 0, 255))], alpha=0.5)
    assert tuple(out[0, 0]) == (100, 100, 100)
    assert tuple(out[10, 10]) == (50, 50, 177)
    assert tuple(out[5, 10]) == (0, 0, 255)  # contorno sólido
