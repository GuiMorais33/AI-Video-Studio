import numpy as np

from studio.masks import anchor_points, contour, load_mask, overlay, save_mask


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


def test_anchor_points_are_deep_spread_and_avoid_negative_clicks():
    mask = np.zeros((100, 200), dtype=bool)
    mask[20:80, 20:180] = True
    points = anchor_points(mask, k=3)
    assert len(points) == 3
    assert all(mask[int(y), int(x)] for x, y in points)
    assert all(30 <= y <= 70 for _, y in points)  # longe da borda
    xs = sorted(x for x, _ in points)
    assert xs[1] - xs[0] > 15 and xs[2] - xs[1] > 15  # espalhados
    near_click = anchor_points(mask, k=3, avoid=[[100.0, 50.0]])
    assert all(abs(x - 100) > 20 for x, _ in near_click)
    assert anchor_points(np.zeros((10, 10), dtype=bool)) == []
