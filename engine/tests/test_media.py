from fractions import Fraction

import pytest

from studio import media
from conftest import make_video


def test_prepare_clip_downscales_trims_and_caps_fps(tmp_path):
    src = make_video(tmp_path / "hd60.mp4", size="1280x720", rate=60, seconds=2.0)
    info = media.prepare_clip(src, tmp_path / "clip.mp4", start=0.5, max_seconds=1.0, max_side=854, max_fps=30)
    assert (info.width, info.height) == (854, 480)
    assert info.fps == Fraction(30)
    assert info.duration == pytest.approx(1.0, abs=0.1)
    assert info.has_audio


def test_prepare_clip_applies_phone_rotation(tmp_path):
    src = make_video(tmp_path / "phone.mp4", size="1280x720", seconds=0.5, rotation=90)
    info = media.prepare_clip(src, tmp_path / "clip.mp4", max_side=854)
    assert (info.width, info.height) == (480, 854)


def test_prepare_clip_never_upscales(tmp_path):
    src = make_video(tmp_path / "small.mp4", size="320x240", rate=24, seconds=0.5, audio=False)
    info = media.prepare_clip(src, tmp_path / "clip.mp4", max_side=854, max_fps=30)
    assert (info.width, info.height) == (320, 240)
    assert info.fps == Fraction(24)
    assert not info.has_audio


def test_prepare_clip_rejects_start_after_end(tmp_path, video):
    with pytest.raises(media.MediaError):
        media.prepare_clip(video, tmp_path / "clip.mp4", start=10)


def test_extract_and_encode_round_trip(tmp_path, video):
    clip = tmp_path / "clip.mp4"
    info = media.prepare_clip(video, clip)
    count = media.extract_frames(clip, tmp_path / "frames")
    assert count == 30
    assert (tmp_path / "frames" / "00000.jpg").exists()

    out = media.encode_frames(tmp_path / "frames", info.fps, tmp_path / "out.mp4", audio_from=clip)
    encoded = media.probe(out)
    assert encoded.has_audio
    assert encoded.duration == pytest.approx(1.0, abs=0.1)


def test_probe_rejects_non_video(tmp_path):
    bogus = tmp_path / "notes.txt"
    bogus.write_text("not a video")
    with pytest.raises(media.MediaError):
        media.probe(bogus)
