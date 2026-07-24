"""Regression test: unreadable images must not shift later image indices.

``load_images_base64`` indexes by position in the ORIGINAL paths list, not by
how many images have loaded so far. Otherwise an unreadable photo shifts
every later ``PhotoAssessment.image_index`` down by one, and the frontend
attaches a critique (and the BEST badge) to the wrong stored photo.
"""

import base64
from pathlib import Path

from adoption_accelerator.agents.nodes.visual_analyst import load_images_base64

# 1x1 transparent PNG
_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
    "YPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


def test_unreadable_first_photo_leaves_second_photos_index_unshifted(tmp_path: Path):
    good_photo = tmp_path / "photo1.png"
    good_photo.write_bytes(_PNG_BYTES)

    paths = [str(tmp_path / "missing.png"), str(good_photo)]

    loaded = load_images_base64(paths)

    assert len(loaded) == 1
    index, mime, _b64 = loaded[0]
    assert index == 1  # original upload position, not len(loaded) - 1 (== 0)
    assert mime == "image/png"


def test_cap_is_still_based_on_number_successfully_loaded(tmp_path: Path):
    good_photo = tmp_path / "photo.png"
    good_photo.write_bytes(_PNG_BYTES)

    # unreadable, then three readable photos, cap=2 -> only 2 survive
    paths = [str(tmp_path / "missing.png"), str(good_photo), str(good_photo), str(good_photo)]

    loaded = load_images_base64(paths, cap=2)

    assert len(loaded) == 2
    indices = [entry[0] for entry in loaded]
    assert indices == [1, 2]  # original positions of the two readable photos
