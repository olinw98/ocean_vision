from pathlib import Path

import numpy as np
from PIL import Image

from fathomfollow.sim.recorded import RecordedSimEnv
from fathomfollow.sim.recorder import build_fathomnet_proxy_fixture


def test_build_fathomnet_proxy_fixture(tmp_path: Path) -> None:
    img_dir = tmp_path / "images" / "train"
    img_dir.mkdir(parents=True)
    Image.new("RGB", (640, 480), color=(10, 80, 120)).save(img_dir / "a.jpg")
    Image.new("RGB", (640, 480), color=(20, 90, 130)).save(img_dir / "b.jpg")

    out = tmp_path / "proxy.npz"
    n = build_fathomnet_proxy_fixture(img_dir, out, n_frames=2)
    assert n == 2
    env = RecordedSimEnv(out)
    obs = env.reset()
    assert obs.rgb.shape == (480, 640, 3)
    env.close()
