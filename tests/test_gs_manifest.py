import json
from pathlib import Path

from fathomfollow.gs.recorded import GSScene
from fathomfollow.gs.watersplatting import load_colmap_poses_fixture


def test_gsscene_roundtrip(tmp_path: Path) -> None:
    scene = GSScene(
        scene_id="test",
        source_dataset="seathru",
        library="watersplatting",
        n_gaussians=1000,
        pose_source="colmap",
        checkpoint_path=str(tmp_path / "ckpt"),
        train_psnr=28.5,
    )
    path = tmp_path / "scene.json"
    scene.write(path)
    loaded = GSScene.load(path)
    assert loaded.scene_id == "test"
    assert loaded.train_psnr == 28.5


def test_colmap_pose_loader() -> None:
    data = {
        "poses": [
            {"position": [0, 0, -5], "orientation": [0, 0, 0, 1]},
            {"position": [1, 0, -5], "orientation": [0, 0, 0, 1]},
        ]
    }
    poses = load_colmap_poses_fixture(data)
    assert len(poses) == 2
    assert poses[0].position == (0, 0, -5)
