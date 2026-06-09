from fathomfollow.config.models import LabelStrategy

__all__ = ["LabelStrategy", "composite_target_bbox", "write_yolo_label"]

composite_target_bbox = __import__(
    "fathomfollow.gs.render_pipeline", fromlist=["composite_target_bbox"]
).composite_target_bbox
write_yolo_label = __import__(
    "fathomfollow.gs.render_pipeline", fromlist=["write_yolo_label"]
).write_yolo_label
