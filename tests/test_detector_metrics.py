from fathomfollow.perception.detector import parse_training_metrics


def test_parse_training_metrics_from_ultralytics_dict() -> None:
    raw = {
        "metrics/mAP50(B)": 0.42,
        "metrics/mAP50-95(B)": 0.21,
        "metrics/precision(B)": 0.55,
        "metrics/recall(B)": 0.48,
    }
    metrics = parse_training_metrics(raw)
    assert metrics["mAP50"] == 0.42
    assert metrics["mAP50-95"] == 0.21
    assert metrics["precision"] == 0.55
    assert metrics["recall"] == 0.48


def test_parse_training_metrics_missing_keys_default_zero() -> None:
    metrics = parse_training_metrics({})
    assert metrics["mAP50"] == 0.0
    assert metrics["mAP50-95"] == 0.0
