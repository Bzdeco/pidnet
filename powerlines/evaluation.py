from powerlines.metrics import SegmentationMetrics


def segmentation_metrics():
    return SegmentationMetrics(
        scoring_thresholds=[0.5],
        tolerance_region=1.42,
    )
