from powerlines.metrics import SegmentationMetrics


def segmentation_metrics(minimal_logging: bool = False):
    return SegmentationMetrics(
        bin_thresholds=[
            0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95
        ],
        tolerance_region=1.42,
        minimal_logging=minimal_logging
    )
