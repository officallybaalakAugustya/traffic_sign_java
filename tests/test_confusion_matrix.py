from trafficsign.evaluation.confusion_matrix import ConfusionMatrix


def test_confusion_matrix():
    cm = ConfusionMatrix(num_classes=3)
    # Class 0: 2 correct, 1 predicted as class 1
    cm.record(0, 0)
    cm.record(0, 0)
    cm.record(0, 1)

    # Class 1: 3 correct
    cm.record(1, 1)
    cm.record(1, 1)
    cm.record(1, 1)

    # Class 2: 1 correct
    cm.record(2, 2)

    summary = cm.compute_summary()
    assert summary.total_samples == 7
    assert summary.accuracy == (6.0 / 7.0)
    assert len(summary.per_class_metrics) == 3
