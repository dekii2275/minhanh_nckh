"""Validate published classification tables using only Python's standard library."""
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def rows(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def truth(value):
    assert value in {'True', 'False'}
    return value == 'True'


def verify():
    all_rows = {}
    hits = {}
    for folder, configurations in [('alignment_step2', ['C1', 'C2', 'C3', 'C4']),
                                   ('calibration_step3', list('ABCDEFGH'))]:
        output = ROOT / 'outputs' / folder
        records = rows(output / 'per_video_predictions.csv')
        assert len(records) == len({row['id'] for row in records}) == 253
        assert sum(int(row['frames']) for row in records) == 17747
        labels = [int(row['label']) for row in records]
        assert set(labels) == set(range(15))
        all_rows[folder] = records
        hits[folder] = {}
        metric_rows = rows(output / 'configuration_metrics.csv')
        metrics = {row['configuration']: row for row in metric_rows}
        for config in configurations:
            predicted = [int(row[f'{config}_prediction']) for row in records]
            correct = [a == b for a, b in zip(labels, predicted)]
            hits[folder][config] = correct
            assert correct == [truth(row[f'{config}_correct']) for row in records]
            accuracy = sum(correct) / 253 * 100
            f1 = sum(2 * sum(y == k and p == k for y, p in zip(labels, predicted)) /
                     (sum(y == k for y in labels) + sum(p == k for p in predicted))
                     for k in range(15)) / 15 * 100
            assert math.isclose(accuracy, float(metrics[config]['top1_accuracy']), abs_tol=1e-10)
            assert math.isclose(f1, float(metrics[config]['macro_f1']), abs_tol=1e-10)
        if folder == 'calibration_step3':
            for config in configurations:
                diff = [int(a) - int(b) for a, b in zip(hits[folder][config], hits[folder]['A'])]
                m = metrics[config]
                assert diff.count(1) == int(m['incorrect_to_correct'])
                assert diff.count(-1) == int(m['correct_to_incorrect'])
                assert math.isclose(sum(diff) / 253 * 100, float(m['accuracy_delta_pp']), abs_tol=1e-10)
        else:
            for row in rows(output / 'paired_comparisons.csv'):
                on, off = row['comparison'].split('-')
                diff = [int(a) - int(b) for a, b in zip(hits[folder][on], hits[folder][off])]
                assert diff.count(1) == int(row['improved_videos'])
                assert diff.count(-1) == int(row['harmed_videos'])
                assert math.isclose(sum(diff) / 253 * 100, float(row['accuracy_delta_pp']), abs_tol=1e-10)
    step2, step3 = all_rows['alignment_step2'], all_rows['calibration_step3']
    for a, b in zip(step2, step3):
        assert (a['id'], a['label'], a['production_swap']) == (b['id'], b['label'], b['production_swap'])
        assert a['C3_prediction'] == b['A_prediction']
        assert a['C4_prediction'] == b['H_prediction']
    print(json.dumps({'passed': True, 'videos': 253, 'frames': 17747,
                      'verified': ['12 configuration metrics', 'paired improvement/harm counts',
                                   'same video IDs and swap flags', 'A=C3 and H=C4 predictions']}))


if __name__ == '__main__':
    verify()
