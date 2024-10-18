from collections import Counter

from scipy.optimize import linear_sum_assignment
import numpy as np

from sparql_kgqa.sparql.utils2 import AskResult, KgManager, SelectResult


def exact_f1_score(pred: list[list[str]], target: list[list[str]]) -> float:
    pred_set = Counter(tuple(p) for p in pred)
    target_set = Counter(tuple(t) for t in target)

    tp = (pred_set & target_set).total()
    if tp == 0:
        return 0.0

    fp = (pred_set - target_set).total()
    fn = (target_set - pred_set).total()
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    return 2 * prec * rec / (prec + rec)


def assignment_f1_score(pred: list[list[str]], target: list[list[str]]) -> float:
    # create a matrix of distances between pred and target
    scores = np.zeros((len(pred), len(target)))

    pred_sets = [Counter(p) for p in pred]
    target_sets = [Counter(t) for t in target]

    for i, p_set in enumerate(pred_sets):
        for j, t_set in enumerate(target_sets):
            r = (p_set & t_set).total() / max(1, t_set.total())
            scores[i, j] = r

    rows, cols = linear_sum_assignment(scores, maximize=True)
    assert len(rows) == len(cols) == min(len(pred), len(target))
    assignment_scores = scores[rows, cols]
    tp = assignment_scores.sum()
    fn = (1 - assignment_scores).sum() + len(target) - len(rows)
    fp = len(pred) - len(rows)
    if tp <= 0.0:
        return 0.0

    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    return 2 * prec * rec / (prec + rec)


def f1_score(
    pred: SelectResult | AskResult,
    target: SelectResult | AskResult,
    exact: bool = False,
) -> float:
    if isinstance(target, AskResult) or isinstance(pred, AskResult):
        return float(pred == target)

    _, pred_rows = pred
    _, target_rows = target
    if len(pred_rows) == 0 and len(target_rows) == 0:
        return 1.0
    elif exact:
        return exact_f1_score(pred_rows, target_rows)
    else:
        return assignment_f1_score(pred_rows, target_rows)


def calculate_f1_score(
    pred: str,
    target: str,
    manager: KgManager,
    allow_empty_target: bool = True,
    endpoint: str | None = None,
    timeout: float | None = None,
    max_retries: int = 1,
    exact: bool = False,
) -> tuple[float | None, str | None, str | None]:
    pred_err, predictions = None, None
    try:
        predictions = manager.execute_sparql(pred, endpoint, timeout, max_retries)
    except Exception as e:
        pred_err = str(e)

    targets, target_err = None, None
    try:
        targets = manager.execute_sparql(target, endpoint, timeout, max_retries)
    except Exception as e:
        target_err = str(e)

    if pred_err is not None or target_err is not None:
        return (None, pred_err, target_err)

    assert targets is not None and predictions is not None
    if (
        not isinstance(targets, AskResult)
        and len(targets[1]) == 0
        and not allow_empty_target
    ):
        return None, None, "target set is empty"

    return f1_score(predictions, targets, exact), None, None
