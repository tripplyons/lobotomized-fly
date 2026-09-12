from copy import deepcopy

from flybird.evaluation import EVALUATION_SEEDS, evaluate_baseline


def experiment():
    conditions = [{"id": "intact", "kind": "intact"}, {"id": "half", "kind": "random"}]
    return {"config": {"duration": 30}, "conditions": conditions,
            "episodes": [{"condition": condition, "seed": seed, "score": 9,
                          "available_pipes": 10, "course_sha256": str(seed)}
                         for seed in EVALUATION_SEEDS for condition in conditions]}


def test_each_seed_must_pass_not_only_the_average():
    run = experiment()
    assert evaluate_baseline(run)["passed"]
    run["episodes"][0]["score"] = 8
    assert not evaluate_baseline(run)["passed"]


def test_missing_duplicate_or_different_course_fails():
    run = experiment()
    missing = deepcopy(run)
    missing["episodes"].pop()
    duplicate = deepcopy(run)
    duplicate["episodes"].append(deepcopy(duplicate["episodes"][0]))
    changed = deepcopy(run)
    changed["episodes"][1]["course_sha256"] = "different"
    for broken in (missing, duplicate, changed):
        assert not evaluate_baseline(broken)["passed"]
