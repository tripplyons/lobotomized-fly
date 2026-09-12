"""Measure the frozen gameplay checks without selecting favorable trials."""

EVALUATION_SEEDS = (7, 11, 19, 23, 41)


def evaluate_baseline(experiment: dict) -> dict:
    """Check intact performance and paired course identity on every fixed seed."""
    problems = []
    if experiment["config"]["duration"] != 30:
        problems.append("Evaluation requires 30-second courses.")
    episodes = experiment["episodes"]
    baselines = [episode for episode in episodes if episode["condition"]["kind"] == "intact"]
    seeds = [episode["seed"] for episode in baselines]
    if sorted(seeds) != sorted(EVALUATION_SEEDS):
        problems.append("Require exactly one intact trial for each fixed evaluation seed.")
    measurements = []
    for episode in baselines:
        available = episode["available_pipes"]
        score = episode["score"]
        if available <= 0 or not 0 <= score <= available:
            raise ValueError("Invalid whole-course pipe count or score")
        fraction = score / available
        measurements.append({"seed": episode["seed"], "score": score,
                             "available_pipes": available, "pipe_completion_fraction": fraction})
        if fraction < 0.9:
            problems.append(f"Seed {episode['seed']} passed {score}/{available} pipes, below 90%.")
    conditions = [condition["id"] for condition in experiment["conditions"]]
    if len(set(conditions)) != len(conditions):
        problems.append("Duplicate condition IDs.")
    for seed in EVALUATION_SEEDS:
        trials = [episode for episode in episodes if episode["seed"] == seed]
        if sorted(episode["condition"]["id"] for episode in trials) != sorted(conditions):
            problems.append(f"Seed {seed} is missing conditions or has duplicate trials.")
        if len({episode["course_sha256"] for episode in trials}) != 1:
            problems.append(f"Seed {seed} does not use one identical course across conditions.")
    return {"passed": not problems, "scope": "gameplay baseline and paired courses only",
            "measurements": measurements, "problems": problems}
