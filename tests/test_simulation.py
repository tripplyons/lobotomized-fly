import json

import pytest

from flybird.brain import Condition
from flybird.simulation import SimulationConfig, _collides, make_course, run_episode, run_experiment


def test_deterministic_json_and_shared_course():
    cfg = SimulationConfig(duration=5)
    intact = Condition("intact", "Intact")
    zero = Condition("zero", "Zero", "amount", 0)
    a = run_episode(intact, 3, cfg)
    assert json.dumps(a, sort_keys=True) == json.dumps(run_episode(intact, 3, cfg), sort_keys=True)
    b = run_episode(zero, 3, cfg)
    assert a["course_sha256"] == b["course_sha256"]
    assert a["available_pipes"] == b["available_pipes"] == 1
    assert len(a["frames"]) == len(b["frames"]) == 151
    assert b["score"] == 0 and not b["completed"]
    assert b["frames"][-1]["y"] == next(frame["y"] for frame in b["frames"] if not frame["alive"])


@pytest.mark.parametrize("seed", range(5))
def test_engineered_teacher_fixture_completes_development_courses(seed):
    episode = run_episode(Condition("intact", "Intact"), seed)
    assert episode["completed"]
    assert episode["score"] == episode["available_pipes"] == 12
    assert episode["pipe_completion_fraction"] == 1


def test_collision_and_pipe_crossing_geometry():
    cfg = SimulationConfig()
    pipe = {"x": cfg.bird_x - .03, "gap_y": .5}
    assert _collides(.5, [pipe], cfg) is None
    assert _collides(.1, [pipe], cfg) == "pipe"
    assert _collides(.001, [], cfg) == "boundary"
    assert make_course(1, cfg) != make_course(2, cfg)


@pytest.mark.parametrize("kwargs", [{"duration": 0}, {"duration": float("nan")},
                                       {"fps": 0}, {"fps": 29}, {"gap_size": 1},
                                       {"pipe_speed": -1}, {"flap_velocity": .2}])
def test_invalid_configs(kwargs):
    with pytest.raises(ValueError):
        SimulationConfig(**kwargs)


def test_experiment_keeps_failed_conditions_and_paired_deltas():
    conditions = [Condition("intact", "Intact"), Condition("zero", "Zero", "amount", 0)]
    experiment = run_experiment(duration=5, seeds=[1, 2], conditions=conditions)
    assert len(experiment["episodes"]) == 4
    assert experiment["summary"][0]["paired_score_delta"] == 0
    assert experiment["summary"][1]["paired_score_delta"] == -1
    with pytest.raises(ValueError):
        run_experiment(seeds=[1, 1])


@pytest.mark.parametrize("duration", [.02, .01, 1.01, 1e-15])
def test_non_frame_aligned_duration_is_rejected(duration):
    with pytest.raises(ValueError, match="frame-aligned"):
        SimulationConfig(duration=duration)


def test_one_frame_episode_reports_actual_end_time():
    config = SimulationConfig(duration=1 / 30)
    episode = run_episode(Condition("intact", "Intact"), config=config)
    assert len(episode["frames"]) == 2
    assert episode["frames"][-1]["t"] == episode["survival_time"] == config.duration
