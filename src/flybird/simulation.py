"""Deterministic offline Flappy Bird trajectories in normalized coordinates."""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import random
import statistics

from .brain import Condition, EngineeredBrain, MODEL_VERSION, default_conditions


@dataclass(frozen=True)
class SimulationConfig:
    duration: float = 30.0
    fps: int = 30
    width: float = 1.0
    height: float = 1.0
    bird_x: float = .24
    radius: float = .018
    pipe_width: float = .13
    gap_size: float = .32
    pipe_speed: float = .24
    pipe_spacing: float = .54
    first_pipe_x: float = 1.02
    gravity: float = 1.35
    flap_velocity: float = -.43
    flap_cooldown: float = .12
    physics_hz: int = 120

    def __post_init__(self):
        if not math.isfinite(self.duration) or self.duration <= 0:
            raise ValueError("duration must be finite and positive")
        if not isinstance(self.fps, int) or self.fps <= 0 or self.physics_hz % self.fps:
            raise ValueError("fps must be a positive divisor of physics_hz")
        if self.physics_hz < 30:
            raise ValueError("physics_hz must be at least 30")
        frames = self.duration * self.fps
        if (not math.isfinite(frames) or round(frames) < 1
                or not math.isclose(frames, round(frames), rel_tol=0, abs_tol=1e-9)):
            raise ValueError("duration must be frame-aligned (duration * fps must be an integer)")
        if self.width != 1 or self.height != 1:
            raise ValueError("world coordinates are normalized to one")
        for name in ("radius", "pipe_width", "gap_size", "pipe_speed", "pipe_spacing", "gravity", "flap_cooldown"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 2 * self.radius < self.gap_size < 1 or self.pipe_spacing <= self.pipe_width:
            raise ValueError("invalid pipe geometry")
        if not self.radius < self.bird_x < 1 - self.radius or self.first_pipe_x <= self.bird_x + self.radius:
            raise ValueError("invalid initial geometry")
        if not math.isfinite(self.flap_velocity) or self.flap_velocity >= 0:
            raise ValueError("flap_velocity must be finite and upward (negative)")


def make_course(seed: int, config: SimulationConfig):
    """Generated before any controller acts; shared by all conditions."""
    rng = random.Random(seed)
    count = math.ceil((config.duration * config.pipe_speed + 2) / config.pipe_spacing) + 1
    margin = config.gap_size / 2 + .08
    previous = .5
    pipes = []
    for index in range(count):
        gap = min(1 - margin, max(margin, previous + rng.uniform(-.19, .19)))
        pipes.append({"id": index, "x": config.first_pipe_x + index * config.pipe_spacing, "gap_y": gap})
        previous = gap
    return pipes


def _collides(y, pipes, config):
    if y - config.radius <= 0 or y + config.radius >= 1:
        return "boundary"
    for pipe in pipes:
        # Circle-versus-rectangle distance, rather than a generous point test.
        nearest_x = min(pipe["x"] + config.pipe_width, max(pipe["x"], config.bird_x))
        dx = config.bird_x - nearest_x
        top = pipe["gap_y"] - config.gap_size / 2
        bottom = pipe["gap_y"] + config.gap_size / 2
        if dx * dx + max(0, y - top) ** 2 <= config.radius ** 2:
            return "pipe"
        if dx * dx + max(0, bottom - y) ** 2 <= config.radius ** 2:
            return "pipe"
    return None


def run_episode(condition: Condition, seed: int = 7, config: SimulationConfig | None = None, *, controller=None):
    config = config or SimulationConfig()
    brain = controller if controller is not None else EngineeredBrain(condition, seed)
    course = make_course(seed, config)
    course_hash = hashlib.sha256(json.dumps(course, sort_keys=True).encode()).hexdigest()
    y, vy, score = .5, 0.0, 0
    alive, last_flap, cause = True, -100.0, None
    survival_time = config.duration
    frames = []
    substeps = config.physics_hz // config.fps
    dt = 1 / config.physics_hz
    ticks = int(round(config.duration * config.fps))
    passed = set()
    for frame_index in range(ticks + 1):
        flap = False
        if frame_index:
            for substep in range(substeps):
                t = ((frame_index - 1) * substeps + substep + 1) * dt
                pipes = [{**pipe, "x": pipe["x"] - config.pipe_speed * t} for pipe in course]
                if not alive:
                    continue
                next_pipe = next(pipe for pipe in pipes if pipe["x"] + config.pipe_width >= config.bird_x - config.radius)
                if substep == 0 and brain.act(y, vy, next_pipe["gap_y"]) and t - last_flap >= config.flap_cooldown:
                    vy, last_flap, flap = config.flap_velocity, t, True
                vy += config.gravity * dt
                y += vy * dt
                cause = _collides(y, pipes, config)
                if cause:
                    alive, survival_time = False, t
                    continue
                for pipe in pipes:
                    if pipe["id"] not in passed and pipe["x"] + config.pipe_width < config.bird_x - config.radius:
                        passed.add(pipe["id"])
                        score += 1
        t = frame_index / config.fps
        visible = [{**pipe, "x": pipe["x"] - config.pipe_speed * t} for pipe in course if -config.pipe_width < pipe["x"] - config.pipe_speed * t < 1.15]
        frames.append({"t": t, "y": y, "vy": vy, "alive": alive, "score": score, "flap": flap,
                       "pipes": visible, "activity": dict(brain.last_activity) if alive else {region: 0.0 for region in brain.last_activity}})
    possible_score = sum(pipe["x"] - config.pipe_speed * config.duration + config.pipe_width < config.bird_x - config.radius for pipe in course)
    return {"condition": asdict(condition), "seed": seed, "frames": frames, "score": score,
            "possible_score": possible_score, "pass_fraction": score / possible_score if possible_score else None,
            "available_pipes": possible_score, "pipe_completion_fraction": score / possible_score if possible_score else None,
            "survival_time": survival_time, "completed": alive, "death_cause": cause,
            "brain": brain.manifest(), "course_sha256": course_hash}


def run_experiment(seed=7, duration=30.0, fps=30, *, seeds=None, conditions=None, config=None, controller_factory=None):
    """Return all trials and paired aggregate metrics without cherry-picking.

    ``seed`` is the displayed trial when ``seeds`` is omitted. Pass multiple
    seeds for evaluation; lesion and course RNG streams are independent.
    """
    config = config or SimulationConfig(duration=duration, fps=fps)
    seeds = list(seeds) if seeds is not None else [seed]
    conditions = list(conditions) if conditions is not None else default_conditions()
    if not seeds or not conditions or len(set(seeds)) != len(seeds):
        raise ValueError("provide nonempty conditions and unique seeds")
    if len({condition.id for condition in conditions}) != len(conditions):
        raise ValueError("condition ids must be unique")
    episodes = [run_episode(condition, trial_seed, config, controller=controller_factory(condition, trial_seed) if controller_factory else None) for trial_seed in seeds for condition in conditions]
    baseline = {episode["seed"]: episode for episode in episodes if episode["condition"]["kind"] == "intact"}
    summary = []
    for condition in conditions:
        trials = [episode for episode in episodes if episode["condition"]["id"] == condition.id]
        scores = [episode["score"] for episode in trials]
        survivals = [episode["survival_time"] for episode in trials]
        summary.append({"condition_id": condition.id, "label": condition.label, "n": len(trials),
                        "mean_score": statistics.mean(scores), "score_std": statistics.stdev(scores) if len(scores) > 1 else 0.0,
                        "mean_survival_time": statistics.mean(survivals), "completion_rate": sum(episode["completed"] for episode in trials) / len(trials),
                        "pass_fractions": [episode["pass_fraction"] for episode in trials],
                        "scores": scores, "survival_times": survivals,
                        "paired_score_delta": statistics.mean(episode["score"] - baseline[episode["seed"]]["score"] for episode in trials) if baseline else None})
    return {"schema_version": 1, "model": episodes[0]["brain"]["model"],
            "claim": "MaleCNS graph with engineered dynamics, encoder and fitted decoder; not biological behavior validation." if controller_factory else "Synthetic engineered controller ablation demo; not a full fly brain or MaleCNS simulation.",
            "config": asdict(config), "seeds": seeds, "conditions": [asdict(condition) for condition in conditions],
            "episodes": episodes, "summary": summary}


def train_connectome_decoder(graph, *, train_seeds=(101, 103, 107), duration=12.0,
                             fps=30, reservoir_config=None, input_regions=("visual",),
                             output_regions=("motor",)):
    """Collect neural activity while an openly engineered teacher flies.

    Training and evaluation seeds are recorded separately by the caller. The
    teacher is used ONLY here; evaluation must instantiate ConnectomeBrain with
    the returned fixed decoder. The teacher never bypasses evaluation activity.
    """
    from .brain import ConnectomeBrain, ReservoirConfig, fit_decoder
    rows, targets, training_trials = [], [], []
    condition = Condition("intact", "Intact retained MaleCNS model")
    for seed in train_seeds:
        brain = ConnectomeBrain(graph, condition, seed, config=reservoir_config,
                                input_regions=input_regions, output_regions=output_regions)
        class Teacher:
            @property
            def last_activity(self):
                return brain.last_activity

            def act(self, y, vy, gap_y):
                rows.append(brain.observe(y, vy, gap_y).copy())
                target = y + .18 * vy - gap_y - .10
                targets.append(target)
                return target > 0

            def manifest(self):
                # Training manifest only; do not repeatedly enumerate full edges.
                return {"model": "engineered teacher collecting MaleCNS neural states"}
        trial = run_episode(condition, seed, SimulationConfig(duration=duration, fps=fps), controller=Teacher())
        training_trials.append({key: trial[key] for key in ("seed", "score", "possible_score", "survival_time", "completed")})
    decoder = fit_decoder(rows, targets, reservoir_config)
    decoder.update({"train_seeds": list(train_seeds), "training_trials": training_trials,
                    "reservoir_config": asdict(reservoir_config or ReservoirConfig()),
                    "simulation_config": asdict(SimulationConfig(duration=duration, fps=fps)),
                    "input_regions": list(input_regions), "output_regions": list(output_regions)})
    return decoder
