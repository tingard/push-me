import pathlib

import numpy as np
import pytest
from hypothesis import HealthCheck, settings

settings.register_profile("default", max_examples=100)
settings.register_profile(
    "pymunk", max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow]
)
settings.load_profile("default")

# physics-stepping properties are slow per-example; test modules touching
# pymunk apply this instead of the loaded default profile
pymunk_settings = settings.get_profile("pymunk")

SNAPSHOT_DIR = pathlib.Path(__file__).resolve().parent / "snapshots"


def pytest_addoption(parser):
    parser.addoption(
        "--snapshot-update",
        action="store_true",
        default=False,
        help="overwrite golden snapshot images instead of asserting against them",
    )


@pytest.fixture
def sdl_dummy_driver(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    yield


@pytest.fixture
def assert_matches_snapshot(request):
    update = request.config.getoption("--snapshot-update")

    def _check(
        name: str,
        frame: np.ndarray,
        mean_tol: float = 1.0,
        outlier_frac_tol: float = 0.02,
        outlier_threshold: int = 10,
    ) -> None:
        import pygame

        SNAPSHOT_DIR.mkdir(exist_ok=True)
        path = SNAPSHOT_DIR / f"{name}.png"

        if update or not path.exists():
            surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
            pygame.image.save(surface, str(path))
            return

        golden = np.transpose(pygame.surfarray.array3d(pygame.image.load(str(path))), (1, 0, 2))
        assert frame.shape == golden.shape, f"{name}: shape {frame.shape} != golden {golden.shape}"

        diff = np.abs(frame.astype(int) - golden.astype(int))
        mean_diff = float(diff.mean())
        outlier_frac = float((diff.max(axis=-1) > outlier_threshold).mean())

        assert mean_diff < mean_tol, f"{name}: mean pixel diff {mean_diff:.3f} >= tolerance {mean_tol}"
        assert outlier_frac < outlier_frac_tol, (
            f"{name}: {outlier_frac:.2%} of pixels differ by more than {outlier_threshold}/255 "
            f"(tolerance {outlier_frac_tol:.2%})"
        )

    return _check
