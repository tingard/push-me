from gymnasium.envs.registration import register

from push_me.config import PushTPOConfig
from push_me.env import PushTPOEnv

__all__ = ["PushTPOConfig", "PushTPOEnv"]

_PRESETS = {
    "PushTPO-Full-Single-v0": PushTPOConfig(obs_mode="full"),
    "PushTPO-Lidar-Single-v0": PushTPOConfig(),
    "PushTPO-Full-Multimodal-v0": PushTPOConfig(obs_mode="full", shapes=["hexagon"], goal_margin=24.0),
    "PushTPO-Lidar-Multimodal-v0": PushTPOConfig(shapes=["hexagon"], goal_margin=24.0),
    "PushTPO-Lidar-Multi3-v0": PushTPOConfig(n_objects=3, shapes=["square"], goal_margin=24.0),
    "PushTPO-Lidar-Trap-v0": PushTPOConfig(
        n_objects=3, shapes=["square"], goal_margin=24.0, traps=True, n_traps=2
    ),
}

for _env_id, _config in _PRESETS.items():
    register(id=_env_id, entry_point="push_me.env:PushTPOEnv", kwargs={"config": _config})
del _env_id, _config
