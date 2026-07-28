"""Model registries — encoders and agents are swappable by config key.

To scale the model stack: drop a new file in this package, register the class
with @encoder("name") or @agent("name"), and point config.yaml at it
(train.encoder / train.agent). The trainer never imports architectures directly.
"""
ENCODER_REGISTRY: dict[str, type] = {}
AGENT_REGISTRY: dict[str, type] = {}


def encoder(name: str):
    def deco(cls):
        ENCODER_REGISTRY[name] = cls
        return cls
    return deco


def agent(name: str):
    def deco(cls):
        AGENT_REGISTRY[name] = cls
        return cls
    return deco


from . import cnn, ppo  # noqa: E402,F401  (import for registration side effects)
