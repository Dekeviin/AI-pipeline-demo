"""Risk-procedure registry — evaluation methods are pluggable like models.

The demo ships Monte Carlo bootstrap only; to scale up, add a module with a
@risk_procedure("name") function (e.g. historical VaR, Kelly-fraction check,
stress scenarios) and list it under `risk.procedures` in config.yaml. The
evaluation stage runs every listed procedure and merges the reports.
"""
RISK_REGISTRY: dict[str, callable] = {}


def risk_procedure(name: str):
    def deco(fn):
        RISK_REGISTRY[name] = fn
        return fn
    return deco


from . import monte_carlo  # noqa: E402,F401  (import for registration side effects)
