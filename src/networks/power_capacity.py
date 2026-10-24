"""Capacity assumptions shared by the power-network simulations."""


HOSPITAL_DEMAND_MW = 4.0


def legacy_capacity_failure_probability(supply_mw, hospital_count):
    """Return the historical intermediate-capacity calculation unchanged."""

    return float(supply_mw) / (HOSPITAL_DEMAND_MW * hospital_count)


def hospital_capacity_failure_probability(supply_mw, hospital_count):
    """Return unmet-demand probability under the model's linear 4 MW rule."""

    if hospital_count <= 0:
        return 0.0
    adequacy = float(supply_mw) / (HOSPITAL_DEMAND_MW * hospital_count)
    return min(1.0, max(0.0, 1.0 - adequacy))
