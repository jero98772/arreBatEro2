from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(
    title="Physics API",
    description="""
Computes energy from mass using:

![equation](https://latex.codecogs.com/svg.image?E=mc^2)
""",
)


class EnergyRequest(BaseModel):
    mass: float = Field(
        ...,
        description="Mass in kg. ![eq](https://latex.codecogs.com/svg.image?E=mc%5E2)",
    )


from collections.abc import Iterable


def sum_values(values: Iterable[float]) -> float:
    r"""
    Compute the sum of a sequence of numbers.

    .. math::

        S = \sum_{i=1}^{n} x_i

    :param values: an iterable of numbers to sum
    :return: the total, as a float
    :raises TypeError: if ``values`` contains non-numeric items
    """
    return float(sum(values))


@app.post("/energy", summary="Compute E=mc²")
def compute_energy(req: EnergyRequest):
    """Computes ![eq](https://latex.codecogs.com/svg.image?E=mc^2)."""
    c = 299_792_458
    return {"energy_joules": req.mass * c**2}


@app.post("/idk", summary="Compute E=mc²")
def idk(req: EnergyRequest):
    """i dont fucking know"""
    c = 1
    return {"energy_joules": req.mass * c**1}
