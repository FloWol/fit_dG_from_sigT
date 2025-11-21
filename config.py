from pydantic import BaseModel, FilePath, PositiveFloat, PositiveInt, Field, conint, confloat, field_validator
from typing import Literal, Optional, Any
import os

class CorrelatorSettings(BaseModel):
    use: Literal["Clustering", "Stat_correlation", "PDBCorrelation"]

class CorrelationTimeSettings(BaseModel):
    calculator: Literal["StokesEinstein", "linear"]

class ModelConfig(BaseModel):
    frequency: PositiveFloat
    fit_method: str
    data_path: str
    sigA: Literal["lowest_T", "fit", "custom", "custom_temp"]
    sigB: Literal["0", "highest_T", "fit", "custom", "custom_temp"]
    correct_for_correlation_time: bool
    max_iterations: PositiveInt
    Cp_model: bool
    fit_SM: bool
    Tm: Optional[confloat(gt=0)]
    fix_Tm: bool
    ignore_lowest_T: bool
    correlator: CorrelatorSettings
    correlation_time_interpolation: CorrelationTimeSettings

    @field_validator('data_path')
    def data_path_must_exist(cls, v):
        if not os.path.exists(v):
            raise ValueError(f"data_path '{v}' does not exist.")
        return v
