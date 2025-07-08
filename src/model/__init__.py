from typing import List

import numpy as np
from pydantic import BaseModel


class BodyModel(BaseModel):
    # allow arbitrary attributes
    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            np.ndarray: lambda v: v.tolist()
        }

    data:np.ndarray

    @classmethod
    def default(cls):
        return cls(data=np.zeros((33,34),dtype=np.float32))
