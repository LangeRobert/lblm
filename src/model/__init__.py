import numpy as np
from pydantic import BaseModel

BONES = [
    [12, 11],  # right shoulder to left shoulder

    # right side
    [12, 24],  # right upper
    [24, 26],  # right hip to right knee
    [26, 28],  # right knee to right ankle
    [12, 14],  # right shoulder to right elbow
    [14, 16],  # right elbow to right wrist

    # left side
    [11, 23],  # left upper
    [23, 25],  # left hip to left knee
    [25, 27],  # left knee to left ankle
    [11, 13],  # left shoulder to left elbow
    [13, 15],  # left elbow to left wrist

    [12, 0],  # right shoulder to nose
    [11, 0],  # left shoulder to nose
]


class BodyModel(BaseModel):
    # allow arbitrary attributes
    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            np.ndarray: lambda v: v.tolist()
        }

    data: np.ndarray

    def get_angle_vector(self) -> np.ndarray:
        # todo scale each vec by the fourth value
        return np.array([self.data[bone[1]][:3] - self.data[bone[0]][:3] for bone in BONES])

    @classmethod
    def default(cls):
        return cls(data=np.zeros((33, 34), dtype=np.float32))
