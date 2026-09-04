"""Shared request schemas for backend API routers."""

from typing import Any, List

import numpy as np
from pydantic import BaseModel, Field, field_validator


class ForecastRequest(BaseModel):
    historical_speeds: List[Any] = Field(
        ...,
        description="Historical speed matrix tensor slice of shape (T, N) or (T, N, C)",
        json_schema_extra={"example": [[55.4, 62.1, 48.0], [54.2, 60.5, 45.2]]},
    )

    @field_validator("historical_speeds")
    @classmethod
    def validate_historical_speeds(cls, value):
        arr = np.asarray(value, dtype=float)
        if arr.ndim not in (2, 3) or arr.shape[0] == 0:
            raise ValueError("historical_speeds must be a non-empty 2D or 3D tensor")
        if arr.shape[1] not in (207, 716):
            raise ValueError("historical_speeds must contain 207 or 716 sensor nodes")
        if arr.ndim == 3 and arr.shape[2] != 3:
            raise ValueError("3D historical_speeds tensors must have 3 channels")
        if not np.isfinite(arr).all():
            raise ValueError("historical_speeds must contain only finite numbers")
        return value


class LLMQueryRequest(BaseModel):
    prompt: str = Field(..., description="User prompt or highway query", json_schema_extra={"example": "Why is I-5 South congested?"})
    sensor_id: int = Field(default=0, ge=0, description="Associated sensor node ID")
    city: str = Field(default="la", description="Target city identifier")
    time_label: str = Field(default="08:15 AM", description="Simulated current time")
    date_label: str = Field(default="2012-03-15", description="Simulated current date")
    step: int = Field(default=96, ge=0, description="Simulated time step index")
    origin_id: int = Field(default=0, ge=0, description="Active route origin ID")
    destination_id: int = Field(default=15, ge=0, description="Active route destination ID")

    @field_validator("city")
    @classmethod
    def normalize_city(cls, value):
        value = value.lower().strip()
        supported = {"la", "sd", "pems04", "pems08", "pems_bay", "pems03", "pems07"}
        if value not in supported:
            raise ValueError(f"city must be one of: {', '.join(sorted(supported))}")
        return value


class RerouteRequest(BaseModel):
    predicted_speeds: List[float] = Field(
        ...,
        description="Predicted speed values across corridor sensors",
        json_schema_extra={"example": [22.5, 18.4, 45.0]},
    )
    target_node_id: str = Field(
        ...,
        description="Queried corridor sensor ID for rerouting advisory",
        json_schema_extra={"example": "716156"},
    )

    @field_validator("predicted_speeds")
    @classmethod
    def validate_predicted_speeds(cls, value):
        if not value or not np.isfinite(np.asarray(value, dtype=float)).all():
            raise ValueError("predicted_speeds must be a non-empty list of finite numbers")
        return value

    @field_validator("target_node_id")
    @classmethod
    def validate_target_node_id(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("target_node_id must not be empty")
        return value


class RouteRequest(BaseModel):
    origin_id: int = Field(..., ge=0, description="Origin sensor node ID", json_schema_extra={"example": 0})
    destination_id: int = Field(..., ge=0, description="Destination sensor node ID", json_schema_extra={"example": 10})
    target_time: str = Field(default="08:45 AM", description="Target departure time string")
    city: str = Field(default="la", description="Target city/corridor identifier (la, sd, pems04...)")

    @field_validator("city")
    @classmethod
    def normalize_city(cls, value):
        value = value.lower().strip()
        supported = {"la", "sd", "pems04", "pems08", "pems_bay", "pems03", "pems07"}
        if value not in supported:
            raise ValueError(f"city must be one of: {', '.join(sorted(supported))}")
        return value
