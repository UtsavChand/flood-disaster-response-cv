from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


RecordStatus = Literal["verified", "user_entered", "simulated"]


class APIModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Provenance(APIModel):
    status: RecordStatus
    source_name: str
    source_url: str | None = None
    license: str | None = None
    last_updated: datetime | None = None
    notes: str | None = None


class Location(APIModel):
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    crs: str | None = None


class Facility(APIModel):
    id: str
    name: str
    facility_type: str
    location: Location = Field(default_factory=Location)
    capacity: int | None = Field(default=None, ge=0)
    provenance: Provenance


class Zone(APIModel):
    id: str
    zone_kind: Literal["administrative_area", "incident_area", "image_only"]
    name: str
    province: str | None = None
    district: str | None = None
    municipality: str | None = None
    location: Location = Field(default_factory=Location)
    flood_fraction: float | None = Field(default=None, ge=0, le=1)
    facility_ids: list[str] = Field(default_factory=list)
    provenance: Provenance


class Depot(APIModel):
    id: str
    name: str
    location: Location = Field(default_factory=Location)
    provenance: Provenance


class ResourceInventoryItem(APIModel):
    resource_type: Literal["boat", "medical", "truck"]
    depot_id: str
    quantity: int = Field(ge=0)
    readiness: Literal["ready", "not_ready", "unknown"] = "unknown"
    provenance: Provenance


class ImageryRecord(APIModel):
    id: str
    filename: str
    captured_at: datetime | None = None
    source_name: str
    georeferenced: bool = False
    location: Location = Field(default_factory=Location)
    provenance: Provenance


class PredictionRecord(APIModel):
    id: str
    imagery_id: str
    model_name: str
    model_version: str | None = None
    flood_fraction: float = Field(ge=0, le=1)
    confidence: float | None = Field(default=None, ge=0, le=1)
    location_verified: bool = False
    created_at: datetime
    provenance: Provenance


class PlanItem(APIModel):
    zone_id: str
    resource_type: Literal["boat", "medical", "truck"]
    quantity: int = Field(ge=0)


class AllocationPlan(APIModel):
    id: str
    strategy: str
    status: Literal["proposed", "edited", "approved", "rejected"]
    items: list[PlanItem] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    created_at: datetime
    policy_version: str | None = None


class PlanDecision(APIModel):
    plan_id: str
    decision: Literal["approved", "rejected"]
    coordinator_id: str
    note: str | None = None
    decided_at: datetime