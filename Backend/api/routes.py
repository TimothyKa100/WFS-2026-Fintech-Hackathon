# backend/api/routes.py

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from state.store import store

router = APIRouter()


def corr_dict_to_matrix(corr: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    if not corr or not isinstance(corr, dict):
        return {"assets": [], "matrix": []}

    assets = list(corr.keys())
    matrix = [
        [float(corr.get(r, {}).get(c, 0.0)) for c in assets]
        for r in assets
    ]
    return {"assets": assets, "matrix": matrix}


@router.get("/state")
def get_state():
    data = store.get_all()

    # Normalize correlation for frontend heatmap
    corr = data.get("correlation_matrix")
    if isinstance(corr, dict):
        data["corr_matrix"] = corr_dict_to_matrix(corr)

    return data


@router.get("/correlation")
def get_correlation():
    corr = store.get("correlation_matrix")
    if isinstance(corr, dict):
        return corr_dict_to_matrix(corr)
    return {"assets": [], "matrix": []}


class ControlsPayload(BaseModel):
    mode: Optional[str] = None
    playing: Optional[bool] = None
    speed: Optional[float] = None
    threshold: Optional[float] = None
    window: Optional[int] = None
    assets: Optional[List[str]] = None


@router.post("/controls")
def set_controls(payload: ControlsPayload):
    store.update("controls", payload.model_dump(exclude_none=True))
    return {"ok": True, "controls": store.get("controls")}


@router.get("/health")
def health():
    return {"status": "ok"}