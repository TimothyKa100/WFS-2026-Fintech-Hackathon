from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime

from state.store import store
from services.pipeline import run_pipeline
from services.asset_overview import get_asset_overview

router = APIRouter()


def corr_dict_to_matrix(corr: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    if not corr or not isinstance(corr, dict):
        return {"assets": [], "matrix": []}

    assets = list(corr.keys())
    matrix = [[float(corr.get(r, {}).get(c, 0.0)) for c in assets] for r in assets]
    return {"assets": assets, "matrix": matrix}


@router.get("/state")
def get_state(date: Optional[str] = Query(None, description="Optional date (YYYY-MM-DD)")):
    """
    If date is provided, run pipeline for that date (days_back) then return updated state.
    If date is not provided, just return current state.
    """
    if date is not None:
        try:
            selected_date = datetime.strptime(date, "%Y-%m-%d").date()
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

        today = datetime.today().date()
        if selected_date > today:
            raise HTTPException(status_code=400, detail="Future dates are not allowed.")

        days_back = (today - selected_date).days
        if days_back < 1:
            days_back = 1
        if days_back > 60:
            raise HTTPException(status_code=400, detail="Date too far in the past for intraday data.")

        run_pipeline(days=days_back, prediction_date=date, include_predictions=True)

    data = store.get_all()

    # Normalize correlation for frontend heatmap (if stored as dict-of-dicts)
    corr = data.get("correlation_matrix")
    if isinstance(corr, dict):
        data["corr_matrix"] = corr_dict_to_matrix(corr)

    return data


@router.get("/correlation")
def get_correlation(date: Optional[str] = Query(None, description="Optional date (YYYY-MM-DD)")):
    """
    If date is provided, run pipeline for that date then return correlation.
    If not provided, return whatever is currently in store.
    """
    if date is not None:
        try:
            selected_date = datetime.strptime(date, "%Y-%m-%d").date()
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

        today = datetime.today().date()
        if selected_date > today:
            raise HTTPException(status_code=400, detail="Future dates not allowed.")

        days_back = (today - selected_date).days
        if days_back < 1:
            days_back = 1
        if days_back > 60:
            raise HTTPException(status_code=400, detail="Date too far in past for intraday data.")

        run_pipeline(days=days_back, prediction_date=date, include_predictions=False)

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


@router.get("/asset-overview")
def asset_overview(asset: str):
    overview = get_asset_overview(asset)
    return {"asset": asset, "overview": overview}
