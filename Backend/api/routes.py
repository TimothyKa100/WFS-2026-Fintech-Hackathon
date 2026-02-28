# backend/api/routes.py

<<<<<<< HEAD
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
=======
from services.asset_overview import get_asset_overview
from fastapi import APIRouter, Query, HTTPException
from datetime import datetime

>>>>>>> 0a0f0c0cece26958cf3b63e42ec376c2cac8a68d
from state.store import store
from services.pipeline import run_pipeline

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
<<<<<<< HEAD
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
=======
def get_state(
    date: str = Query(..., description="Select date (YYYY-MM-DD)")
):
    """
    Updates backend state based on selected date.
    Calculates how many days back that date is
    and propagates it through the pipeline.
    """

    # -------------------------
    # Validate Date Format
    # -------------------------

    try:
        selected_date = datetime.strptime(date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid date format. Use YYYY-MM-DD."
        )

    today = datetime.today().date()

    if selected_date > today:
        raise HTTPException(
            status_code=400,
            detail="Future dates are not allowed."
        )

    # -------------------------
    # Calculate Days Back
    # -------------------------

    days_back = (today - selected_date).days

    # Ensure at least 1 day
    if days_back < 1:
        days_back = 1

    # Prevent too old intraday requests (Yahoo limitation)
    if days_back > 60:
        raise HTTPException(
            status_code=400,
            detail="Date too far in the past for intraday data."
        )

    # -------------------------
    # Run Pipeline With Date
    # -------------------------

    run_pipeline(days=days_back)

    # Return updated state
    return store.get_all()


@router.get("/correlation")
def get_correlation(
    date: str = Query(..., description="Select date (YYYY-MM-DD)")
):
    """
    Returns correlation matrix for selected date.
    """

    try:
        selected_date = datetime.strptime(date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid date format. Use YYYY-MM-DD."
        )

    today = datetime.today().date()

    if selected_date > today:
        raise HTTPException(
            status_code=400,
            detail="Future dates not allowed."
        )

    days_back = (today - selected_date).days

    if days_back < 1:
        days_back = 1

    if days_back > 60:
        raise HTTPException(
            status_code=400,
            detail="Date too far in past for intraday data."
        )

    # Run pipeline
    run_pipeline(days=days_back)

    # Return only correlation matrix
    correlation = store.get("correlation_matrix")

    if correlation is None:
        raise HTTPException(
            status_code=404,
            detail="Correlation matrix not available."
        )

    return correlation
>>>>>>> 0a0f0c0cece26958cf3b63e42ec376c2cac8a68d


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/asset-overview")
def asset_overview(asset: str):
    """
    Returns a hardcoded AI-style overview for selected asset.
    """
    overview = get_asset_overview(asset)

    return {
        "asset": asset,
        "overview": overview
    }

