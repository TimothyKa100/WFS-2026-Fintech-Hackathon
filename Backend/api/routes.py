# backend/api/routes.py

from fastapi import APIRouter
from state.store import store

router = APIRouter()


@router.get("/state")
def get_state():
    return store.get_all()


@router.get("/correlation")
def get_correlation():
    return store.get("correlation_matrix")


@router.get("/anomalies")
def get_anomalies():
    return store.get("anomalies_summary")


@router.get("/health")
def health():
    return {"status": "ok"}