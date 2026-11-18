from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
from pathlib import Path
from typing import List

app = FastAPI(title="Trip Evaluator API")

# Load static trips data once at startup
TRIPS_PATH = Path(__file__).with_name("trips.json")
with TRIPS_PATH.open() as f:
    TRIPS = json.load(f)


class EvalRequest(BaseModel):
    evaluator_id: str  # sent from UI


class EvalResponse(BaseModel):
    trip_id: int
    trip_name: str
    evaluator_id: str
    prediction: str
    evaluation: str


def find_trip_by_evaluator(eid: str):
    for trip in TRIPS:
        if trip["evaluator_id"] == eid:
            return trip
    return None


def mock_evaluation(trip):
    """Placeholder for real business logic."""
    return f"Trip {trip['name']} is VALID for reservation."


def mock_prediction(trip):
    """Placeholder for an ML model prediction."""
    return f"Predicted demand rating: {len(trip['evaluator_id']) * 7}%"


@app.get("/trips", response_model=List[dict])
def list_trips():
    """Return the static list of 10 reserved trips."""
    return TRIPS


@app.post("/evaluate", response_model=EvalResponse)
def evaluate(req: EvalRequest):
    trip = find_trip_by_evaluator(req.evaluator_id)
    if not trip:
        raise HTTPException(status_code=404, detail="Evaluator ID not found")

    evaluation = mock_evaluation(trip)
    prediction = mock_prediction(trip)

    return EvalResponse(
        trip_id=trip["id"],
        trip_name=trip["name"],
        evaluator_id=req.evaluator_id,
        prediction=prediction,
        evaluation=evaluation,
    )
