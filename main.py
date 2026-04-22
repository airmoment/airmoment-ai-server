import pandas as pd                                       
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from decision_rule import decide
                                            
app = FastAPI()                         
model = joblib.load("xgb_regressor_best_0421.joblib")
                                                                                                                                                                
FEATURE_COLUMNS = [
    "route_id", "departure_airport_code", "arrival_airport_code", "outbound_date", "searched_day_of_week", "days_to_departure",                                                                                                    
    "is_weekend_search", "is_long_haul", "offer_count", "nonstop_ratio",
    "cheapest_nonstop_price", "cheapest_offer_has_layover", "current_cheapest_price",
    "curr_gap_to_typical_min", "curr_gap_to_typical_max",
    "hist_recent_std", "hist_recent_slope", "curr_vs_hist_mean",                                                                                                
    "price_change_1", "rolling_std_3", "price_vs_rolling_mean_3"
]                                                                                                                                                               
                                                                                                                                                                
class FlightFeatureRequest(BaseModel):
    route_id: str
    departure_airport_code: str
    arrival_airport_code: str
    outbound_date: str                                                                                                                                       
    searched_day_of_week: str                             
    days_to_departure: int                                                                                                                                      
    is_weekend_search: bool                               
    is_long_haul: bool
    offer_count: int                        
    nonstop_ratio: float                
    cheapest_nonstop_price: Optional[int] = None
    cheapest_offer_has_layover: bool                                                                                                                            
    current_cheapest_price: int
    curr_gap_to_typical_min: Optional[int] = None                                                                                                               
    curr_gap_to_typical_max: Optional[int] = None         
    hist_recent_std: Optional[float] = None
    hist_recent_slope: Optional[float] = None
    curr_vs_hist_mean: Optional[float] = None                                                                                                                   
    price_change_1: Optional[float] = None  
    rolling_std_3: Optional[float] = None                                                                                                                       
    price_vs_rolling_mean_3: Optional[float] = None 

@app.post("/predict")
def predict(request: FlightFeatureRequest):
    df = pd.DataFrame([request.model_dump()], columns=FEATURE_COLUMNS)
    predicted_drop = float(model.predict(df)[0])
    decision = decide(predicted_drop)
    return {"predictedDrop": predicted_drop, "decision": decision}
