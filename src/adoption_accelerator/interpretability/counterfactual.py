"""
Counterfactual generation for the Feature Interpretation Layer.
"""
from __future__ import annotations

import copy
import logging
from typing import Any

from langchain_core.tools import tool

from adoption_accelerator.inference.contracts import PredictionRequest
from adoption_accelerator.inference.feature_builder import build_feature_vector
from adoption_accelerator.inference.serving import get_inference_pipeline

logger = logging.getLogger(__name__)

def _get_alternatives(request: PredictionRequest) -> list[tuple[str, str, Any, str]]:
    """Get realistic alternatives for actionable features.
    
    Returns a list of tuples: (feature_name, display_name, new_value, expected_impact_text)
    """
    t = request.tabular
    alts = []
    
    if t.video_amt == 0:
        alts.append(("video_amt", "Video Amount", 1, "Add a video"))
    if t.fee > 0:
        alts.append(("fee", "Adoption Fee", 0.0, "Waive the adoption fee"))
    if t.vaccinated != 1:
        alts.append(("vaccinated", "Vaccinated", 1, "Vaccinate the pet"))
    if t.dewormed != 1:
        alts.append(("dewormed", "Dewormed", 1, "Deworm the pet"))
    if t.sterilized != 1:
        alts.append(("sterilized", "Sterilized", 1, "Sterilize the pet"))
    if not t.name or not t.name.strip():
        alts.append(("name", "Pet Name", "Buddy", "Give the pet a name"))
        
    return alts

@tool
def generate_counterfactuals(
    request: PredictionRequest,
    target_class: int,
    current_class: int = -1,
) -> list[dict[str, Any]]:
    """Generate counterfactual suggestions for improving adoption speed.
    
    Simulates changing actionable attributes of the listing and re-running
    the inference pipeline to observe the expected impact on predictability.
    
    Parameters
    ----------
    request : PredictionRequest
        The original prediction request.
    target_class : int
        The target adoption speed class (lower is faster).
    current_class : int
        The current predicted class. If -1, it will be dynamically computed.
        
    Returns
    -------
    list[dict]
        List of counterfactual recommendations that improve the predicted class
        or expected value.
    """
    try:
        pipeline = get_inference_pipeline()
    except Exception as exc:
        logger.error("Could not load inference pipeline for counterfactuals: %s", exc)
        return []
        
    feature_schema = pipeline.feature_schema.get("features", [])
    
    # 1. Baseline prediction
    baseline_fv = build_feature_vector(request, feature_schema)
    baseline_pred = pipeline.predict_batch(baseline_fv)
    
    if current_class == -1:
        current_class = int(baseline_pred["predictions"][0])
        
    baseline_ev = float(baseline_pred["expected_values"][0])
    
    alts = _get_alternatives(request)
    results = []
    
    for feat_name, display_name, new_val, impact_text in alts:
        # Create modified request
        mod_req = copy.deepcopy(request)
        setattr(mod_req.tabular, feat_name, new_val)
        
        # Predict modified
        mod_fv = build_feature_vector(mod_req, feature_schema)
        mod_pred = pipeline.predict_batch(mod_fv)
        
        mod_class = int(mod_pred["predictions"][0])
        mod_ev = float(mod_pred["expected_values"][0])
        
        # We want to improve adoption speed (lower class number is faster)
        # Class 0: Same day, ..., Class 4: 100+ days
        class_improved = mod_class < current_class
        ev_improved = mod_ev < baseline_ev
        
        if class_improved or ev_improved:
            if class_improved:
                impact_desc = f"Improves adoption speed prediction from class {current_class} to {mod_class}"
            else:
                diff = baseline_ev - mod_ev
                impact_desc = f"Improves expected adoption time score by {diff:.2f}"
                
            results.append({
                "feature": display_name,
                "current_value": str(getattr(request.tabular, feat_name)),
                "suggested_value": str(new_val) if feat_name != "name" else "Yes",
                "expected_impact": impact_desc,
                "ev_improvement": baseline_ev - mod_ev,
                "class_improved": class_improved,
            })
            
    # Rank by improvement magnitude
    results.sort(key=lambda x: x["ev_improvement"], reverse=True)
    
    return results
