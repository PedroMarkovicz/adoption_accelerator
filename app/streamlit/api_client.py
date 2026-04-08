"""HTTP client for the Adoption Accelerator FastAPI backend.

All methods return raw dicts (parsed JSON). No Pydantic imports here.

Error handling strategy:
  - ConnectionError / ConnectError: sets st.session_state.connection_error = True.
  - HTTP 4xx: returns {"error": True, "status_code": ..., "details": ...}.
  - HTTP 5xx: returns {"error": True, "status_code": ..., "message": ...}.
  - Unexpected errors: returns {"error": True, "status_code": 0, "message": ...}.
"""

from __future__ import annotations

import json as _json
import logging

import httpx
import streamlit as st

from config import (
    API_BASE_URL,
    EXPLORE_TIMEOUT,
    HEALTH_TIMEOUT,
    PREDICT_TIMEOUT,
    STATUS_TIMEOUT,
)

logger = logging.getLogger(__name__)


def _make_error_dict(
    status_code: int,
    message: str,
    details: list[dict] | None = None,
) -> dict:
    """Build a client-side error dict matching the API error shape."""
    result: dict = {
        "error": True,
        "status_code": status_code,
        "message": message,
    }
    if details is not None:
        result["details"] = details
    return result


class AdoptionAPI:
    """Thin httpx wrapper for the Adoption Accelerator API."""

    def __init__(self, base_url: str = API_BASE_URL) -> None:
        self.base_url = base_url.rstrip("/")

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(
        self,
        profile: dict,
        images: list[tuple[str, bytes]] | None = None,
    ) -> dict:
        """POST /predict -- returns PredictionStatusResponse or error dict.

        If ``images`` is provided (list of (filename, raw_bytes)), the
        request is sent as multipart/form-data with the profile JSON in a
        ``profile`` form field and each image as an ``images`` file part.
        Otherwise, the profile is sent as a plain JSON body.
        """
        try:
            with httpx.Client(timeout=PREDICT_TIMEOUT) as client:
                if images:
                    files = [
                        ("images", (name, data, "image/jpeg"))
                        for name, data in images
                    ]
                    resp = client.post(
                        f"{self.base_url}/predict",
                        data={"profile": _json.dumps(profile)},
                        files=files,
                    )
                else:
                    resp = client.post(
                        f"{self.base_url}/predict", json=profile
                    )

            if resp.status_code == 422:
                body = resp.json()
                return _make_error_dict(
                    422,
                    body.get("message", "Invalid input."),
                    details=body.get("details", body.get("detail")),
                )
            if resp.status_code >= 500:
                body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
                return _make_error_dict(
                    resp.status_code,
                    body.get("message", f"Server error ({resp.status_code})."),
                )
            if resp.status_code >= 400:
                body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
                return _make_error_dict(
                    resp.status_code,
                    body.get("message", f"Request error ({resp.status_code})."),
                    details=body.get("details"),
                )

            return resp.json()

        except (httpx.ConnectError, httpx.ConnectTimeout, ConnectionError):
            st.session_state["connection_error"] = True
            logger.warning("Connection error when calling POST /predict")
            return _make_error_dict(0, "Unable to connect to the prediction server.")

        except httpx.TimeoutException:
            return _make_error_dict(0, "The prediction request timed out. Please try again.")

        except Exception as exc:
            logger.exception("Unexpected error in predict()")
            return _make_error_dict(0, f"Unexpected error: {exc}")

    # ------------------------------------------------------------------
    # Status polling
    # ------------------------------------------------------------------

    def get_status(self, session_id: str) -> dict:
        """GET /predict/{session_id}/status -- returns PredictionStatusResponse or error dict."""
        try:
            with httpx.Client(timeout=STATUS_TIMEOUT) as client:
                resp = client.get(f"{self.base_url}/predict/{session_id}/status")

            if resp.status_code >= 400:
                body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
                return _make_error_dict(
                    resp.status_code,
                    body.get("message", body.get("detail", f"Error ({resp.status_code}).")),
                )

            return resp.json()

        except (httpx.ConnectError, httpx.ConnectTimeout, ConnectionError):
            st.session_state["connection_error"] = True
            logger.warning("Connection error when polling status for %s", session_id)
            return _make_error_dict(0, "Lost connection to the prediction server.")

        except Exception as exc:
            logger.exception("Unexpected error in get_status()")
            return _make_error_dict(0, f"Unexpected error: {exc}")

    # ------------------------------------------------------------------
    # Result (direct fetch)
    # ------------------------------------------------------------------

    def get_result(self, session_id: str) -> dict:
        """GET /predict/{session_id}/result -- returns completed result or error dict."""
        try:
            with httpx.Client(timeout=STATUS_TIMEOUT) as client:
                resp = client.get(f"{self.base_url}/predict/{session_id}/result")

            if resp.status_code >= 400:
                body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
                return _make_error_dict(
                    resp.status_code,
                    body.get("message", body.get("detail", f"Error ({resp.status_code}).")),
                )

            return resp.json()

        except (httpx.ConnectError, httpx.ConnectTimeout, ConnectionError):
            st.session_state["connection_error"] = True
            logger.warning("Connection error when fetching result for %s", session_id)
            return _make_error_dict(0, "Lost connection to the prediction server.")

        except Exception as exc:
            logger.exception("Unexpected error in get_result()")
            return _make_error_dict(0, f"Unexpected error: {exc}")

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def health(self) -> dict:
        """GET /health -- returns HealthResponse or error dict."""
        try:
            with httpx.Client(timeout=HEALTH_TIMEOUT) as client:
                resp = client.get(f"{self.base_url}/health")

            if resp.status_code >= 400:
                return _make_error_dict(resp.status_code, "Health check returned an error.")

            return resp.json()

        except (httpx.ConnectError, httpx.ConnectTimeout, ConnectionError):
            st.session_state["connection_error"] = True
            return _make_error_dict(0, "Unable to reach the API server.")

        except Exception as exc:
            logger.exception("Unexpected error in health()")
            return _make_error_dict(0, f"Health check failed: {exc}")

    # ------------------------------------------------------------------
    # Model info
    # ------------------------------------------------------------------

    def model_info(self) -> dict:
        """GET /health/model -- returns ModelInfoResponse or error dict."""
        try:
            with httpx.Client(timeout=HEALTH_TIMEOUT) as client:
                resp = client.get(f"{self.base_url}/health/model")

            if resp.status_code >= 400:
                return _make_error_dict(resp.status_code, "Model info request failed.")

            return resp.json()

        except (httpx.ConnectError, httpx.ConnectTimeout, ConnectionError):
            st.session_state["connection_error"] = True
            return _make_error_dict(0, "Unable to reach the API server.")

        except Exception as exc:
            logger.exception("Unexpected error in model_info()")
            return _make_error_dict(0, f"Model info failed: {exc}")

    # ------------------------------------------------------------------
    # Explore
    # ------------------------------------------------------------------

    def explore_features(self) -> dict:
        """GET /explore/features -- returns available features list or error dict."""
        try:
            with httpx.Client(timeout=EXPLORE_TIMEOUT) as client:
                resp = client.get(f"{self.base_url}/explore/features")

            if resp.status_code >= 400:
                return _make_error_dict(resp.status_code, "Failed to fetch features.")

            return resp.json()

        except (httpx.ConnectError, httpx.ConnectTimeout, ConnectionError):
            st.session_state["connection_error"] = True
            return _make_error_dict(0, "Unable to reach the API server.")

        except Exception as exc:
            logger.exception("Unexpected error in explore_features()")
            return _make_error_dict(0, f"Unexpected error: {exc}")

    def explore_distributions(self, feature: str, color_by_class: bool = False) -> dict:
        """GET /explore/distributions -- returns distribution data or error dict."""
        try:
            params = {"feature": feature, "color_by_class": color_by_class}
            with httpx.Client(timeout=EXPLORE_TIMEOUT) as client:
                resp = client.get(
                    f"{self.base_url}/explore/distributions", params=params
                )

            if resp.status_code >= 400:
                body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
                return _make_error_dict(
                    resp.status_code,
                    body.get("detail", f"Error ({resp.status_code})."),
                )

            return resp.json()

        except (httpx.ConnectError, httpx.ConnectTimeout, ConnectionError):
            st.session_state["connection_error"] = True
            return _make_error_dict(0, "Unable to reach the API server.")

        except Exception as exc:
            logger.exception("Unexpected error in explore_distributions()")
            return _make_error_dict(0, f"Unexpected error: {exc}")

    def explore_performance(self) -> dict:
        """GET /explore/performance -- returns model performance data or error dict."""
        try:
            with httpx.Client(timeout=EXPLORE_TIMEOUT) as client:
                resp = client.get(f"{self.base_url}/explore/performance")

            if resp.status_code >= 400:
                return _make_error_dict(resp.status_code, "Failed to fetch performance data.")

            return resp.json()

        except (httpx.ConnectError, httpx.ConnectTimeout, ConnectionError):
            st.session_state["connection_error"] = True
            return _make_error_dict(0, "Unable to reach the API server.")

        except Exception as exc:
            logger.exception("Unexpected error in explore_performance()")
            return _make_error_dict(0, f"Unexpected error: {exc}")

    def explore_patterns(self) -> dict:
        """GET /explore/patterns -- returns adoption pattern data or error dict."""
        try:
            with httpx.Client(timeout=EXPLORE_TIMEOUT) as client:
                resp = client.get(f"{self.base_url}/explore/patterns")

            if resp.status_code >= 400:
                return _make_error_dict(resp.status_code, "Failed to fetch patterns data.")

            return resp.json()

        except (httpx.ConnectError, httpx.ConnectTimeout, ConnectionError):
            st.session_state["connection_error"] = True
            return _make_error_dict(0, "Unable to reach the API server.")

        except Exception as exc:
            logger.exception("Unexpected error in explore_patterns()")
            return _make_error_dict(0, f"Unexpected error: {exc}")

    # ------------------------------------------------------------------
    # Recent predictions
    # ------------------------------------------------------------------

    def recent_predictions(self, limit: int = 20) -> dict:
        """GET /predictions/recent -- returns recent predictions or error dict."""
        try:
            with httpx.Client(timeout=HEALTH_TIMEOUT) as client:
                resp = client.get(
                    f"{self.base_url}/predictions/recent",
                    params={"limit": limit},
                )

            if resp.status_code >= 400:
                return _make_error_dict(resp.status_code, "Failed to fetch predictions.")

            return resp.json()

        except (httpx.ConnectError, httpx.ConnectTimeout, ConnectionError):
            st.session_state["connection_error"] = True
            return _make_error_dict(0, "Unable to reach the API server.")

        except Exception as exc:
            logger.exception("Unexpected error in recent_predictions()")
            return _make_error_dict(0, f"Unexpected error: {exc}")
