"""The single supported calibration profile; expansion requires new evidence."""
CALIBRATION_PROFILE = {"resolution": "480p", "aspect_ratio": "16:9", "duration": 5, "num_outputs": 1}


def video_profile(request) -> dict:
    return {"resolution": request.resolution, "aspect_ratio": request.aspect_ratio,
            "duration": request.duration or 5, "num_outputs": getattr(request, "num_outputs", 1),
            "image_conditioned": bool(request.source_image)}


def require_calibration_profile(profile: dict) -> None:
    if profile != {**CALIBRATION_PROFILE, "image_conditioned": False}:
        raise ValueError("Local video currently supports only calibrated text-to-video: 480p, 16:9, 5 seconds, one output")
