"""The measured local-video envelope; expansion requires new evidence.

Calibration measures one 480p, 5-second render. Every shape in the envelope
denoises at most the calibrated token count: 9:16 is the 16:9 transpose
(832x480 pixels either way) and 1:1 is 512x512 (fewer). Image conditioning
adds one tiled VAE encode of the source frame to the same pipeline.
"""
CALIBRATION_PROFILE = {"resolution": "480p", "aspect_ratios": ["16:9", "9:16", "1:1"], "duration": 5, "num_outputs": 1}


def video_profile(request) -> dict:
    return {"resolution": request.resolution, "aspect_ratio": request.aspect_ratio,
            "duration": request.duration or 5, "num_outputs": getattr(request, "num_outputs", 1),
            "image_conditioned": bool(request.source_image)}


def require_calibration_profile(profile: dict) -> None:
    if (profile.get("resolution") != CALIBRATION_PROFILE["resolution"]
            or profile.get("aspect_ratio") not in CALIBRATION_PROFILE["aspect_ratios"]
            or profile.get("duration") != CALIBRATION_PROFILE["duration"]
            or profile.get("num_outputs") != CALIBRATION_PROFILE["num_outputs"]
            or type(profile.get("image_conditioned")) is not bool):
        raise ValueError("Local video supports only its calibrated profile: 480p, "
                         f"{', '.join(CALIBRATION_PROFILE['aspect_ratios'])}, 5 seconds, one output")
