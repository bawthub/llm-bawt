"""BawtHub's voice service tells open UIs its cloning setup moved (TASK-2664).

The voice service is a separate process, so it marks the topic over HTTP. The
request carries nothing: the UI refetches the cloning status itself. The
watcher coalesces marks to at most one event per second.
"""

from fastapi import APIRouter

from ..dependencies import get_service
from ..state_events import VOICE_CLONING, mark

router = APIRouter()


@router.post("/v1/state/voice-cloning", status_code=204)
def voice_cloning_changed() -> None:
    try:
        mark(get_service(), VOICE_CLONING)
    except RuntimeError:  # service not initialized (tests, CLI use)
        pass
