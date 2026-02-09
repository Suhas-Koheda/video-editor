from posthog import Posthog
import os

# Get API key from environment or use placeholder
POSTHOG_KEY = os.environ.get("POSTHOG_API_KEY", "phc_sCjB9GVPTUw1v1wxa57neDkTyX2i16YVjsR0Jk8uEwp")
POSTHOG_HOST = "https://us.i.posthog.com"

ph = None
try:
    ph = Posthog(
        project_api_key=POSTHOG_KEY,
        host=POSTHOG_HOST,
    )
except Exception as e:
    pass
def track(event, data=None):
    if ph:
        try:
            ph.capture(
                distinct_id="anonymous_user",
                event=event,
                properties=data or {}
            )
        except Exception as e:
            pass

def shutdown_tracker():
    if ph:
        try:
            ph.shutdown()
        except:
            pass