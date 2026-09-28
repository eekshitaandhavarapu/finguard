---
name: Replit Streamlit deployment
description: Replit-specific requirements for publishing the FinGuard Streamlit dashboard.
---

Replit Streamlit autoscale deployments need an explicit headless run command bound to `0.0.0.0:5000`, with CORS and WebSocket compression disabled, plus a pinned `requirements.txt`.

**Why:** Without headless mode and the explicit server settings, Streamlit can stop at its first-run onboarding prompt or fail to expose the port required by the Replit preview and publisher.

**How to apply:** Keep the validated `.replit` run command and `requirements.txt` in sync when changing the dashboard runtime or dependencies. Production publishing still requires the workspace owner to confirm Publish.