"""WSGI entrypoint.

Production:

    gunicorn healthscan.wsgi:app --bind 0.0.0.0:5000

Local development:

    python -m healthscan.wsgi
"""

from __future__ import annotations

import logging
import os

from healthscan.api import create_app

logging.basicConfig(
    level=os.environ.get("HEALTHSCAN_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)

app = create_app()


def main() -> None:
    # Debug is opt-in: the Werkzeug debugger exposes an interactive console,
    # which is remote code execution on anything reachable from outside.
    debug = os.environ.get("FLASK_DEBUG", "").lower() in ("1", "true", "yes")
    app.run(
        host=os.environ.get("HEALTHSCAN_HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "5000")),
        debug=debug,
    )


if __name__ == "__main__":
    main()
