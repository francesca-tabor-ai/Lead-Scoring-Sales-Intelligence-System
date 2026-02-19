"""
Lambda entrypoint for FastAPI (container image deployment).
Use this when deploying as a Lambda container image (up to 10 GB).
Avoids the 500 MB zip/ephemeral storage limit by using an image instead of a zip.
"""
from mangum import Mangum
from api.main import app

handler = Mangum(app, lifespan="on")
