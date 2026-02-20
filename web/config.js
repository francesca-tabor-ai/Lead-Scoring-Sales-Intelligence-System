/**
 * GapFinder — Runtime configuration
 *
 * Set GAPFINDER_API_URL to point the dashboard at your live API.
 * This file is served as static JS by Vercel alongside index.html.
 *
 * Deployment options:
 *   • Railway:  https://your-app.up.railway.app
 *   • Render:   https://gapfinder-api.onrender.com
 *   • AWS:      https://xxxxxxxx.lambda-url.us-east-1.on.aws
 *   • Local:    http://localhost:8000  (auto-detected, no need to set)
 *
 * To update: change the URL below and redeploy the frontend.
 */

// Leave as empty string to auto-detect (localhost in dev, same origin in prod).
window.GAPFINDER_API_URL = '';
