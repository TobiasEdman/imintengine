/* Deployment setting. Empty uses the page's origin (local API/reverse proxy).
 * The static showcase calls the public annual-map API, which exposes only
 * /v1/lclu and allows only this showcase's origin (k8s/lclu-public.yaml).
 * No secrets here.
 */
window.IMINT_LCLU_CONFIG = {apiBase: 'https://lclu.icedc.se'};
