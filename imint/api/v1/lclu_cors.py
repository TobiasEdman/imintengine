"""Optional, exact-origin CORS only for the annual-map API."""
from urllib.parse import urlsplit

from starlette.middleware.cors import CORSMiddleware


def validate_origins(origins: list[str]) -> list[str]:
    for origin in origins:
        value = urlsplit(origin)
        if (value.scheme != 'https' or not value.hostname or value.username or value.password
                or value.query or value.fragment or value.path or '*' in origin):
            raise ValueError('Showcase origins must be exact HTTPS origins without paths')
    return origins


class LcluCORSMiddleware:
    def __init__(self, app, origins: list[str]):
        self.app = app
        self.cors = CORSMiddleware(app, allow_origins=validate_origins(origins),
                                   allow_methods=['GET', 'POST'], allow_headers=['Content-Type'],
                                   expose_headers=['Content-Disposition'], allow_credentials=False)

    async def __call__(self, scope, receive, send):
        path = scope.get('path', '')
        target = self.cors if scope['type'] == 'http' and (path == '/v1/lclu' or path.startswith('/v1/lclu/')) else self.app
        await target(scope, receive, send)
