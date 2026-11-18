from ipaddress import ip_address

from django.core.exceptions import DisallowedHost
from django.http import HttpResponse


class LocalDashboardMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        try:
            request.get_host()
            address = ip_address(request.META.get("REMOTE_ADDR", ""))
            if not address.is_loopback:
                return HttpResponse("Local dashboard only.", status=403)
        except (DisallowedHost, ValueError):
            return HttpResponse("Invalid dashboard host or peer.", status=400)
        response = self.get_response(request)
        response["Cache-Control"] = "no-store"
        response["X-Content-Type-Options"] = "nosniff"
        response["X-Frame-Options"] = "DENY"
        response["Content-Security-Policy"] = (
            "default-src 'none'; style-src 'unsafe-inline'; frame-ancestors 'none'; "
            "base-uri 'none'; form-action 'self'"
        )
        return response
