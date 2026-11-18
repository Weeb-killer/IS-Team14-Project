from . import settings

# Use a separate host policy for the local dashboard.
globals().update({name: getattr(settings, name) for name in dir(settings) if name.isupper()})
ROOT_URLCONF = "secbert_app.dashboard_urls"
ALLOWED_HOSTS = settings.LOCAL_HOSTS
MIDDLEWARE = [
    "gateway.middleware.LocalDashboardMiddleware",
    "django.middleware.security.SecurityMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
]
CSRF_COOKIE_NAME = "secbert_dashboard_csrf"
SECURE_REFERRER_POLICY = "no-referrer"
