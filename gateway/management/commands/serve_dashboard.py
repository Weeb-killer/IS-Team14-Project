from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.core.wsgi import get_wsgi_application


class Command(BaseCommand):
    help = "Serve the local request dashboard with raw JSON cache cleanup (port 8081)."

    def add_arguments(self, parser):
        parser.add_argument("--listen", default="127.0.0.1:8081")

    def handle(self, *args, **options):
        from waitress import serve

        if settings.ROOT_URLCONF != "secbert_app.dashboard_urls":
            raise CommandError("Pass --settings=secbert_app.dashboard_settings.")
        if any(
            part.rsplit(":", 1)[0] not in {"127.0.0.1", "[::1]", "localhost"}
            for part in options["listen"].split()
        ):
            raise CommandError("The unauthenticated dashboard must listen on loopback only.")
        serve(get_wsgi_application(), listen=options["listen"], max_request_body_size=16 * 1024)
