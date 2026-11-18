from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.core.wsgi import get_wsgi_application

from gateway.inspectors import get_inspectors


class Command(BaseCommand):
    help = "Serve the Django HTTP ingress on a local port."

    def add_arguments(self, parser):
        parser.add_argument("--listen", default="127.0.0.1:8080")

    def handle(self, *args, **options):
        from waitress import serve

        if settings.ROOT_URLCONF != "secbert_app.ingress_urls":
            raise CommandError("Use --settings=secbert_app.settings for the ingress.")
        get_inspectors()
        self.stdout.write(
            f"SecBERT gateway: {settings.GATEWAY_MODE} -> {settings.GATEWAY_UPSTREAM}"
        )
        if not settings.SECBERT_CHECKPOINT:
            self.stdout.write(
                self.style.WARNING(
                    "SECBERT_CHECKPOINT is empty: detection is unavailable. "
                    "In PowerShell, run '. .\\env.ps1' in this terminal before starting the gateway."
                )
            )
        serve(
            get_wsgi_application(),
            listen=options["listen"],
            # Waitress rejects sizes >= its limit; Django allows exactly the configured cap.
            max_request_body_size=settings.GATEWAY_MAX_BODY + 1,
            clear_untrusted_proxy_headers=False,
        )
