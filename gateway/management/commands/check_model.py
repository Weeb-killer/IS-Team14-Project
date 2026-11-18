"""Verify configured artifacts through the same isolated worker used by ingress."""

import json

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from gateway.inspectors import Inspectors
from gateway.request import Request


class Command(BaseCommand):
    help = (
        "Load the configured SecBERT/probes and inspect one request without forwarding or auditing."
    )

    def add_arguments(self, parser):
        parser.add_argument("--target", default="/", help="ASCII HTTP origin-form request target.")
        parser.add_argument("--json", action="store_true", help="Print full detection evidence.")

    def handle(self, *args, **options):
        try:
            target = options["target"].encode("ascii")
        except UnicodeError as exc:
            raise CommandError("Encode non-ASCII target characters with percent escapes.") from exc
        request = Request(
            method=b"GET",
            target=target,
            version="HTTP/1.1",
            headers=((b"Host", b"localhost"), (b"User-Agent", b"SecBERT-Check")),
            body=b"",
            peer=("127.0.0.1", 0),
            local=("127.0.0.1", 8080),
        )
        if rejected := request.rejection():
            raise CommandError(rejected[1])
        worker = Inspectors().model
        if worker is None:
            raise CommandError(
                "SECBERT_CHECKPOINT is empty. Load your environment configuration first."
            )
        try:
            result = worker.inspect(request)
        finally:
            worker.close()
        if result.get("status") != "complete":
            raise CommandError(
                f"SecBERT inspection failed: {result.get('error', 'unknown')}. See worker logs."
            )
        concepts = result.get("concepts", {})
        if settings.SECBERT_PROBE_BANK and concepts.get("status") != "complete":
            raise CommandError(f"Concept probes failed: {concepts.get('error', 'unknown')}.")
        if options["json"]:
            self.stdout.write(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        else:
            self.stdout.write(
                self.style.SUCCESS(
                    f"SecBERT OK: {len(result['scores'])} labels; "
                    f"{concepts.get('valid_count', 0)}/{concepts.get('total_count', 0)} valid probes; "
                    f"would_block={result['would_block']}; "
                    f"load + inference={result['duration_ms']} ms."
                )
            )
