from django.db import models


class RequestRecord(models.Model):
    id = models.UUIDField(primary_key=True, editable=False)
    received_at = models.DateTimeField(db_index=True)
    method = models.CharField(max_length=64)
    path = models.CharField(max_length=320)
    body_bytes = models.PositiveIntegerField()
    mode = models.CharField(max_length=12)
    action = models.CharField(max_length=16, db_index=True)
    reason = models.CharField(max_length=50)
    http_status = models.PositiveSmallIntegerField()
    duration_ms = models.PositiveIntegerField()
    disagreement = models.BooleanField(null=True, db_index=True)
    evidence = models.JSONField()
    human_is_attack = models.BooleanField(null=True, blank=True, db_index=True)
    human_reviewed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ("-received_at", "-id")
