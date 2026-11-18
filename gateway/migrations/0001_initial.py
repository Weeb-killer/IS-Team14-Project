from django.db import migrations, models


class Migration(migrations.Migration):
    initial = True
    dependencies = []
    operations = [
        migrations.CreateModel(
            name="RequestRecord",
            fields=[
                ("id", models.UUIDField(editable=False, primary_key=True, serialize=False)),
                ("received_at", models.DateTimeField(db_index=True)),
                ("method", models.CharField(max_length=64)),
                ("path", models.CharField(max_length=320)),
                ("body_bytes", models.PositiveIntegerField()),
                ("mode", models.CharField(max_length=12)),
                ("action", models.CharField(db_index=True, max_length=16)),
                ("reason", models.CharField(max_length=50)),
                ("http_status", models.PositiveSmallIntegerField()),
                ("duration_ms", models.PositiveIntegerField()),
                ("disagreement", models.BooleanField(db_index=True, null=True)),
                ("evidence", models.JSONField()),
            ],
            options={"ordering": ("-received_at", "-id")},
        ),
    ]
