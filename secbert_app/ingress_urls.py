from django.urls import re_path

from gateway.views import ingest

urlpatterns = [re_path(r"(?s)^.*$", ingest)]
