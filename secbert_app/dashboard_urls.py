from django.urls import path

from gateway import views

urlpatterns = [
    path("", views.request_list, name="request_list"),
    path("cache/clear/", views.clear_evidence_cache, name="clear_evidence_cache"),
    path("requests/<uuid:request_id>/", views.request_detail, name="request_detail"),
    path("requests/<uuid:request_id>/review/", views.review_request, name="review_request"),
    path("api/status/", views.api_status, name="api_status"),
    path("api/requests/", views.api_requests, name="api_requests"),
    path("api/requests/<uuid:request_id>/", views.api_request, name="api_request"),
]
