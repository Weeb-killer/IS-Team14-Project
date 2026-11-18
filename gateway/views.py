import logging
import time
import uuid

from django.conf import settings
from django.core.exceptions import DisallowedHost, RequestDataTooBig
from django.core.paginator import Paginator
from django.db import DatabaseError
from django.db.models import Count, Q
from django.http import HttpResponseRedirect, JsonResponse
from django.shortcuts import get_object_or_404, render
from django.urls import reverse
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt, csrf_protect
from django.views.decorators.http import require_POST, require_safe

from .evidence_store import cache_stats, clear_evidence, save_evidence
from .inspectors import get_inspectors, unknown
from .models import RequestRecord
from .policy import decision
from .presentation import evidence_context
from .proxy import ProxyResponse, UpstreamFailure, forward
from .request import from_django

logger = logging.getLogger(__name__)


def error_response(reason, status, request_id):
    response = JsonResponse({"error": reason, "request_id": str(request_id)}, status=status)
    response["Cache-Control"] = "no-store"
    return response


@csrf_exempt
def ingest(request):
    started = time.monotonic()
    request_id = uuid.uuid4()
    # Establish an audit row before any upstream side effect. No raw body/headers/query stored.
    record = RequestRecord(
        id=request_id,
        received_at=timezone.now(),
        method=request.method[:64],
        path=request.path[:320],
        body_bytes=0,
        mode=settings.GATEWAY_MODE,
        action="processing",
        reason="pending",
        http_status=0,
        duration_ms=0,
        evidence={},
    )
    try:
        record.save(force_insert=True)
    except DatabaseError:
        logger.exception("Cannot create audit record %s", request_id)
        response = error_response("audit_unavailable", 503, request_id)
        response["X-SecBERT-Request-ID"] = str(request_id)
        response["X-SecBERT-Mode"] = settings.GATEWAY_MODE
        response["X-SecBERT-Action"] = "error"
        response["X-SecBERT-Audit"] = "unavailable"
        return response

    evidence = {"model": unknown("not_inspected"), "crs": unknown("not_inspected")}

    def finish(response, action, reason):
        record.action, record.reason = action, reason
        record.http_status = response.status_code
        record.duration_ms = max(0, round((time.monotonic() - started) * 1000))
        record.evidence = evidence
        model, crs = evidence["model"], evidence["crs"]
        record.disagreement = (
            model["would_block"] != crs["would_block"]
            if model.get("status") == crs.get("status") == "complete"
            and type(model.get("would_block")) is bool
            and type(crs.get("would_block")) is bool
            else None
        )
        try:
            record.save(
                force_update=True,
                update_fields=[
                    "method",
                    "path",
                    "body_bytes",
                    "action",
                    "reason",
                    "http_status",
                    "duration_ms",
                    "evidence",
                    "disagreement",
                ],
            )
            response["X-SecBERT-Audit"] = "complete"
        except DatabaseError:
            logger.exception("Cannot finalize audit record %s", request_id)
            # Preserve a response after an upstream side effect; the pending row remains visible.
            response["X-SecBERT-Audit"] = "incomplete"
        try:
            save_evidence(request_id, evidence)
            response["X-SecBERT-Evidence"] = "saved"
        except (OSError, ValueError):
            logger.exception("Cannot save raw JSON evidence for request %s", request_id)
            response["X-SecBERT-Evidence"] = "unavailable"
        response["X-SecBERT-Request-ID"] = str(request_id)
        response["X-SecBERT-Action"] = action
        response["X-SecBERT-Mode"] = settings.GATEWAY_MODE
        return response

    def reject(reason, status, action="rejected"):
        return finish(error_response(reason, status, request_id), action, reason)

    try:
        request.get_host()  # Django does not validate ALLOWED_HOSTS until explicitly accessed.
        captured = from_django(request)
    except DisallowedHost:
        return reject("invalid_host", 400)
    except RequestDataTooBig:
        return reject("request_body_too_large", 413)
    except (UnicodeError, ValueError, OSError):
        return reject("invalid_request", 400)
    record.method = captured.method.decode("ascii")[:64]
    record.path = captured.display_path
    record.body_bytes = len(captured.body)
    if rejection := captured.rejection():
        return reject(rejection[1], rejection[0])

    model, crs = get_inspectors().inspect(captured)
    evidence.update(model=model, crs=crs)
    action, reason, status = decision(model, settings.GATEWAY_MODE, settings.MODEL_FAILURE_POLICY)
    if status:
        return reject(reason, status, action)
    try:
        upstream = forward(captured)
        response = ProxyResponse(upstream)
    except UpstreamFailure as exc:
        return reject(exc.reason, exc.status, "error")
    return finish(response, action, reason)


def runtime_status():
    return {
        "mode": settings.GATEWAY_MODE,
        "model_failure_policy": settings.MODEL_FAILURE_POLICY,
        "upstream": settings.GATEWAY_UPSTREAM,
        "checkpoint_configured": bool(settings.SECBERT_CHECKPOINT),
        "probe_bank_configured": bool(settings.SECBERT_PROBE_BANK),
        "crs_configured": bool(settings.OWASP_CRS),
        "device": settings.SECBERT_DEVICE,
    }


def filtered_records(request):
    records = RequestRecord.objects.all()
    action = request.GET.get("action", "")
    if action in {"forwarded", "blocked", "rejected", "error", "processing"}:
        records = records.filter(action=action)
    if request.GET.get("disagreement") == "1":
        records = records.filter(disagreement=True)
    review = request.GET.get("review", "")
    if review == "attack":
        records = records.filter(human_is_attack=True)
    elif review == "benign":
        records = records.filter(human_is_attack=False)
    elif review == "unreviewed":
        records = records.filter(mode="observe", human_is_attack__isnull=True)
    return records


def serialize_record(record, detail=False):
    data = {
        "id": str(record.id),
        "received_at": record.received_at.isoformat(),
        "method": record.method,
        "path": record.path,
        "body_bytes": record.body_bytes,
        "mode": record.mode,
        "action": record.action,
        "reason": record.reason,
        "http_status": record.http_status,
        "duration_ms": record.duration_ms,
        "disagreement": record.disagreement,
        "human_is_attack": record.human_is_attack,
        "human_reviewed_at": record.human_reviewed_at.isoformat()
        if record.human_reviewed_at
        else None,
    }
    if detail:
        data["evidence"] = record.evidence
    return data


@require_safe
def request_list(request):
    records = filtered_records(request)
    page = Paginator(records, 50).get_page(request.GET.get("page"))
    stats = RequestRecord.objects.aggregate(
        total=Count("id"),
        blocked=Count("id", filter=Q(action="blocked")),
        forwarded=Count("id", filter=Q(action="forwarded")),
        disagreements=Count("id", filter=Q(disagreement=True)),
    )
    params = request.GET.copy()
    params.pop("page", None)
    cache_result = None
    if request.GET.get("cache_error") == "unavailable":
        cache_result = {"error": "unavailable"}
    elif all(
        value is not None and value.isascii() and value.isdecimal() and len(value) <= 9
        for value in (request.GET.get("cache_cleared"), request.GET.get("cache_failed"))
    ):
        cache_result = {
            "deleted_count": int(request.GET["cache_cleared"]),
            "failed_count": int(request.GET["cache_failed"]),
        }
    for key in ("cache_cleared", "cache_failed", "cache_error"):
        params.pop(key, None)
    return render(
        request,
        "gateway/list.html",
        {
            "page": page,
            "stats": stats,
            "runtime": runtime_status(),
            "filters": params.urlencode(),
            "cache": cache_stats(),
            "cache_result": cache_result,
        },
    )


@require_safe
def request_detail(request, request_id):
    record = get_object_or_404(RequestRecord, pk=request_id)
    return detail_response(request, record)


def detail_response(request, record, *, error="", status=200):
    evidence_view = evidence_context(record.evidence, mode=record.mode)
    verdict = record.human_is_attack
    label = "Awaiting review"
    tone = "muted"
    comparison = "Choose whether this request is an attack, then save your review."
    if record.mode != "observe":
        label = "Automatic enforcement"
        comparison = "The automated inspection policy determines how this request is handled."
    elif record.action == "processing":
        label = "Waiting for inspection"
        comparison = "Review the complete evidence after the request has finished processing."
    if verdict is not None:
        label = "Marked as attack" if verdict else "Marked as non-attack"
        tone = "danger" if verdict else "success"
        model = evidence_view["model"]
        comparison = (
            (
                "Your review matches the model's decision."
                if verdict == model["would_block"]
                else "Your review does not match the model's decision. Review the evidence below."
            )
            if model["status"] == "complete"
            else "The model has no result available. Your review is saved independently."
        )
    return render(
        request,
        "gateway/detail.html",
        {
            "record": record,
            "evidence_view": evidence_view,
            "human_review": {
                "allowed": record.mode == "observe" and record.action != "processing",
                "label": label,
                "tone": tone,
                "comparison": comparison,
                "saved": not error
                and request.method in {"GET", "HEAD"}
                and request.GET.get("review_saved") == "1",
                "error": error,
            },
        },
        status=status,
    )


@require_POST
@csrf_protect
def review_request(request, request_id):
    record = get_object_or_404(RequestRecord, pk=request_id)
    if record.mode != "observe":
        return detail_response(
            request,
            record,
            error="Human review is available only for observe-mode requests.",
            status=409,
        )
    if record.action == "processing":
        return detail_response(
            request,
            record,
            error="The request is still processing. Wait for inspection to finish.",
            status=409,
        )
    values = request.POST.getlist("verdict")
    choices = {"attack": True, "benign": False, "unreviewed": None}
    if len(values) != 1 or values[0] not in choices:
        return detail_response(
            request, record, error="Select exactly one valid review option.", status=400
        )
    verdict = choices[values[0]]
    try:
        updated = (
            RequestRecord.objects.filter(pk=record.id, mode="observe")
            .exclude(action="processing")
            .update(
                human_is_attack=verdict,
                human_reviewed_at=timezone.now() if verdict is not None else None,
            )
        )
    except DatabaseError:
        logger.exception("Cannot save human review for request %s", record.id)
        return detail_response(
            request, record, error="Your review could not be saved. Try again later.", status=503
        )
    if not updated:
        record = get_object_or_404(RequestRecord, pk=request_id)
        return detail_response(
            request,
            record,
            error="The request status has changed. Refresh the page and try again.",
            status=409,
        )
    return HttpResponseRedirect(
        reverse("request_detail", kwargs={"request_id": record.id}) + "?review_saved=1",
        status=303,
    )


@require_POST
@csrf_protect
def clear_evidence_cache(request):
    try:
        result = clear_evidence()
        query = f"?cache_cleared={result['deleted_count']}&cache_failed={result['failed_count']}"
    except OSError:
        logger.exception("Cannot clear raw JSON evidence cache")
        query = "?cache_error=unavailable"
    return HttpResponseRedirect(reverse("request_list") + query, status=303)


@require_safe
def api_status(request):
    # Configuration status is intentionally distinct from successful live inference.
    return JsonResponse(runtime_status())


@require_safe
def api_requests(request):
    page = Paginator(filtered_records(request), 50).get_page(request.GET.get("page"))
    return JsonResponse(
        {
            "count": page.paginator.count,
            "page": page.number,
            "pages": page.paginator.num_pages,
            "results": [serialize_record(record) for record in page],
        }
    )


@require_safe
def api_request(request, request_id):
    return JsonResponse(
        serialize_record(get_object_or_404(RequestRecord, pk=request_id), detail=True)
    )
