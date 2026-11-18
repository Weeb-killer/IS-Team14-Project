def decision(model, mode, failure_policy):
    """CRS and concept probes provide evidence; only SecBERT controls enforcement."""
    complete = model.get("status") == "complete" and type(model.get("would_block")) is bool
    if mode == "observe":
        return "forwarded", "observe" if complete else "observe_model_unavailable", None
    if not complete:
        if failure_policy == "closed":
            return "blocked", "model_unavailable", 503
        return "forwarded", "model_failure_open", None
    if model["would_block"]:
        return "blocked", "model_attack", 403
    return "forwarded", "model_pass", None
