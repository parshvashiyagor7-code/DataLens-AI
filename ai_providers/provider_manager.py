"""Provider-independent text generation for DataLens AI.

Configuration is supplied through environment variables. Streamlit's page
passes secrets to the legacy subprocess without writing them to disk.
"""
from __future__ import annotations

import os
from typing import Any

import requests


GEMINI_DEFAULT_MODEL = "gemini-3.8-flash"
GROQ_DEFAULT_MODEL = "openai/gpt-oss-20b"
OLLAMA_DEFAULT_MODEL = "llama3.1"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

_LAST_PROVIDER: str | None = None


class ProviderError(RuntimeError):
    """A provider request failed; message is safe to show without credentials."""


def get_last_provider() -> str | None:
    return _LAST_PROVIDER


def _env(name: str, default: str = "") -> str:
    return str(os.environ.get(name, default) or "").strip()


def _provider_order() -> list[str]:
    selected = _env("AI_PROVIDER", "auto").lower() or "auto"
    raw_order = _env("AI_PROVIDER_ORDER", "gemini,groq")
    configured = [item.strip().lower() for item in raw_order.split(",") if item.strip()]
    if selected != "auto":
        if selected not in {"gemini", "groq", "ollama"}:
            raise ProviderError(
                f"Unsupported AI_PROVIDER '{selected}'. Choose auto, gemini, groq, or ollama."
            )
        configured = [selected] + [item for item in configured if item != selected]
    if not configured:
        configured = ["gemini", "groq"]
    result: list[str] = []
    for provider in configured:
        if provider in {"gemini", "groq", "ollama"} and provider not in result:
            result.append(provider)
    if not result:
        raise ProviderError("AI_PROVIDER_ORDER contains no supported provider names.")
    return result


def _is_configured(provider: str, selected: str) -> bool:
    if provider == "gemini":
        return bool(_env("GEMINI_API_KEY"))
    if provider == "groq":
        return bool(_env("GROQ_API_KEY"))
    if provider == "ollama":
        # On local development, selecting Ollama implies the default localhost URL.
        # In auto/fallback mode require an explicit URL so cloud deployments do not
        # accidentally try the Streamlit container's own localhost.
        return selected == "ollama" or bool(_env("OLLAMA_BASE_URL"))
    return False


def _timeout() -> float:
    try:
        value = float(_env("AI_REQUEST_TIMEOUT_SECONDS", "90"))
    except ValueError:
        value = 90.0
    return max(10.0, min(value, 300.0))


def _response_json(response: requests.Response, provider: str) -> dict[str, Any]:
    if not response.ok:
        message = ""
        try:
            body = response.json()
            error = body.get("error", body) if isinstance(body, dict) else body
            message = str(error)
        except Exception:
            message = response.text[:500]
        raise ProviderError(f"{provider} returned HTTP {response.status_code}: {message}")
    try:
        data = response.json()
    except ValueError as exc:
        raise ProviderError(f"{provider} returned invalid JSON.") from exc
    if not isinstance(data, dict):
        raise ProviderError(f"{provider} returned an unexpected response format.")
    return data


def _generate_gemini(prompt: str) -> str:
    api_key = _env("GEMINI_API_KEY")
    if not api_key:
        raise ProviderError("GEMINI_API_KEY is not configured.")
    model = _env("GEMINI_MODEL", GEMINI_DEFAULT_MODEL)
    url = GEMINI_URL.format(model=model)
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0, "maxOutputTokens": 1200},
    }
    response = requests.post(
        url,
        headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
        json=payload,
        timeout=_timeout(),
    )
    data = _response_json(response, "Gemini")
    candidates = data.get("candidates") or []
    if not candidates:
        feedback = data.get("promptFeedback") or {}
        reason = feedback.get("blockReason") or "No candidate returned"
        raise ProviderError(f"Gemini returned no text candidate: {reason}.")
    parts = ((candidates[0].get("content") or {}).get("parts") or [])
    text = "\n".join(str(part.get("text", "")) for part in parts if part.get("text"))
    if not text.strip():
        raise ProviderError("Gemini returned an empty response.")
    return text.strip()


def _generate_groq(prompt: str) -> str:
    api_key = _env("GROQ_API_KEY")
    if not api_key:
        raise ProviderError("GROQ_API_KEY is not configured.")
    model = _env("GROQ_MODEL", GROQ_DEFAULT_MODEL)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 1200,
    }
    response = requests.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=_timeout(),
    )
    data = _response_json(response, "Groq")
    choices = data.get("choices") or []
    if not choices:
        raise ProviderError("Groq returned no completion choices.")
    message = choices[0].get("message") or {}
    text = message.get("content")
    if isinstance(text, list):
        text = "\n".join(str(part.get("text", "")) for part in text if isinstance(part, dict))
    if not isinstance(text, str) or not text.strip():
        raise ProviderError("Groq returned an empty response.")
    return text.strip()


def _generate_ollama(prompt: str) -> str:
    base_url = _env("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    model = _env("OLLAMA_MODEL", OLLAMA_DEFAULT_MODEL)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0, "num_predict": 1200},
    }
    response = requests.post(
        f"{base_url}/api/chat",
        json=payload,
        timeout=_timeout(),
    )
    data = _response_json(response, "Ollama")
    message = data.get("message") or {}
    text = message.get("content")
    if not isinstance(text, str) or not text.strip():
        raise ProviderError("Ollama returned an empty response. Check that the configured model is pulled.")
    return text.strip()


_PROVIDER_CALLS = {
    "gemini": _generate_gemini,
    "groq": _generate_groq,
    "ollama": _generate_ollama,
}


def generate_text(prompt: str) -> str:
    """Generate text using the preferred provider and fall back in configured order.

    Quota (429), temporary server (5xx), timeout and connection errors trigger
    failover to the next configured provider. Invalid keys also fail over, while
    the final error retains a short per-provider explanation.
    """
    global _LAST_PROVIDER
    _LAST_PROVIDER = None
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("AI prompt must be a non-empty string.")

    selected = _env("AI_PROVIDER", "auto").lower() or "auto"
    order = _provider_order()
    failures: list[str] = []
    attempted = 0

    for provider in order:
        if not _is_configured(provider, selected):
            if provider == "gemini":
                failures.append("gemini: skipped (GEMINI_API_KEY missing)")
            elif provider == "groq":
                failures.append("groq: skipped (GROQ_API_KEY missing)")
            else:
                failures.append("ollama: skipped (set OLLAMA_BASE_URL for fallback use)")
            continue

        attempted += 1
        print(f"[DataLens AI] Trying provider: {provider}")
        try:
            result = _PROVIDER_CALLS[provider](prompt)
            _LAST_PROVIDER = provider
            return result
        except (requests.Timeout, requests.ConnectionError) as exc:
            failures.append(f"{provider}: network/timeout error ({type(exc).__name__})")
        except ProviderError as exc:
            failures.append(f"{provider}: {exc}")
        except requests.RequestException as exc:
            failures.append(f"{provider}: request error ({type(exc).__name__})")
        except Exception as exc:
            # A provider-specific unexpected failure must not prevent trying the
            # next configured provider, but its details remain visible in logs.
            failures.append(f"{provider}: unexpected error ({type(exc).__name__}: {exc})")

        print(f"[DataLens AI] Provider unavailable; checking next configured provider.")

    if attempted == 0:
        raise ProviderError(
            "No configured AI provider is ready. Add GEMINI_API_KEY or GROQ_API_KEY; "
            "for local Ollama set AI_PROVIDER='ollama' and ensure the Ollama server/model is running."
        )
    details = " | ".join(failures)
    raise ProviderError(
        "All configured AI providers failed. Check provider quota, model access, API keys, "
        f"and network connectivity. Details: {details}"
    )
