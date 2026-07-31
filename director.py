"""
Stage 0 — LLM Audio Director (GPT-4o).  docs/02 §1, docs/03 §1.

Takes plain narration `script_text` and produces a tag-annotated `directed_text` for expressive
`eleven_v3` delivery, WITHOUT altering wording. A content-preservation guardrail strips the
annotations back out and confirms they reduce to the original text; on mismatch it retries a
bounded number of times, then hard-rejects.

Pure Python (no Modal decorators) so the orchestrator can call it in-process on the
director_tts_image. A standalone Modal spike function for Phase 1 lives at the bottom.
"""

import json
import os
import re

import constants


class DirectorGuardrailError(Exception):
    """Raised when directed_text cannot be reduced back to clean_text within the retry budget."""

    stage = "director"


# All tags, flattened, for the guardrail's strip step.
_ALL_TAGS = {t for group in constants.DIRECTOR_TAGS.values() for t in group}


def _build_system_prompt(voice_character_hint: str | None) -> str:
    emotion = " ".join(f"[{t}]" for t in constants.DIRECTOR_TAGS["emotion"])
    delivery = " ".join(f"[{t}]" for t in constants.DIRECTOR_TAGS["delivery"])
    reactions = " ".join(f"[{t}]" for t in constants.DIRECTOR_TAGS["reactions"])
    pacing = " ".join(f"[{t}]" for t in constants.DIRECTOR_TAGS["pacing"])

    hint_line = (
        f"\nTarget voice character: {voice_character_hint}. Do not insert tags that contradict "
        f"it (e.g. no [shouts] for a hushed, calm voice).\n"
        if voice_character_hint
        else ""
    )

    return (
        "You are an audio director. You annotate narration scripts for expressive "
        "text-to-speech. You NEVER change, add, remove, or reorder words, facts, or sentence "
        "structure from the user's script — output must be word-for-word identical to the "
        "input except for inserted bracketed tags, capitalization changes for emphasis, and "
        "ellipses/punctuation adjustments for pacing.\n\n"
        "Use ONLY these tags, exactly as written, and never invent new ones:\n"
        f"  Emotion:   {emotion}\n"
        f"  Delivery:  {delivery}\n"
        f"  Reactions: {reactions}\n"
        f"  Pacing:    {pacing}\n\n"
        "Placement: insert a tag immediately before the word(s) it should affect. Each tag "
        "influences roughly the next 4-5 words before delivery reverts to neutral — do not "
        "expect one tag to carry a whole paragraph. Only tag where delivery genuinely shifts; "
        "do not over-tag every sentence.\n"
        "Use ALL CAPS for word-level emphasis, ellipses (...) for dramatic pauses/trailing "
        "off, and standard punctuation for pacing. Do NOT use SSML — eleven_v3 does not "
        "support <break/> or any SSML tags.\n"
        "Aim for natural, HUMAN, conversational delivery, not a flat announcer read: lean on "
        "commas and ellipses (...) for natural micro-pauses and breath, vary the rhythm, and "
        "place emotional tags only at genuine shifts in feeling. Mechanical, evenly-spaced tags "
        "sound robotic — under-tag rather than over-tag, and let punctuation carry most of the "
        "prosody. A relaxed, spoken cadence beats a dramatic one for a talking-head presenter.\n"
        f"{hint_line}"
        "Also decide stability_hint: 'creative' when you inserted a high density of tags "
        "(maximize expressiveness), 'natural' for lighter annotation.\n\n"
        "Return JSON with exactly: clean_text (verbatim pass-through of the input), "
        "directed_text (the tag-annotated version), stability_hint ('creative' or 'natural')."
    )


# OpenAI Structured Outputs schema (docs/02 §1).
_RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "directed_script",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "clean_text": {"type": "string"},
                "directed_text": {"type": "string"},
                "stability_hint": {"type": "string", "enum": ["creative", "natural"]},
            },
            "required": ["clean_text", "directed_text", "stability_hint"],
        },
    },
}


def _normalize(text: str) -> str:
    """Lowercase, collapse whitespace, drop non-alphanumerics — the comparison canonical form."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def strip_tags(directed_text: str) -> str:
    """Remove [tag] annotations and inserted ellipses; leave words intact for comparison."""
    without_tags = re.sub(r"\[[^\]]*\]", " ", directed_text)
    without_ellipses = without_tags.replace("...", " ").replace("…", " ")
    return without_ellipses


def verify_content_preserved(clean_text: str, directed_text: str) -> bool:
    """docs/03 §1 guardrail: stripped directed_text must equal clean_text, normalized."""
    return _normalize(strip_tags(directed_text)) == _normalize(clean_text)


def _uses_only_allowed_tags(directed_text: str) -> bool:
    found = re.findall(r"\[([^\]]*)\]", directed_text)
    return all(tag.strip().lower() in _ALL_TAGS for tag in found)


def has_tags(directed_text: str) -> bool:
    return bool(re.search(r"\[[^\]]+\]", directed_text))


def _resolve_client_and_model(client=None):
    """
    Build (client, model_id) per constants.DIRECTOR_PROVIDER. Groq hosts an OpenAI-compatible
    endpoint, so the same `openai` SDK/response_format contract works unchanged — only the
    api_key/base_url/model differ (docs/02 §1 note on the provider toggle).
    """
    if client is not None:
        model_id = (constants.DIRECTOR_MODEL_GROQ if constants.DIRECTOR_PROVIDER == "groq"
                    else constants.DIRECTOR_MODEL)
        return client, model_id

    from openai import OpenAI

    if constants.DIRECTOR_PROVIDER == "groq":
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY not set (add it to .env, or create a groq-secret in named mode)"
            )
        return OpenAI(api_key=api_key, base_url=constants.GROQ_BASE_URL), constants.DIRECTOR_MODEL_GROQ

    return OpenAI(), constants.DIRECTOR_MODEL


def direct_script(
    script_text: str,
    voice_character_hint: str | None = None,
    client=None,
) -> dict:
    """
    Run the director with a bounded retry on guardrail failure.

    Returns {clean_text, directed_text, stability_hint}. Raises DirectorGuardrailError if the
    content-preservation guardrail can't be satisfied within DIRECTOR_MAX_RETRIES.
    `client` is an OpenAI(-compatible) client; injected for testability, constructed lazily if
    None per constants.DIRECTOR_PROVIDER ("openai" or "groq").
    """
    client, model_id = _resolve_client_and_model(client)

    system_prompt = _build_system_prompt(voice_character_hint)
    last_directed = None

    for attempt in range(constants.DIRECTOR_MAX_RETRIES + 1):
        user_content = script_text
        if attempt > 0:
            user_content = (
                f"{script_text}\n\n(Your previous annotation changed the wording. Re-annotate "
                f"WITHOUT altering any words — only insert bracketed tags, caps, and ellipses.)"
            )

        resp = client.chat.completions.create(
            model=model_id,
            response_format=_RESPONSE_SCHEMA,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        )
        result = json.loads(resp.choices[0].message.content)

        # Trust the input as the source of truth for clean_text (not the model's echo).
        result["clean_text"] = script_text
        directed = result["directed_text"]
        last_directed = directed

        if not _uses_only_allowed_tags(directed):
            continue  # invented a tag — retry
        if verify_content_preserved(script_text, directed):
            return result

    raise DirectorGuardrailError(
        f"director_guardrail_failed after {constants.DIRECTOR_MAX_RETRIES} retries; "
        f"last directed_text did not reduce to the input script. last={last_directed!r}"
    )
