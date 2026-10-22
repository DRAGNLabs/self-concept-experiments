"""Render identical content in each native role for probe training."""

from selfconcept.measurement.templates import reasoning_template_kwargs

ROLES = ("user", "cot", "assistant")


def render_probe(tokenizer, family, text, role, context="A neutral document follows."):
    """Render identical content in three native roles; return its character span.

    A fixed prior exchange supplies the same context for every role copy. The
    assistant role uses an empty thought block where the native template does.
    """
    if role not in ROLES:
        raise ValueError(role)
    if family == "kimi":
        raise ValueError("Kimi-Dev-72B has no verified native reasoning-role template")
    # Cropping to a token budget can end on whitespace; native templates trim
    # that whitespace. Apply the same normalization to every role copy.
    text = text.strip()
    if not text:
        raise ValueError("Probe content must be nonempty")
    history = [{"role": "user", "content": context}, {"role": "assistant", "content": "Understood."}]
    if role == "user":
        messages = history + [{"role": "user", "content": text}]
    else:
        messages = history + [{"role": "user", "content": "Continue."}]
        message = {"role": "assistant", "content": text if role == "assistant" else ""}
        if role == "cot":
            message["reasoning_content"] = text
        messages.append(message)
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, **reasoning_template_kwargs(family))
    if rendered.count(text) != 1:
        raise ValueError(f"Template lost or duplicated {role} content")
    start = rendered.index(text)
    return rendered, (start, start + len(text))
