"""Input guardrails and PII masking.

`check_input` runs BEFORE the LLM sees a message. It blocks:
  * prompt injection / jailbreak attempts      -> reason "prompt_injection"
  * blocked topics (DIY wiring, bypassing a lock, medical claims about water) -> "blocked_topic"
  * sensitive identity data (Aadhaar-like numbers, card numbers, OTPs)        -> "pii"

`mask_pii` hides phone numbers, emails and Aadhaar-like numbers. It is used before
anything is logged and on the bot's final answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Verdict:
    """Result of an input check. `reason` is None when allowed."""

    allowed: bool
    reason: str | None = None


_I = re.IGNORECASE

INJECTION_PATTERNS = [
    r"\b(ignore|disregard|forget|override|bypass)\b.{0,40}\b(instructions?|rules|guidelines|prompt|guardrails?)\b",
    r"\b(system|developer|hidden|initial)\s+(prompt|message|instructions?)\b",
    r"\b(reveal|show|print|repeat|leak|output)\b.{0,30}\b(your|the system|hidden|initial)\s+(prompt|instructions|rules)\b",
    r"\byou are (now|no longer)\b",
    r"\b(pretend|act) (to be|as)\b.{0,40}\b(unrestricted|no rules|evil|hacker|dan|admin|developer)\b",
    r"\b(jailbreak|dan mode|developer mode|god mode)\b",
    r"\bnew (instructions|rules)\s*:",
    r"\b(instructions?|rules)\b.{0,20}\b(bhool jao|bhul jao|ignore karo|mat mano)\b",
    r"\b(pichle|purane|saare)\b.{0,20}\b(instructions?|rules)\b.{0,20}\b(bhool|bhul|ignore)\b",
    r"</?\s*(system|assistant)\s*>",
]

BLOCKED_TOPIC_PATTERNS = [
    # DIY electrical / wiring work
    r"\b(open|opening|remove|change|rewire|rewiring|repair|fix|connect|cut|join|replace)\b.{0,40}"
    r"\b(wiring|wires?|capacitor|junction box|switchboard|mcb|live wire|neutral wire|mains)\b",
    r"\b(wiring|wires?)\b.{0,30}\b(khud|myself|by myself|diy|kaise (jode|badle|kholu))\b",
    r"\bhow (do|can|to) i?\s*(wire|install the wiring)\b",
    # bypassing / breaking a lock
    r"\b(bypass|hack|break|pick|crack|force open|forcefully open|todna|tod)\b.{0,40}\b(lock|door|s1)\b",
    r"\b(lock|door)\b.{0,40}\b(bypass|hack|pick|crack|break into|without (the )?(owner|permission))\b",
    r"\b(open|unlock)\b.{0,30}\b(someone else'?s|neighbou?r'?s|other person'?s|without (the )?(owner|permission))\b",
    r"\b(unlock|open)\b.{0,40}\bwithout\b.{0,25}\b(owner|permission|their (knowledge|consent))\b",
    # medical claims about water
    r"\b(cure|cures|treat|treats|heal|heals|prevent|prevents|reverse)\b.{0,50}"
    r"\b(cancer|diabetes|kidney|stones?|disease|diseases|bp|blood pressure|thyroid|covid|infection|arthritis)\b",
    r"\b(cancer|diabetes|kidney stones?|disease|thyroid)\b.{0,50}\b(cure|treat|heal|theek|thik)\b",
]

# Hinglish / English "someone else's lock" requests. Each tuple = all parts must appear
# (any order). "mera lock nahi khul raha" (my own lock) does NOT match: no third party.
LOCK_THIRD_PARTY_PARTS: list[tuple[str, ...]] = [
    (r"\b(padosi\w*|pados|dusr[eiao]|doosr[eiao]|kisi aur|kisi ka|neighbou?r'?s|someone else'?s)\b",
     r"\b(khol\w*|khul\w*|unlock\w*|open\w*|ghus\w*|andar)\b",
     r"\b(lock|s1|taala|tala|darwaa?za|door|ghar|house|flat)\b"),
    (r"\b(lock|s1|taala|tala|darwaa?za|door)\b", r"\b(tod\w*|todna|tor\w*)\b"),
]

SENSITIVE_PATTERNS = [
    r"(?<!\d)\d{4}[\s-]?\d{4}[\s-]?\d{4}(?!\d)",            # Aadhaar-like 12 digits
    r"(?<!\d)\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}(?!\d)",  # card number
    r"\b(otp|cvv|upi pin|atm pin)\b\D{0,15}\d{3,8}\b",
    r"\b(password|passcode|passwd|pwd)\b\s*(?:is|:|=|hai|h)\s*\S{3,}",
    r"\b(password|passcode)\s+(?!is\b)\S{3,}\s+(hai|h)\b",     # "mera password hunter2 hai"
]

REFUSAL_MESSAGES: dict[str, str] = {
    "prompt_injection": ("Sorry, I can't help with that. I can only help with Hawa Home products "
                         "- for example fan speeds, lock batteries, purifier filters, warranty "
                         "or raising a service ticket."),
    "blocked_topic": ("Sorry, I can't help with that. For your safety I can't give instructions for "
                      "electrical wiring work, bypassing a lock, or make medical claims about "
                      "water. Please contact a Hawa certified technician - I can raise a service "
                      "ticket for you."),
    "pii": ("Sorry, I can't help with that. Please don't share Aadhaar numbers, card numbers, "
            "OTPs or passwords in chat - Hawa will never ask for them. A mobile number is only "
            "needed to raise a service ticket."),
}


def _matches(patterns: list[str], text: str) -> bool:
    return any(re.search(p, text, _I) for p in patterns)


def check_input(text: str) -> Verdict:
    """Decide whether a user message may be sent to the LLM."""
    if _matches(INJECTION_PATTERNS, text):
        return Verdict(False, "prompt_injection")
    if _matches(BLOCKED_TOPIC_PATTERNS, text) or any(
            all(re.search(part, text, _I) for part in parts) for parts in LOCK_THIRD_PARTY_PARTS):
        return Verdict(False, "blocked_topic")
    # A valid phone number is fine (needed for tickets); mask phones before looking for
    # 12/16 digit IDs so "+91 9876543210" is not mistaken for an Aadhaar number.
    if _matches(SENSITIVE_PATTERNS, _mask_phones(text)):
        return Verdict(False, "pii")
    return Verdict(True, None)


# Indian mobile: optional +91 / 91 / 0 prefix, then 10 digits starting 6-9. One space or
# dash is allowed between any digits, so 98765 43210, 9876 543 210 and 098765-43210 all match.
PHONE_RE = re.compile(r"(?<![\d+])(?:(?:\+?91|0)[\s-]?)?[6-9](?:[\s-]?\d){9}(?!\d)")
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
CARD_RE = re.compile(r"(?<!\d)\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}(?!\d)")
AADHAAR_RE = re.compile(r"(?<!\d)\d{4}[\s-]?\d{4}[\s-]?\d{4}(?!\d)")
OTP_RE = re.compile(r"\b((?:otp|cvv|upi pin|atm pin)\b\D{0,15}?)\d{3,8}\b", _I)
PASSWORD_RE = re.compile(r"\b((?:password|passcode|passwd|pwd)\b\s*(?:is|:|=|hai|h)\s*)[\"']?[^\s\"',.]{3,}[\"']?", _I)
PASSWORD_HAI_RE = re.compile(r"\b((?:password|passcode)\s+)(?!is\b)[^\s\"',.]{3,}(?=\s+(?:hai|h)\b)", _I)
# Official Hawa support contacts are not personal data and must stay visible.
ALLOWED_EMAIL_DOMAINS = ("@hawahome.example",)


def _mask_phones(text: str) -> str:
    return PHONE_RE.sub("[PHONE]", text)


def _mask_email(match: re.Match) -> str:
    email = match.group(0)
    return email if email.lower().endswith(ALLOWED_EMAIL_DOMAINS) else "[EMAIL]"


def mask_pii(text: str) -> str:
    """Replace phones, emails, card / Aadhaar numbers, OTPs and passwords with placeholders."""
    if not text:
        return text
    text = EMAIL_RE.sub(_mask_email, text)
    text = OTP_RE.sub(r"\1[OTP]", text)
    text = PASSWORD_RE.sub(r"\1[SECRET]", text)
    text = PASSWORD_HAI_RE.sub(r"\1[SECRET]", text)
    text = _mask_phones(text)
    text = CARD_RE.sub("[CARD]", text)
    return AADHAAR_RE.sub("[AADHAAR]", text)


def mask_obj(value):
    """mask_pii applied to every string inside a dict / list (used for tool args and results)."""
    if isinstance(value, str):
        return mask_pii(value)
    if isinstance(value, dict):
        return {k: mask_obj(v) for k, v in value.items()}
    if isinstance(value, list):
        return [mask_obj(v) for v in value]
    return value


# ----------------------------------------------------------------- indirect injection
# Data the bot reads (device names, tool results, manual chunks) is untrusted: an attacker
# could plant instructions there. These extra patterns only apply to such data.
DATA_INJECTION_PATTERNS = INJECTION_PATTERNS + [
    r"\b(system note|assistant instructions?|note to (the )?(assistant|ai|bot|model)|new task)\b",
    r"\binstructions? (for|to) (the )?(assistant|ai|bot|model)\b",
    r"</?\s*(data|system|assistant|instructions?|manual|devices?)\s*>",
    r"\b(print|reveal|show|repeat|output)\b.{0,30}\binstructions\b",
]
REMOVED_DATA = "[removed: text that looked like instructions]"


def looks_like_injection(text: str) -> bool:
    """True if a piece of untrusted data contains instruction-like text."""
    return _matches(DATA_INJECTION_PATTERNS, text)


def sanitize_data(value):
    """Replace any string inside untrusted data that looks like an injected instruction."""
    if isinstance(value, str):
        return REMOVED_DATA if looks_like_injection(value) else value
    if isinstance(value, dict):
        return {k: sanitize_data(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_data(v) for v in value]
    return value
