import re
from typing import Literal

Language = Literal["hu", "en"]

LANGUAGE_NAMES: dict[Language, str] = {"hu": "Hungarian", "en": "English"}

HUNGARIAN_LETTERS = set("áéíóöőúüű")
HUNGARIAN_WORDS = {
    "mi", "mit", "mik", "mely", "melyik", "milyen", "mennyi", "mikor", "hol", "hogyan",
    "miért", "ki", "kik", "kinek", "kell", "lehet", "van", "nincs", "nem", "az", "és",
    "egy", "vagy", "ha", "hogy", "adó", "adót", "társasági",
}
WORD = re.compile(r"\w+")


def detect_language(text: str) -> Language:
    lowered = text.lower()
    if HUNGARIAN_LETTERS & set(lowered):
        return "hu"
    if HUNGARIAN_WORDS & set(WORD.findall(lowered)):
        return "hu"
    return "en"
