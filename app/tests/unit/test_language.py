import pytest

from rag.language import detect_language


@pytest.mark.parametrize("text, expected", [
    ("Mi a növekedési adóhitel?", "hu"),
    ("Mi a NAHI?", "hu"),
    ("Ki a TAO alanya?", "hu"),
    ("What is the NAHI?", "en"),
    ("Who is subject to corporate tax?", "en"),
])
def test_detect_language(text: str, expected: str):
    assert detect_language(text) == expected


@pytest.mark.xfail(strict=True, reason="Known limitation: a Hungarian term makes an English question count as Hungarian")
def test_english_question_with_hungarian_term():
    assert detect_language("What is TAO-felajánlás?") == "en"
