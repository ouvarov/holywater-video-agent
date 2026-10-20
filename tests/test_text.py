import pytest

from hwva.text import detect_lang, diff_words, extract_quoted_line, has_digits, normalize


@pytest.mark.parametrize(
    ("idea", "line"),
    [
        ('Кіт каже: "Я тут головний"', "Я тут головний"),
        ("Кіт каже: «Я тут головний»", "Я тут головний"),
        ("A cat says “I am the boss” and leaves", "I am the boss"),
        ("Кіт каже: „Я тут головний“", "Я тут головний"),
        ('Кіт каже: "Я «тут» головний"', "Я «тут» головний"),
        ('Кіт каже: «Я тут "головний"»', 'Я тут "головний"'),
        ("Кіт каже: «Я «тут» головний»", "Я «тут» головний"),
        ('Кіт каже: "М\'ята — моя"', "М'ята — моя"),
        ('Кіт каже: "Перше" і "друге"', "Перше"),
        ("Кіт шукає господаря", None),
        ('Кіт каже: "незакрита', None),
        ('Кіт каже: ""', None),
    ],
)
def test_extract_quoted_line(idea: str, line: str | None) -> None:
    assert extract_quoted_line(idea) == line


@pytest.mark.parametrize(
    ("text", "lang"),
    [
        ("Я тут головний", "uk"),
        ("Їжак", "uk"),
        ("I am the boss", "en"),
        ("I am the boss, Олег", "en"),
        ("Я тут, ok", "uk"),
    ],
)
def test_detect_lang(text: str, lang: str) -> None:
    assert detect_lang(text) == lang


def test_russian_yo_is_not_counted_as_ukrainian() -> None:
    """ё is not a Ukrainian letter, so it doesn't count as Cyrillic for the majority."""
    assert detect_lang("ёёё ok") == "en"


def test_has_digits() -> None:
    assert has_digits("2 гривні")
    assert not has_digits("дві гривні")


def test_normalize_strips_stress_apostrophes_case_punctuation() -> None:
    assert normalize("За́мок — м’який, П'ять!") == "замок м'який п'ять"
    assert normalize("Сім'я") == normalize("Сімʼя") == normalize("Сім`я")


def test_normalize_drops_quote_like_apostrophes_at_word_edges() -> None:
    assert normalize("‘друже’") == "друже"


def test_normalize_splits_hyphens() -> None:
    assert normalize("будь-ласка") == normalize("будь ласка")


def test_diff_words_empty_on_verbatim_match() -> None:
    assert diff_words("Я тут головний!", "я тут головний") == []


def test_diff_words_reports_paraphrase() -> None:
    assert diff_words("Я тут головний", "Я тут найголовніший") == [
        "expected 'головний', heard 'найголовніший'"
    ]


def test_diff_words_truncated_transcript_fails() -> None:
    assert diff_words("Я тут головний", "Я тут") == ["missing 'головний'"]


def test_diff_words_extra_word_does_not_shift_the_rest() -> None:
    assert diff_words("я тут головний кіт", "я тут дуже головний кіт") == ["extra 'дуже'"]


def test_diff_words_tolerates_stt_writing_numbers_as_digits() -> None:
    assert diff_words("У мене дві гривні", "у мене 2 гривні", "uk") == []
    assert diff_words("I have twenty five cats", "I have 25 cats", "en") == []


def test_diff_words_digits_do_not_hide_other_mismatches() -> None:
    assert diff_words("У мене дві гривні", "у тебе 2 гривні", "uk") != []


# --- numbers heard as digits compare by value (specs/voice.md) --------------


@pytest.mark.parametrize(
    ("expected", "heard", "lang"),
    [
        ("дві гривні", "5 гривні", "uk"),
        ("сім хвилин", "5 хвилин", "uk"),
        ("Тримай двері", "Тримай 3 двері", "uk"),  # «двері» is not «два»
        ("Тримай двері", "3 двері", "uk"),
        ("одразу йди", "1 йди", "uk"),
        ("я стою", "я 100", "uk"),
        ("наша сім'я", "наша 7", "uk"),
        ("a tent", "a 10", "en"),
        ("tennis at noon", "10 at noon", "en"),
        ("an android", "an 1", "en"),
    ],
)
def test_diff_words_digits_must_have_the_same_value(expected: str, heard: str, lang: str) -> None:
    assert diff_words(expected, heard, lang) != []  # type: ignore[arg-type]


def test_diff_words_number_mismatch_names_words_and_digits() -> None:
    assert diff_words("У мене дві гривні", "у мене 5 гривні", "uk") == ["expected 'дві', heard '5'"]


def test_diff_words_number_words_heard_as_words_keep_their_inflection() -> None:
    # digits elsewhere in the transcript must not relax the word-for-word check of «дві»
    assert diff_words("дві гривні, дві копійки", "2 гривні, два копійки", "uk") == [
        "expected 'дві', heard 'два'"
    ]


@pytest.mark.parametrize(
    ("expected", "heard", "lang"),
    [
        ("У мене дві гривні", "у мене 2 гривні", "uk"),
        ("дві гривні і три копійки", "2 гривні і три копійки", "uk"),
        ("Мені двадцять п'ять років", "мені 25 років", "uk"),
        ("Тут сто двадцять три кроки", "тут 123 кроки", "uk"),
        ("Їй п'ятнадцять", "їй 15", "uk"),
        ("Дві тисячі двадцять гривень", "2020 гривень", "uk"),
        ("Двадцять п'ять тисяч", "25 000", "uk"),
        ("Одна тисяча", "1000", "uk"),
        ("Тисяча гривень", "1000 гривень", "uk"),
        ("I have twenty five cats", "I have 25 cats", "en"),
        ("one hundred and five dogs", "105 dogs", "en"),
        ("two thousand and twenty", "2020", "en"),
    ],
)
def test_diff_words_equal_number_values_match(expected: str, heard: str, lang: str) -> None:
    assert diff_words(expected, heard, lang) == []  # type: ignore[arg-type]


# digits followed by a scale word: how Whisper/Scribe usually write amounts (review round 1)
@pytest.mark.parametrize(
    ("expected", "heard", "lang"),
    [
        ("Дві тисячі гривень", "2 тисячі гривень", "uk"),
        ("двадцять п'ять тисяч", "25 тисяч", "uk"),
        ("три мільйони", "3 мільйони", "uk"),
        ("дві тисячі двадцять", "2 тисячі двадцять", "uk"),
        ("five million", "5 million", "en"),
        ("five hundred", "5 hundred", "en"),
    ],
)
def test_diff_words_digits_with_scale_word_match(expected: str, heard: str, lang: str) -> None:
    assert diff_words(expected, heard, lang) == []  # type: ignore[arg-type]


def test_diff_words_digits_with_scale_word_compare_by_value() -> None:
    assert diff_words("дві тисячі", "3 тисячі", "uk") == ["expected 'дві тисячі', heard '3 тисячі'"]


@pytest.mark.parametrize("heard", ["²", "5²", "①"])
def test_diff_words_unicode_non_decimal_digits_are_words_not_a_crash(heard: str) -> None:
    assert diff_words("дві", heard, "uk") != []


@pytest.mark.parametrize(
    ("expected", "heard", "lang"),
    [
        ("півтори години", "1,5 години", "uk"),  # not an integer
        ("у дві тисячі двадцять п'ятому році", "у 2025 році", "uk"),  # ordinal
        ("nineteen eighty four", "1984", "en"),  # year reading in pairs
    ],
)
def test_diff_words_known_limits_fail_safe(expected: str, heard: str, lang: str) -> None:
    """specs/voice.md, known limits: fractions, ordinals and pair-read years heard as digits are a
    mismatch (re-synthesis, then GateFailed), never a false match."""
    assert diff_words(expected, heard, lang) != []  # type: ignore[arg-type]


# --- line_gender (specs/script.md, "Voice persona") ---------------------------------------------
# Imported inside the helper so the rest of this file still runs before line_gender exists.


def _line_gender(line: str, lang: str) -> str | None:
    from hwva.text import line_gender

    result: str | None = line_gender(line, lang)
    return result


@pytest.mark.parametrize(
    ("line", "gender"),
    [
        # first-person past tense: «я … -в» / «я … -ла»
        ("Я зробив це", "male"),
        ("Я зробила це", "female"),
        ("Я вчора прийшов додому", "male"),  # one adverb between «я» and the verb
        ("Я вчора прийшла додому", "female"),
        # «головний/головна»
        ("Я тут головний", "male"),
        ("Я тут головна", "female"),
        # «сам/сама»
        ("Я сам", "male"),
        ("Я сама", "female"),
        # «радий/рада»
        ("Я радий тебе бачити", "male"),
        ("Я рада тебе бачити", "female"),
        # «готовий/готова»
        ("Я готовий!", "male"),
        ("Я готова!", "female"),
        # several markers of the same gender, punctuation and case
        ("Та ну, я сама все зробила!", "female"),
        ("ТА НУ, Я САМ УСЕ ЗРОБИВ!", "male"),
    ],
)
def test_line_gender_markers(line: str, gender: str) -> None:
    assert _line_gender(line, "uk") == gender


@pytest.mark.parametrize(
    "line",
    [
        "Red Bull надає крила",  # «-ла» without first-person «я» is not a marker
        "Дощ знову, а я сухий і щасливий",
        "Привіт, це тестова репліка",
        "Я тут",
    ],
)
def test_line_gender_no_marker_is_none(line: str) -> None:
    assert _line_gender(line, "uk") is None


@pytest.mark.parametrize(
    "line",
    # every marker here counts under the new rule (verb / «рада» right after «я»)
    ["Я сам, я зробила", "Я радий, а я рада?", "Я готова, і я головний"],
)
def test_line_gender_both_genders_is_none(line: str) -> None:
    assert _line_gender(line, "uk") is None


@pytest.mark.parametrize(
    ("line", "lang"),
    [
        ("I did it all by myself", "en"),
        ("She is ready, I am the boss", "en"),
        ("Я тут головний", "en"),  # English -> always None, whatever the text
    ],
)
def test_line_gender_english_is_none(line: str, lang: str) -> None:
    assert _line_gender(line, lang) is None


# --- line_gender false positives (specs/script.md, "Voice persona") ---------
# Cases that already pass with the current heuristic are plain tests (strict xfail would XPASS).


@pytest.mark.parametrize(
    "line",
    [
        "Я люблю Львів",  # a capitalised word after «я» is never a past-tense verb
        "Я їду у Харків",
    ],
)
def test_line_gender_capitalised_word_is_not_a_verb(line: str) -> None:
    assert _line_gender(line, "uk") is None


@pytest.mark.parametrize(
    "line",
    [
        "Міська рада вирішила",  # «рада» counts only right after «я»
        "Наша рада ухвалила рішення",
    ],
)
def test_line_gender_rada_not_after_ya_is_none(line: str) -> None:
    assert _line_gender(line, "uk") is None


@pytest.mark.parametrize(
    ("line", "gender"),
    [
        ("я вже зробив", "male"),  # one adverb between «я» and the verb
        ("я вже зробила", "female"),
        ("я рада", "female"),  # «рада» right after «я»
        ("я радий", "male"),
    ],
)
def test_line_gender_true_positives_still_hold(line: str, gender: str) -> None:
    assert _line_gender(line, "uk") == gender
