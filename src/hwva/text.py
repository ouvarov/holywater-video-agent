"""Text handling for the verbatim gate. Pure functions, fully covered by tests.

Numbers policy: digits are NOT accepted in the dialogue line — the user writes numbers in words.
Converting digits ourselves can't get Ukrainian gender/case right ("2 гривні" -> "два гривні"),
and the line must sound exactly as written. STT may still return digits; the comparison
tolerates that (see `diff_words`)."""

from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass

from hwva.schemas import Gender, Lang

_APOSTROPHES = "'’ʼ`´‘"  # all variants seen in Ukrainian text -> one
_PAIRS = {'"': '"', "«": "»", "„": "“”", "“": "”"}
_DIGIT = re.compile(r"\d")
_CYRILLIC = re.compile(r"[а-яіїєґ]", re.IGNORECASE)
_LATIN = re.compile(r"[a-z]", re.IGNORECASE)
_UK_ONLY = re.compile(r"[іїєґ]", re.IGNORECASE)

# --- numerals: whole known forms only (no stems: «двері», «одразу», «стою» are words) -----------
# kind: unit 0..9, teen 10..19, ten 20..90, hundred 100..900 (uk words), mult (en "hundred"),
# scale 1000/10^6/10^9, and (en connector inside a number)

_UNIT, _TEEN, _TEN, _HUNDRED, _MULT, _SCALE, _AND = (
    "unit", "teen", "ten", "hundred", "mult", "scale", "and"
)  # fmt: skip


def _uk_tsyat(base: str) -> tuple[str, ...]:
    """п'ять/одинадцять/двадцять/п'ятдесят -> the case forms (п'яти, п'ятьох, п'ятьом, п'ятьма,
    п'ятьома; …десяти, …десятьох…)."""
    stem = base[:-1] if base.endswith("ь") else base
    return (base, stem + "и", stem + "ьох", stem + "ьом", stem + "ьма", stem + "ьома")


_UK_FORMS: list[tuple[int, str, tuple[str, ...]]] = [
    (0, _UNIT, ("нуль", "нуля", "нулю", "нулем", "нулі", "нулів")),
    (1, _UNIT, ("один", "одна", "одне", "одно", "одного", "одної", "одній", "одному", "одним",
                "одною", "одні", "одних", "одними")),
    (2, _UNIT, ("два", "дві", "двох", "двом", "двома", "двоє")),
    (3, _UNIT, ("три", "трьох", "трьом", "трьома", "троє")),
    (4, _UNIT, ("чотири", "чотирьох", "чотирьом", "чотирма", "четверо")),
    (5, _UNIT, (*_uk_tsyat("п'ять"), "п'ятеро")),
    (6, _UNIT, ("шість", "шести", "шістьох", "шістьом", "шістьма", "шістьома", "шестеро")),
    (7, _UNIT, ("сім", "семи", "сімох", "сімом", "сьома", "сімома", "семеро")),
    (8, _UNIT, ("вісім", "восьми", "вісьмох", "восьмох", "вісьмом", "вісьма", "вісьмома",
                "восьмеро")),
    (9, _UNIT, (*_uk_tsyat("дев'ять"), "дев'ятеро")),
    (10, _TEEN, (*_uk_tsyat("десять"), "десятеро")),
    *[(10 + i, _TEEN, _uk_tsyat(w)) for i, w in enumerate(
        ("одинадцять", "дванадцять", "тринадцять", "чотирнадцять", "п'ятнадцять",
         "шістнадцять", "сімнадцять", "вісімнадцять", "дев'ятнадцять"), start=1)],
    (20, _TEN, _uk_tsyat("двадцять")),
    (30, _TEN, _uk_tsyat("тридцять")),
    (40, _TEN, ("сорок", "сорока")),
    (50, _TEN, _uk_tsyat("п'ятдесят")),
    (60, _TEN, _uk_tsyat("шістдесят")),
    (70, _TEN, _uk_tsyat("сімдесят")),
    (80, _TEN, _uk_tsyat("вісімдесят")),
    (90, _TEN, ("дев'яносто", "дев'яноста")),
    (100, _HUNDRED, ("сто", "ста")),
    (200, _HUNDRED, ("двісті", "двохсот", "двомстам", "двомастами", "двохстах")),
    (300, _HUNDRED, ("триста", "трьохсот", "трьомстам", "трьомастами", "трьохстах")),
    (400, _HUNDRED, ("чотириста", "чотирьохсот", "чотирьомстам", "чотирмастами",
                     "чотирьохстах")),
    (500, _HUNDRED, ("п'ятсот", "п'ятисот", "п'ятистам", "п'ятьмастами", "п'ятистах")),
    (600, _HUNDRED, ("шістсот", "шестисот", "шестистам", "шістьмастами", "шестистах")),
    (700, _HUNDRED, ("сімсот", "семисот", "семистам", "сьомастами", "семистах")),
    (800, _HUNDRED, ("вісімсот", "восьмисот", "восьмистам", "вісьмастами", "восьмистах")),
    (900, _HUNDRED, ("дев'ятсот", "дев'ятисот", "дев'ятистам", "дев'ятьмастами",
                     "дев'ятистах")),
    (1000, _SCALE, ("тисяча", "тисячі", "тисяч", "тисячу", "тисячею", "тисячам", "тисячами",
                    "тисячах")),
    (10**6, _SCALE, ("мільйон", "мільйона", "мільйону", "мільйони", "мільйонів", "мільйоном",
                     "мільйонам", "мільйонами", "мільйонах")),
    (10**9, _SCALE, ("мільярд", "мільярда", "мільярду", "мільярди", "мільярдів", "мільярдом",
                     "мільярдам", "мільярдами", "мільярдах")),
]  # fmt: skip
_EN_UNITS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
_EN_TEENS = ("ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
             "seventeen", "eighteen", "nineteen")  # fmt: skip
_EN_TENS = ("twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")
_EN_FORMS: list[tuple[int, str, tuple[str, ...]]] = [
    *[(i, _UNIT, (w,)) for i, w in enumerate(_EN_UNITS)],
    *[(10 + i, _TEEN, (w,)) for i, w in enumerate(_EN_TEENS)],
    *[(20 + 10 * i, _TEN, (w,)) for i, w in enumerate(_EN_TENS)],
    (100, _MULT, ("hundred",)),
    (1000, _SCALE, ("thousand",)),
    (10**6, _SCALE, ("million",)),
    (10**9, _SCALE, ("billion",)),
    (0, _AND, ("and",)),
]
_NUMERALS: dict[Lang, dict[str, tuple[int, str]]] = {
    "uk": {form: (value, kind) for value, kind, group in _UK_FORMS for form in group},
    "en": {form: (value, kind) for value, kind, group in _EN_FORMS for form in group},
}
_DIGITS = "digits"  # a digit group heard by STT, followed by scale words: "2 тисячі", "5 million"
_MAY_FOLLOW: dict[str, set[str | None]] = {  # what the previous word of the same number may be
    _UNIT: {None, _TEN, _HUNDRED, _MULT, _SCALE, _AND},
    _TEEN: {None, _HUNDRED, _MULT, _SCALE, _AND},
    _TEN: {None, _HUNDRED, _MULT, _SCALE, _AND},
    _HUNDRED: {None, _SCALE},
    _MULT: {None, _UNIT, _TEEN, _TEN, _DIGITS},
    _SCALE: {None, _UNIT, _TEEN, _TEN, _HUNDRED, _MULT, _DIGITS},
}


def extract_quoted_line(idea: str) -> str | None:
    """The dialogue line is the first quoted span; nested quotes of another kind are kept inside.

    Single quotes are not supported: they collide with Ukrainian apostrophes ("м'ята")."""
    start = next((i for i, ch in enumerate(idea) if ch in _PAIRS), None)
    if start is None:
        return None
    opener, closers = idea[start], _PAIRS[idea[start]]
    depth = 1
    for i in range(start + 1, len(idea)):
        ch = idea[i]
        if opener == "«" and ch == "«":
            depth += 1
        elif ch in closers:
            depth -= 1
            if depth == 0:
                line = idea[start + 1 : i].strip()
                return line or None
    return None


def has_digits(text: str) -> bool:
    return bool(_DIGIT.search(text))


def detect_lang(text: str) -> Lang:
    """uk/en by letter majority. Cyrillic without Ukrainian-only letters is still treated as uk —
    only uk and en are supported."""
    if _UK_ONLY.search(text):
        return "uk"
    return "uk" if len(_CYRILLIC.findall(text)) > len(_LATIN.findall(text)) else "en"


def normalize(text: str) -> str:
    """Comparison form: NFC, no stress marks, one apostrophe (only inside words), lowercase,
    no punctuation. Hyphens split words: "будь-ласка" == "будь ласка"."""
    return " ".join(_words(text)).lower()


def _words(text: str) -> list[str]:
    """`normalize` without lowercasing, as a word list (line_gender needs the original case)."""
    text = unicodedata.normalize("NFD", text).replace("́", "")
    text = unicodedata.normalize("NFC", text)
    for ch in _APOSTROPHES:
        text = text.replace(ch, "'")
    text = "".join(c if c.isalnum() or c in " '" else " " for c in text)
    return [w.strip("'") for w in text.split() if w.strip("'")]


_GENDER_WORDS: dict[str, Gender] = {  # count anywhere in the line
    "головний": "male",
    "головна": "female",
    "сам": "male",
    "сама": "female",
    "готовий": "male",
    "готова": "female",
}
_AFTER_YA_WORDS: dict[str, Gender] = {  # count only right after «я» («Міська рада» is a noun)
    "радий": "male",
    "рада": "female",
}
_CLAUSE = re.compile(r"[,.!?;:…—–()\"«»„“”]")
_MIN_PAST_LEN = 3  # "в"/"ла" alone are not verbs


def line_gender(line: str, lang: Lang | str) -> Gender | None:
    """Grammatical gender of the speaker of a Ukrainian line, from a small marker list:
    - «головний/головна», «сам/сама», «готовий/готова» anywhere;
    - «радий/рада» only as the word right after «я» («я рада»; «Міська рада вирішила» -> none);
    - first-person past tense: a lowercase word ending in -в / -ла right after «я», or with one
      word (an adverb) between («я вже зробив»), in the same clause. A capitalised word is never
      a verb («Я люблю Львів» -> none).
    None when no marker or markers of both genders. English -> always None.

    A heuristic: a lowercase noun ending in -в/-ла in those slots («я бачу крила») is still a
    false positive. It only picks the voice gender; the line itself is never changed."""
    if lang != "uk":
        return None
    found: set[Gender] = set()
    for clause in _CLAUSE.split(line):
        cased = _words(clause)
        words = [w.lower() for w in cased]
        found.update(_GENDER_WORDS[w] for w in words if w in _GENDER_WORDS)
        for i, word in enumerate(words):
            if word != "я":
                continue
            if i + 1 < len(words) and words[i + 1] in _AFTER_YA_WORDS:
                found.add(_AFTER_YA_WORDS[words[i + 1]])
                continue
            for j in (i + 1, i + 2):  # the verb, or one adverb and then the verb
                gender = _past_tense_gender(cased[j]) if j < len(cased) else None
                if gender is not None:
                    found.add(gender)
                    break
    return found.pop() if len(found) == 1 else None


def _past_tense_gender(word: str) -> Gender | None:
    """-в -> male, -ла -> female, for a lowercase word that is not itself a marker word."""
    lower = word.lower()
    if word != lower or len(word) < _MIN_PAST_LEN:
        return None
    if lower in _GENDER_WORDS or lower in _AFTER_YA_WORDS:
        return None
    if lower.endswith("в"):
        return "male"
    if lower.endswith("ла"):
        return "female"
    return None


@dataclass(frozen=True)
class _Token:
    key: str  # what difflib compares: the word, or "#<value>" for a number
    text: str  # what a problem message shows
    spelled: bool = False  # a number written in words (its exact form still matters)


def _parse_number(
    words: list[str], i: int, table: dict[str, tuple[int, str]], digits: int | None = None
) -> tuple[int, int]:
    """The integer spelled by the number words starting at `i`, and the index after them. Stops
    where the next word can't continue the same number ("два три" -> 2, then 3). (0, i) when
    words[i] doesn't start a number. `digits`: the value of a digit group just before `i`, which
    only scale words may continue ("2 тисячі" -> 2000)."""
    total, current = 0, digits or 0
    prev: str | None = None if digits is None else _DIGITS
    last_scale: int | None = None
    j = i
    while j < len(words) and words[j] in table:
        value, kind = table[words[j]]
        if kind == _AND:  # only inside a number, right before its units/tens
            nxt = table.get(words[j + 1]) if j + 1 < len(words) else None
            if prev not in (_HUNDRED, _MULT, _SCALE) or nxt is None or nxt[1] not in (
                _UNIT, _TEEN, _TEN
            ):  # fmt: skip
                break
            prev, j = kind, j + 1
            continue
        if value == 0 and kind == _UNIT:  # "zero" / "нуль" is a number on its own
            if prev is None:
                return 0, j + 1
            break
        if prev not in _MAY_FOLLOW[kind]:
            break
        if kind == _MULT:
            if current >= 100:
                break
            current = max(current, 1) * value
        elif kind == _SCALE:
            if last_scale is not None and value >= last_scale:
                break
            count = current if prev == _DIGITS else max(current, 1)  # "тисяча" alone is 1000
            total, current, last_scale = total + count * value, 0, value
        else:
            current += value
        prev, j = kind, j + 1
    return total + current, j


def _tokens(words: list[str], lang: Lang) -> list[_Token]:
    """Words, with every number as one token keyed by its value: a run of digit groups
    ("25 000", "2 тисячі") or a run of number words ("двадцять п'ять тисяч")."""
    table = _NUMERALS[lang]
    out: list[_Token] = []
    i = 0
    while i < len(words):
        if words[i].isdecimal():  # not isdigit: "²" passes that, and int() can't read it
            j = i + 1
            # thousands groups: "25 000". Two numbers heard as one group ("2 100" for «два, сто»)
            # stay one value: a mismatch, never a false match.
            while j < len(words) and words[j].isdecimal() and len(words[j]) == 3:
                j += 1
            value, end = _parse_number(words, j, table, digits=int("".join(words[i:j])))
            out.append(_Token(f"#{value}", " ".join(words[i:end])))
            i = end
            continue
        value, j = _parse_number(words, i, table)
        if j > i:
            out.append(_Token(f"#{value}", " ".join(words[i:j]), spelled=True))
            i = j
            continue
        out.append(_Token(words[i], words[i]))
        i += 1
    return out


def diff_words(expected: str, actual: str, lang: Lang = "uk") -> list[str]:
    """Word-level mismatches (aligned with difflib, so one extra word doesn't shift the rest).
    Empty list = verbatim match.

    If STT returned digits ("5" for "п'ять"), a number is compared by its value: a run of known
    numeral forms in the line must have the same value as the digits heard in its place. The TTS
    read our exact words, the STT just chose to write them as digits. Numbers heard as words are
    still compared word for word («дві» vs «два» is a mismatch)."""
    exp_words, act_words = normalize(expected).split(), normalize(actual).split()
    if exp_words == act_words:
        return []
    if any(w.isdecimal() for w in act_words):
        exp, act = _tokens(exp_words, lang), _tokens(act_words, lang)
    else:
        exp, act = [_Token(w, w) for w in exp_words], [_Token(w, w) for w in act_words]
    problems: list[str] = []
    matcher = difflib.SequenceMatcher(
        a=[t.key for t in exp], b=[t.key for t in act], autojunk=False
    )
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":  # same value; both spelled in words -> the form must be the same too
            problems += [
                f"expected {e.text!r}, heard {a.text!r}"
                for e, a in zip(exp[i1:i2], act[j1:j2], strict=True)
                if e.spelled and a.spelled and e.text != a.text
            ]
            continue
        want = " ".join(t.text for t in exp[i1:i2])
        heard = " ".join(t.text for t in act[j1:j2])
        if tag == "replace":
            problems.append(f"expected {want!r}, heard {heard!r}")
        elif tag == "delete":
            problems.append(f"missing {want!r}")
        elif tag == "insert":
            problems.append(f"extra {heard!r}")
    return problems
