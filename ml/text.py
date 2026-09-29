"""Text utilities: normalisation, cleaning, tokenisation, sentence splitting.

Works for Russian, Kazakh (Cyrillic) and English.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Iterable, List

KAZAKH_LETTERS = set("әғқңөұүһіӘҒҚҢӨҰҮҺІ")

# ---------------------------------------------------------------- stopwords
STOPWORDS = {
    "ru": set("""и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по только
    ее мне было вот от меня еще нет о из ему теперь когда даже ну вдруг ли если уже или ни быть был него
    до вас нибудь опять уж вам ведь там потом себя ничего ей может они тут где есть надо ней для мы тебя
    их чем была сам чтоб без будто чего раз тоже себе под будет ж тогда кто этот того потому этого какой
    совсем ним здесь этом один почти мой тем чтобы нее сейчас были куда зачем всех никогда можно при
    наконец два об другой хоть после над больше тот через эти нас про всего них какая много разве три
    эту моя впрочем хорошо свою этой перед иногда лучше чуть том нельзя такой им более всегда конечно
    всю между это также является которые который которая которое каких какие ли либо данного данной
    настоящего настоящее настоящий пункт статья""".split()),
    "kk": set("""және мен да де та те бен пен ал бірақ немесе не ма ме ба бе па пе мұнда сол осы ол олар
    біз сіз сен мен оның оған одан онда бұл бұның соның үшін туралы бойынша арқылы сияқты кейін дейін
    бар жоқ болып болады болса болған болу тиіс керек қажет әрбір барлық кез келген қандай қалай неше
    қашан қайда кім не неге ретінде сәйкес""".split()),
    "en": set("""a an the and or but if of to in on at by for with from as is are was were be been being
    it its this that these those there here what which who whom whose when where why how do does did
    can could should would may might must shall will not no yes than then so such into about over under
    between within without per any all each every some other our your their his her they them we you i
    me my he she us also""".split()),
}
# question words typed in everyday / colloquial form (also without Kazakh letters, e.g. "калай алуга болады")
STOPWORDS["kk"] |= set("""калай канша кандай кайда кашан неге кім кимге болама бола ма ме ба бе ше алам аламын
    кимнин кимге кимди алуга алу алады алса алсам алсак аламыз беру береди керекпе керек пе мага маган бизге сизге кажетпе
    истеу істеу істеймін истеймин жасау жасаймын
    айт айтшы айтып айтыныз айтыңыз айтасыз айтасын айтсан айтсаныз берши берші бер беринизши бериниз беріңіз
    туралы жайлы жайында акпарат ақпарат маглумат мәлімет малимет тусиндир түсіндір тусиндирши түсіндірші
    білгім билгим келеди каламын калам каласын алсам алсан болсам болса болады болады ма керекпин
    шыгып кайтып келуге келу кету кетсем кетип биздин бизде бизге бизди наш наша наше нашем нашей our""".split())
STOPWORDS["ru"] |= set("""как где когда почему можно мне хочу нужно могу надо получить сделать узнать
    подскажите скажите пожалуйста какой какая какие каким каков расскажи расскажите рассказать про
    информация информацию инфо объясни объясните""".split())
STOPWORDS["en"] |= set("""get obtain can could how need want please tell know about info information explain""".split())

# Kazakh-specific letters -> closest Russian letter.  People often type Kazakh without the special letters
# ("жатакхана", "калай") and mix up щ/ш, ё/е.  Folding makes these spellings match the documents.
_FOLD = str.maketrans({"ә": "а", "ғ": "г", "қ": "к", "ң": "н", "ө": "о", "ұ": "у", "ү": "у", "һ": "х",
                       "і": "и", "щ": "ш", "ё": "е", "ъ": "ь", "э": "е", "ї": "и", "ı": "и"})


def fold(text: str) -> str:
    """Lower-case and fold Kazakh / look-alike letters: 'Жатақханаға' -> 'жатакханага'."""
    return (text or "").lower().translate(_FOLD)


ALL_STOPWORDS = {fold(w) for w in set().union(*STOPWORDS.values())}

_WORD_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁёӘәҒғҚқҢңӨөҰұҮүҺһІі]+", re.UNICODE)
_SENT_SPLIT_RE = re.compile(r"(?<=[\.\!\?;])\s+(?=[\"«(]?[A-ZА-ЯЁӘҒҚҢӨҰҮҺІ0-9])")


def normalize(text: str) -> str:
    """Unicode-normalise and unify quotes/dashes/whitespace."""
    text = unicodedata.normalize("NFC", text or "")
    text = text.replace("­", "")  # soft hyphen
    text = re.sub(r"[“”„«»]", '"', text)
    text = re.sub(r"[‐‑‒–—―]", "-", text)
    text = text.replace("\xa0", " ")
    # Latin look-alikes that PDFs sometimes put inside Kazakh words
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def fix_hyphenation(text: str) -> str:
    """Join words broken with a hyphen at the end of a line: 'обуче-\nния' -> 'обучения'."""
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)


def tokenize(text: str, lower: bool = True) -> List[str]:
    text = text.lower() if lower else text
    return _WORD_RE.findall(text.replace("ё", "е"))


def content_tokens(text: str) -> List[str]:
    """Folded tokens without stopwords and one-letter tokens."""
    return [t for t in tokenize(fold(text)) if t not in ALL_STOPWORDS and len(t) > 1]


def stem(token: str) -> str:
    """Very light language-agnostic stemmer: keeps the first 6 characters.

    Kazakh is agglutinative and Russian is highly inflected, so a prefix stem
    is a robust, dependency-free way to match word forms
    ("стипендии" / "стипендия" / "стипендиясы" -> "стипен").
    """
    return token[:6] if len(token) > 6 else token


def stems(text: str) -> List[str]:
    return [stem(t) for t in content_tokens(text)]


_ABBREV_RE = re.compile(r"\b(ҚР|РК|РФ|ҚазҰТЗУ|т\.б|т\.с\.с|т\.е|т\.д|т\.п|им|г|гг|ж|жж|стр|бет|п|пп|No|№|etc|e\.g|i\.e)\.(?=\s)")


_INITIAL_RE = re.compile(r"(?<=\b[A-ZА-ЯЁӘҒҚҢӨҰҮҺІ])\.(?=\s?[A-ZА-ЯЁӘҒҚҢӨҰҮҺІ]\.|\s[A-ZА-ЯЁӘҒҚҢӨҰҮҺІ][a-zа-яёәғқңөұүһі]+)")


def split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s*\n\s*", " ", text.strip())
    text = _ABBREV_RE.sub(lambda m: m.group(1) + "\u2024", text)     # "ҚР. ҒжЖБМ" is not a sentence end
    text = _INITIAL_RE.sub("\u2024", text)                            # "К.И. Сатпаева": initials
    parts = [p.replace("\u2024", ".") for p in _SENT_SPLIT_RE.split(text)]
    out: List[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # very long "sentences" (tables, lists) are split on list markers
        if len(p) > 600:
            sub = re.split(r"\s(?=\d{1,2}\)\s|[-•]\s)", p)
            out.extend(s.strip() for s in sub if s.strip())
        else:
            out.append(p)
    return out


# ---------------------------------------------------------------- cleaning
_HEADER_PATTERNS = [
    re.compile(r"(?i)стр(\.|аница)?\s*\d+\s*(из|/)\s*\d+"),
    re.compile(r"(?i)page\s*\d+\s*(of|/)\s*\d+"),
    re.compile(r"(?i)\d+\s*(беттің|бет)\s*\d+"),
    re.compile(r"(?i)редакция\s*№?\s*\d+"),
    re.compile(r"(?i)басылым\s*№?\s*\d+"),
    re.compile(r"(?i)edition\s*№?\s*\d+"),
]


def clean_page(text: str) -> str:
    text = normalize(fix_hyphenation(text))
    lines = []
    for line in text.split("\n"):
        s = line.strip()
        if not s:
            continue
        if any(p.search(s) for p in _HEADER_PATTERNS) and len(s) < 160:
            continue  # running header / footer
        if re.fullmatch(r"[\d\s\.\-–/]+", s):
            continue  # bare page numbers
        lines.append(s)
    return "\n".join(lines)


def remove_repeated_lines(pages: List[str], min_ratio: float = 0.5) -> List[str]:
    """Remove lines that repeat on most pages of a document (headers/footers)."""
    if len(pages) < 3:
        return pages
    counts: Counter = Counter()
    for p in pages:
        counts.update(set(l.strip() for l in p.split("\n") if l.strip()))
    limit = max(2, int(len(pages) * min_ratio))
    repeated = {l for l, c in counts.items() if c >= limit and len(l) < 200}
    return ["\n".join(l for l in p.split("\n") if l.strip() not in repeated) for p in pages]


def kazakh_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(c in KAZAKH_LETTERS for c in letters) / len(letters)


def cyrillic_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum("Ѐ" <= c <= "ӿ" for c in letters) / len(letters)


def overlap_ratio(a: Iterable[str], b: Iterable[str]) -> float:
    a, b = set(a), set(b)
    if not a:
        return 0.0
    return len(a & b) / len(a)
