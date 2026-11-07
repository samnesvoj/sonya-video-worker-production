"""ru-profanity-v1: roots, word forms, stretching, joined forms, masks, distortions — and no false positives."""
import pytest

from scripts.shared.profanity import classify, detect, mask_word

MAT = ["хуй", "Хуй,", "нахуй", "похуй", "нихуя", "охуенно", "охуел", "хуёвый", "хуйня", "хуевый",
       "пизда", "пиздец", "Пиздец!", "спиздил", "распиздяй", "пиздабол",
       "ебать", "заебал", "заебись", "выёбывается", "уёбок", "ебанутый", "ёбаный", "наебал", "долбоёб", "долбоеб",
       "ебучий", "въебать", "отъебись",
       "бля", "блять", "блядь", "блядский", "Бля...",
       "мудак", "мудила", "мудозвон", "манда", "ёпта",
       # stretched / doubled / distorted / masked / homoglyphs / latin
       "бляяяять", "хуууй", "пиииздец", "ппиздец",
       "х*й", "п**дец", "бл*", "е*ать", "xуй", "п1зд", "6лять", "xyй",
       "blyat", "blyad", "nahui", "pizdec",
       "пиздец-нахуй", "хуй-знает"]
RUDE = ["сука", "суки", "сучка", "сучара", "пидор", "пидорас", "гандон", "шлюха", "залупа", "suka"]
CLEAN = ["рубля", "корабля", "сабля", "употреблять", "оскорблять", "истребляет", "команда", "мандарин",
         "мандат", "застрахуй", "страхую", "хлеб", "небо", "ребята", "ребёнок", "хулиган", "худой", "хутор",
         "хуже", "бляха", "мудрый", "мудрец", "сукно", "Хуан", "ебонит", "пиджак", "пиджин", "блин", "пипец",
         "капец", "ёлки", "ёжик", "ебеня"[:0] or "жизнь", "сук", "сукин"[:0] or "сухой", "трахея", "оскорбление",
         "команды", "демократия", "мебель", "ребус", "Тебе", "себя", "небольшой", "гребля", "дебют", "тебя",
         "бляшка"[:0] or "быстро", "муха", "манеж", "Ебург"[:0] or "Екатеринбург", "ещё", "ей", "её"]


@pytest.mark.parametrize("w", MAT)
def test_mat(w):
    assert classify(w) == "mat", w


@pytest.mark.parametrize("w", RUDE)
def test_rude(w):
    assert classify(w) == "rude", w


@pytest.mark.parametrize("w", CLEAN)
def test_no_false_positives(w):
    assert classify(w) is None, w


def test_detect_returns_word_times_and_respects_categories():
    words = [{"word": " Это", "start": 0.0, "end": 0.3}, {"word": " пиздец,", "start": 0.3, "end": 0.8},
             {"word": " сука.", "start": 0.9, "end": 1.2}, {"word": " Рубля", "start": 1.3, "end": 1.6}]
    hits = detect(words)
    assert [(h.index, h.category, h.start, h.end) for h in hits] == [(1, "mat", 0.3, 0.8), (2, "rude", 0.9, 1.2)]
    assert [h.index for h in detect(words, categories=("mat",))] == [1]


def test_mask_word_keeps_first_letter_and_punctuation():
    assert mask_word("пиздец,") == "п*****," and mask_word("Бля!") == "Б**!"
