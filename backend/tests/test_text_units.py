from app.services.text_units import (
    contains_cjk,
    iter_search_terms,
    iter_unit_spans,
    space_cjk_characters,
)


def test_unit_spans_split_words_and_cjk_characters() -> None:
    text = "Hello world 你好"

    units = [text[start:end] for start, end in iter_unit_spans(text)]

    assert units == ["Hello", "world", "你", "好"]


def test_search_terms_drop_punctuation_and_keep_cjk_runs() -> None:
    assert list(iter_search_terms("Revenue, 2024! 年度报告?")) == [
        "Revenue",
        "2024",
        "年度报告",
    ]


def test_space_cjk_characters_and_detection() -> None:
    assert space_cjk_characters("年报a").split() == ["年", "报", "a"]
    assert contains_cjk("abc年")
    assert not contains_cjk("abc")
