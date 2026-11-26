from app.services.text_units import (
    contains_cjk,
    keyword_coverage,
    query_terms,
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


def test_query_terms_drop_function_words_unless_nothing_remains() -> None:
    assert query_terms("How do I make the Tomato soup?") == ["make", "tomato", "soup"]
    assert query_terms("what is it") == ["what", "is", "it"]


def test_keyword_coverage_counts_content_words() -> None:
    text = "This project searches local documents without sending data to the cloud."

    assert keyword_coverage("local documents without the cloud", text) == 1.0
    assert keyword_coverage("local budget report", text) == 1 / 3
    assert keyword_coverage("red car", "", "red_car.png") == 1.0
    assert keyword_coverage("离线检索", "这是离线检索系统") == 1.0
    assert keyword_coverage("...", text) == 0.0
