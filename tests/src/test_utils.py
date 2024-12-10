import json
import os
import tempfile

import pytest

from src.utils import extract_texts, load_data


@pytest.fixture
def summaries():
    return [
        {"text_raw": "aaa", "summary": "a11", "metadata": {}},
        {"text_raw": "bbb", "summary": "b11", "metadata": {"ai_summary": "b22"}},
        {"text_raw": "ccc", "summary": "c11", "metadata": {"ai_summary": "c22"}},
        {"text_raw": "ddd", "summary": None, "metadata": {"ai_summary": "d22"}},
        {"text_raw": "", "summary": "", "metadata": {}},
    ]


def test_load_data_valid_path():
    path = "./data/summarization-7-heb.jsonl"
    summaries = load_data(path)
    assert summaries is not None


def test_load_invalid_path():
    path = "./data/fake.jsonl"
    with pytest.raises(FileNotFoundError) as excinfo:
        load_data(path)
    assert excinfo.type is FileNotFoundError


def test_load_invalid_json():
    with tempfile.TemporaryDirectory() as tmpdirname:
        path = os.path.join(tmpdirname, "invald.json")
        with open(path, "w") as fd:
            fd.write("""
            {\"key: 12},
            {\"good\": 12}
            """)
        with pytest.raises(json.JSONDecodeError) as excinfo:
            load_data(path)
        assert excinfo.type is json.JSONDecodeError


def test_extract_texts_only_summaries(summaries):
    texts = extract_texts(summaries, only_summaries=True, use_ai_summaries=False)

    assert len(texts) == 3
    assert texts == ["a11", "b11", "c11"]


def test_extract_all(summaries):
    texts = extract_texts(summaries, only_summaries=False, use_ai_summaries=True)

    assert len(texts) == 10
    assert texts == ["aaa", "a11", "bbb", "b22", "b11", "ccc", "c22", "c11", "ddd", "d22"]


def test_extract_no_ai(summaries):
    texts = extract_texts(summaries, only_summaries=False, use_ai_summaries=False)

    assert len(texts) == 7
    assert texts == ["aaa", "a11", "bbb", "b11", "ccc", "c11", "ddd"]
