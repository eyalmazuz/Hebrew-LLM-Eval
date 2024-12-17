import pickle
import tempfile

import networkx
import pytest

from src.augmentations import Augmentation, KeyboardSwap, get_augmentations


@pytest.fixture
def a_e_graph():
    with tempfile.TemporaryDirectory() as tmpdir:
        graph = networkx.Graph()
        graph.add_edges_from([("a", "e")])
        with open(f"{tmpdir}/graph.gpickle", "wb") as fd:
            pickle.dump(graph, fd)

        yield f"{tmpdir}/graph.gpickle"


@pytest.fixture
def empty_graph():
    with tempfile.TemporaryDirectory() as tmpdir:
        graph = networkx.Graph()
        with open(f"{tmpdir}/graph.gpickle", "wb") as fd:
            pickle.dump(graph, fd)

        yield f"{tmpdir}/graph.gpickle"


@pytest.fixture
def text():
    return "the word replacement probability can only be between 0 and 1"


@pytest.mark.parametrize("types", [["keyboard-swapping"]])
def test_get_keyboard_swap_augmentation(types, empty_config):
    augmentations = get_augmentations(types, augmentations_config=empty_config)
    assert len(augmentations) == 1
    assert isinstance(augmentations[0], Augmentation)
    assert isinstance(augmentations[0], KeyboardSwap)


def test_keyboard_swap_name():
    augmentation = KeyboardSwap()
    assert str(augmentation) == "keyboard-swapping"


def test_test_no_chage(empty_graph, text):
    augmentation = KeyboardSwap(graph_path=empty_graph, word_p=0.0, char_p=0.0)

    augmented_text = augmentation(text)
    assert augmented_text == text


def test_test_a_changed(a_e_graph, text):
    augmentation = KeyboardSwap(graph_path=a_e_graph, word_p=1.0, char_p=1.0, insert_double=False)

    augmented_text = augmentation(text)
    assert augmented_text != text
    assert augmented_text == "tha word raplecamant probebility cen only ba batwaan 0 end 1"
