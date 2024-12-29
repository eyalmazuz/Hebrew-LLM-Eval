from abc import ABC, abstractmethod

from PyQt5.QtGui import QBrush, QColor, QTextCharFormat, QTextCursor


def highlight_range(original_text_edit, start, end, char_format):
    cursor = original_text_edit.textCursor()
    cursor.setPosition(start)
    cursor.setPosition(end, QTextCursor.KeepAnchor)
    cursor.mergeCharFormat(char_format)


class Highlighter(ABC):
    @abstractmethod
    def highlight(self, original_text, augmented_text, original_text_edit):
        pass


class SentenceRemovalHighlighter(Highlighter):
    def highlight(self, original_text, augmented_text, original_text_edit):
        original_lines = [line.strip() for line in original_text.split(".") if line.strip()]
        augmented_lines = [line.strip() for line in augmented_text.split(".") if line.strip()]

        for line in original_lines:
            if line not in augmented_lines:
                self.highlight_sentence(line, original_text_edit)

    def highlight_sentence(self, sentence, original_text_edit):
        doc = original_text_edit.document()
        full_text = doc.toPlainText()

        candidates = [sentence + ".", sentence]

        highlight_format = QTextCharFormat()
        highlight_format.setBackground(QBrush(QColor(255, 255, 0, 128)))

        for s in candidates:
            idx = full_text.find(s)
            if idx != -1:
                highlight_range(original_text_edit, idx, idx + len(s), highlight_format)
                break


class WordRemovalHighlighter(Highlighter):
    def highlight(self, original_text, augmented_text, original_text_edit):
        original_words = original_text.split()
        augmented_words = augmented_text.split()

        # Use an iterator for the augmented words to ensure positional matching
        augmented_iter = iter(augmented_words)
        augmented_word = next(augmented_iter, None)  # Get the first word from the iterator

        for index, word in enumerate(original_words):
            if word == augmented_word:
                # Move to the next word in augmented_words
                augmented_word = next(augmented_iter, None)
            else:
                # If the word is not matched, it's considered removed
                self.highlight_word(word, index, original_text_edit)

    def highlight_word(self, word, idx, original_text_edit):
        doc = original_text_edit.document()
        full_text = doc.toPlainText().split()

        if idx < len(full_text):
            start_char = sum(len(w) + 1 for w in full_text[:idx])  # +1 for space
            end_char = start_char + len(word)

            highlight_format = QTextCharFormat()
            highlight_format.setBackground(QBrush(QColor(255, 255, 0, 128)))
            highlight_range(original_text_edit, start_char, end_char, highlight_format)


class SentenceShuffleHighlighter(Highlighter):
    def highlight(self, original_text, augmented_text, original_text_edit):
        original_lines = [line.strip() for line in original_text.split(".") if line.strip()]
        augmented_lines = [line.strip() for line in augmented_text.split(".") if line.strip()]

        for i, line in enumerate(original_lines):
            if line in augmented_lines:
                aug_index = augmented_lines.index(line)
                if aug_index != i:
                    self.highlight_sentence(line, original_text_edit)
            else:
                self.highlight_sentence(line, original_text_edit)

    def highlight_sentence(self, sentence, original_text_edit):
        doc = original_text_edit.document()
        full_text = doc.toPlainText()

        candidates = [sentence + ".", sentence]

        highlight_format = QTextCharFormat()
        highlight_format.setBackground(QBrush(QColor(255, 255, 0, 128)))

        for s in candidates:
            idx = full_text.find(s)
            if idx != -1:
                highlight_range(original_text_edit, idx, idx + len(s), highlight_format)
                break


class KeyboardSwapHighlighter(Highlighter):
    def highlight(self, original_text, augmented_text, original_text_edit):
        original_words = original_text.split()
        augmented_words = augmented_text.split()

        for idx, (original_word, augmented_word) in enumerate(zip(original_words, augmented_words)):
            if original_word != augmented_word:
                self.highlight_word(original_word, idx, original_text_edit)

    def highlight_word(self, word, idx, original_text_edit):
        doc = original_text_edit.document()
        full_text = doc.toPlainText().split()

        if idx < len(full_text):
            start_char = sum(len(w) + 1 for w in full_text[:idx])  # +1 for space
            end_char = start_char + len(word)

            highlight_format = QTextCharFormat()
            highlight_format.setBackground(QBrush(QColor(255, 255, 0, 128)))
            highlight_range(original_text_edit, start_char, end_char, highlight_format)
