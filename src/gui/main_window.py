from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAction,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from src.gui.data_manager import DataManager
from src.gui.highlighter import (
    KeyboardSwapHighlighter,
    SentenceRemovalHighlighter,
    SentenceShuffleHighlighter,
    WordRemovalHighlighter,
)

# Dictionary mapping augmentation type to strategy
HIGHLIGHTER_STRATEGIES = {
    "sentence-removal": SentenceRemovalHighlighter(),
    "word-removal": WordRemovalHighlighter(),
    "span-removal": WordRemovalHighlighter(),
    "sentence-shuffle": SentenceShuffleHighlighter(),
    "keyboard-swapping": KeyboardSwapHighlighter(),
}


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("CSV Augmentation Viewer")

        self.data_manager = DataManager()  # Manages data loading/saving/filtering
        self.current_index = 0
        self.data_changed = False

        # Toolbar
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)

        load_action = QAction("Load CSV", self)
        load_action.triggered.connect(self.load_csv)
        toolbar.addAction(load_action)

        save_action = QAction("Save CSV", self)
        save_action.triggered.connect(self.save_csv)
        toolbar.addAction(save_action)

        # Filter combo box
        self.filter_combo = QComboBox()
        self.filter_combo.addItem("all")
        self.filter_combo.currentTextChanged.connect(self.apply_filter)
        self.filter_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.filter_combo.setMinimumContentsLength(15)
        toolbar.addWidget(self.filter_combo)

        font_size_spinbox = QSpinBox()
        font_size_spinbox.setRange(6, 72)  # Set a sensible range of font sizes
        font_size_spinbox.setValue(14)  # Default font size
        font_size_spinbox.setToolTip("Change Font Size")
        font_size_spinbox.valueChanged.connect(self.change_font_size)
        toolbar.addWidget(font_size_spinbox)

        # Central layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Augmentation label
        self.augmentation_label = QLabel("Augmentation Type: ")
        main_layout.addWidget(self.augmentation_label, alignment=Qt.AlignCenter)

        # Text layout
        text_layout = QHBoxLayout()

        # Original text
        original_layout = QVBoxLayout()
        original_title = QLabel("Original Text")
        original_title.setAlignment(Qt.AlignCenter)
        self.original_text_edit = QTextEdit()
        self.original_text_edit.setReadOnly(True)
        self.original_text_edit.setLayoutDirection(Qt.RightToLeft)

        font = self.original_text_edit.font()
        font.setPointSize(14)  # Choose your desired size
        self.original_text_edit.setFont(font)

        original_layout.addWidget(original_title)
        original_layout.addWidget(self.original_text_edit)

        # Augmented text
        augmented_layout = QVBoxLayout()
        augmented_title = QLabel("Augmented Text")
        augmented_title.setAlignment(Qt.AlignCenter)
        self.augmented_text_edit = QPlainTextEdit()
        self.augmented_text_edit.setLayoutDirection(Qt.RightToLeft)
        self.augmented_text_edit.textChanged.connect(self.on_augmented_text_changed)

        font = self.augmented_text_edit.font()
        font.setPointSize(14)
        self.augmented_text_edit.setFont(font)

        augmented_layout.addWidget(augmented_title)
        augmented_layout.addWidget(self.augmented_text_edit)

        text_layout.addLayout(original_layout)
        text_layout.addLayout(augmented_layout)

        main_layout.addLayout(text_layout)

        # Buttons layout
        button_layout = QHBoxLayout()

        self.prev_button = QPushButton("Previous")
        self.prev_button.clicked.connect(self.show_previous_row)
        self.prev_button.setEnabled(False)
        button_layout.addWidget(self.prev_button)

        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.show_next_row)
        self.next_button.setEnabled(False)
        button_layout.addWidget(self.next_button)

        self.apply_button = QPushButton("Apply")
        self.apply_button.clicked.connect(self.apply_changes)
        self.apply_button.setEnabled(False)
        button_layout.addWidget(self.apply_button)

        main_layout.addLayout(button_layout)

    def change_font_size(self, size):
        # Change the font size for both original_text_edit and augmented_text_edit
        original_font = self.original_text_edit.font()
        original_font.setPointSize(size)
        self.original_text_edit.setFont(original_font)

        augmented_font = self.augmented_text_edit.font()
        augmented_font.setPointSize(size)
        self.augmented_text_edit.setFont(augmented_font)

    def load_csv(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (*.csv)")
        if file_name:
            success = self.data_manager.load_csv(file_name)
            if success:
                self.populate_filter_combo()
                self.current_index = 0
                self.display_row(self.current_index)
                self.update_navigation_buttons()
            else:
                self.data_manager.clear()
                self.clear_display()
                self.update_navigation_buttons()
                self.augmentation_label.setText("Augmentation Type: ")

    def save_csv(self):
        if self.data_manager.df is not None:
            file_name, _ = QFileDialog.getSaveFileName(self, "Save CSV File", "", "CSV Files (*.csv)")
            if file_name:
                self.data_manager.save_csv(file_name)

    def apply_filter(self, selected_filter):
        if self.data_manager.df is None:
            return
        # Keep current augmentation before filter
        current_augmentation = self.data_manager.get_current_augmentation(self.current_index)

        self.data_manager.apply_filter(selected_filter, current_augmentation, self.current_index)

        if self.data_manager.filtered_df is not None and len(self.data_manager.filtered_df) > 0:
            # After applying filter, we must update our current_index to what data_manager returned
            self.current_index = self.data_manager.current_index
            self.display_row(self.current_index)
        else:
            self.clear_display()
            self.augmentation_label.setText("Augmentation Type: ")
            self.current_index = 0

        self.update_navigation_buttons()

    def populate_filter_combo(self):
        self.filter_combo.clear()
        self.filter_combo.addItem("all")
        for aug in self.data_manager.get_unique_augmentations():
            self.filter_combo.addItem(str(aug))

    def display_row(self, index):
        row = self.data_manager.get_row(index)
        if row is not None:
            original_text = str(row["original-text"])
            augmented_text = str(row["augmented-text"])
            augmentation_type = str(row["augmentation"])

            self.augmentation_label.setText(f"Augmentation Type: {augmentation_type}")

            self.original_text_edit.setPlainText(original_text)
            self.augmented_text_edit.setPlainText(augmented_text)

            # Use strategy if exists
            highlighter = HIGHLIGHTER_STRATEGIES.get(augmentation_type)
            if highlighter is not None:
                highlighter.highlight(original_text, augmented_text, self.original_text_edit)

            self.data_changed = False
            self.apply_button.setEnabled(False)
        else:
            self.clear_display()

    def clear_display(self):
        self.original_text_edit.clear()
        self.augmented_text_edit.clear()

    def update_navigation_buttons(self):
        if self.data_manager.filtered_df is None or len(self.data_manager.filtered_df) == 0:
            self.prev_button.setEnabled(False)
            self.next_button.setEnabled(False)
            return

        self.prev_button.setEnabled(self.current_index > 0)
        self.next_button.setEnabled(self.current_index < len(self.data_manager.filtered_df) - 1)

    def show_next_row(self):
        if self.data_manager.filtered_df is not None and self.current_index < len(self.data_manager.filtered_df) - 1:
            self.current_index += 1
            self.display_row(self.current_index)
            self.update_navigation_buttons()

    def show_previous_row(self):
        if self.data_manager.filtered_df is not None and self.current_index > 0:
            self.current_index -= 1
            self.display_row(self.current_index)
            self.update_navigation_buttons()

    def on_augmented_text_changed(self):
        if self.data_manager.filtered_df is not None and len(self.data_manager.filtered_df) > 0:
            current_aug_text = self.augmented_text_edit.toPlainText()
            original_aug_text = self.data_manager.get_current_augmented_text(self.current_index)
            self.data_changed = current_aug_text != original_aug_text
            self.apply_button.setEnabled(self.data_changed)

    def apply_changes(self):
        if self.data_changed and self.data_manager.filtered_df is not None and len(self.data_manager.filtered_df) > 0:
            new_text = self.augmented_text_edit.toPlainText()
            self.data_manager.apply_changes(self.current_index, new_text)

            self.data_changed = False
            self.apply_button.setEnabled(False)
