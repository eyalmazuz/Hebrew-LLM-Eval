import pandas as pd


class DataManager:
    def __init__(self):
        self.df = None
        self.filtered_df = None
        self.current_index = 0

    def load_csv(self, file_name):
        df = pd.read_csv(file_name)
        if "marked" not in df.columns:
            df["marked"] = False
        if all(col in df.columns for col in ["original-text", "augmented-text", "augmentation"]):
            self.df = df
            self.filtered_df = df
            return True
        else:
            return False

    def save_csv(self, file_name):
        if self.df is not None:
            self.df.to_csv(file_name, index=False)

    def clear(self):
        self.df = None
        self.filtered_df = None
        self.current_index = 0

    def get_unique_augmentations(self):
        if self.df is not None:
            return self.df["augmentation"].unique()
        return []

    def apply_filter(self, selected_filter, current_augmentation, current_index):
        if self.df is None:
            return

        new_filtered_df = self.df if selected_filter == "all" else self.df[self.df["augmentation"] == selected_filter]

        # Try to maintain current row if possible
        if new_filtered_df is not None and len(new_filtered_df) > 0:
            if selected_filter == "all":
                # Try to keep same row
                if (
                    current_augmentation is not None
                    and self.filtered_df is not None
                    and 0 <= current_index < len(self.filtered_df)
                ):
                    current_global_index = self.filtered_df.index[current_index]
                    if current_global_index in new_filtered_df.index:
                        self.filtered_df = new_filtered_df
                        self.current_index = new_filtered_df.index.get_loc(current_global_index)
                    else:
                        self.filtered_df = new_filtered_df
                        self.current_index = 0
                else:
                    self.filtered_df = new_filtered_df
                    self.current_index = 0
            else:
                # specific augmentation
                if (
                    current_augmentation == selected_filter
                    and self.filtered_df is not None
                    and 0 <= current_index < len(self.filtered_df)
                ):
                    current_global_index = self.filtered_df.index[current_index]
                    if current_global_index in new_filtered_df.index:
                        self.filtered_df = new_filtered_df
                        self.current_index = new_filtered_df.index.get_loc(current_global_index)
                    else:
                        self.filtered_df = new_filtered_df
                        self.current_index = 0
                else:
                    self.filtered_df = new_filtered_df
                    self.current_index = 0
        else:
            self.filtered_df = new_filtered_df
            self.current_index = 0

    def get_current_augmentation(self, current_index):
        if self.filtered_df is not None and 0 <= current_index < len(self.filtered_df):
            return str(self.filtered_df.iloc[current_index]["augmentation"])
        return None

    def get_row(self, index):
        if self.filtered_df is not None and 0 <= index < len(self.filtered_df):
            return self.filtered_df.iloc[index]
        return None

    def get_current_augmented_text(self, current_index):
        if self.filtered_df is not None and 0 <= current_index < len(self.filtered_df):
            return str(self.filtered_df.iloc[current_index]["augmented-text"])
        return ""

    def apply_changes(self, current_index, new_text):
        if self.filtered_df is not None and 0 <= current_index < len(self.filtered_df):
            current_global_index = self.filtered_df.index[current_index]
            self.df.at[current_global_index, "augmented-text"] = new_text
            self.filtered_df.at[current_global_index, "augmented-text"] = new_text

    def set_marked(self, current_index, checked):
        """
        Updates the 'marked' column for the currently displayed row in both
        the original df and the filtered df.
        """
        if self.filtered_df is not None and 0 <= current_index < len(self.filtered_df):
            current_global_index = self.filtered_df.index[current_index]
            self.df.at[current_global_index, "marked"] = checked
            self.filtered_df.at[current_global_index, "marked"] = checked
