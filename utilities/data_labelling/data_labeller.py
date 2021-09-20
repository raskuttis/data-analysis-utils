"""
Purpose

Classes to rapidly label CSV datasets in jupyter notebooks

"""

import pandas as pd
import ipywidgets as widgets
import datetime
from IPython.display import display, clear_output


class JupyterDataLabeller(object):
    """
    Class to read in a csv dataset and rapidly label it from inside of a jupyter
    notebook. Will return an output csv that contains an extra boolean column for
    whether that column has been labelled and then additional columns containing
    the labellings
    """

    def __init__(self, working_dir, input_csv, label_names, overwrite_flag=False,
                 backup=True):
        """
        Init method for Jupyter Data Labeller

        :param working_dir: Directory in which input and output files are located
        :param input_csv: CSV that we want to label
        :param label_names: Names of the labels that we want to add to CSV
        :param overwrite_flag: Boolean flag for whether or not to overwrite
        :param backup: Whether or not to create a backup of the original file
        existing labels
        """

        self.working_dir = working_dir
        self.label_df = pd.read_csv(f"{working_dir}/{input_csv}")
        self.label_names = label_names
        self.first_index = self.get_initial_labels(overwrite_flag)
        if backup:
            current_ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            self.label_df.to_csv(f"{working_dir}/{input_csv}.{current_ts}.bkp")

    def get_initial_labels(self, overwrite_flag):
        """
        Function to check which rows in the CSV have already been labelled

        :param overwrite_flag: Boolean flag for whether or not to overwrite
        """

        # First check whether the CSV has been touched at all i.e. does it have
        # a "Labelled" row
        if "Labelled" in self.label_df.columns and not overwrite_flag:
            self.label_df["Labelled"] = self.label_df["Labelled"].fillna(False)
            first_index = self.label_df[(self.label_df["Labelled"] == False)].index[0] - 1
            # Initialize any new columns
            for name in self.label_names:
                if name not in self.label_df.columns:
                    self.label_df[name] = ""
        # If not, then add one
        else:
            self.label_df["Labelled"] = False
            for name in self.label_names:
                self.label_df[name] = ""
            first_index = -1

        return first_index

    def label_new_data(self, output_csv, visualize_row, continuous_write=True, starting_index=None):
        """
        Label new data in the CSV from the provided starting index

        :param output_csv: Name of the CSV output
        :param visualize_row: Function for how to visualize the row of CSV data so as to best
        facilitate labelling
        :param continuous_write: Whether or not we're continuously writing or just writing once
        at the end of labelling
        :param starting_index: Index at which to start labelling
        :return:
        """

        # Define button to continue labelling and widgets for labels
        label_defs = {name: widgets.Checkbox(description=name) for name in self.label_names}
        pb = widgets.Button(description="Begin Labelling Next", disabled=False, button_style="",
                            tooltip="Next")

        # Initialize by displaying a button that we click to continue labelling
        display(widgets.VBox([pb]))

        # Define what happens on the button click
        self.current_index = self.first_index
        if starting_index:
            self.current_index = starting_index - 1
        self.needs_update = False

        def on_button_clicked(b):

            if self.current_index < len(self.label_df):

                # First clear the current button
                clear_output(wait=True)

                # Write to output if this row needs to be updated/labelled
                if self.needs_update:
                    for label_name, label_def in label_defs.items():
                        self.label_df.loc[self.current_index, label_name] = label_def.value
                    self.label_df.loc[self.current_index, "Labelled"] = True
                    if continuous_write:
                        self.label_df.to_csv(f"{self.working_dir}/{output_csv}")

                # Then augment and continue labelling
                self.current_index += 1
                row = self.label_df.loc[self.current_index]
                labelled = row.get("Labelled", False)

                if not labelled:
                    print(f"Starting {self.current_index} of {len(self.label_df)}")
                    visualize_row(row)
                    ui = widgets.VBox([pb, widgets.VBox(list(label_defs.values()))])
                    self.needs_update = True
                else:
                    self.needs_update = False
                    ui = widgets.VBox([pb])

                display(ui)

        pb.on_click(on_button_clicked)