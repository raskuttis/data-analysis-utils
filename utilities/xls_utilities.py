# _*_ coding: utf-8
"""
    Modification of csvwriter to work better with unicode
"""

import json
import logging
import itertools
import xlsxwriter
import sys
import six

if sys.version_info[0] < 3:
    from itertools import izip_longest as zip_longest
else:
    from itertools import zip_longest

class ReportWorkbook(xlsxwriter.Workbook):
    """
        Child class to the xlsxwriter workbook, which has predefined formats
    """

    def __init__(self, filename, options=None):
        """
        Init method for the Report Workbook. Initializes the workbook and adds
        formatting
        """

        # Call Workbook Init
        xlsxwriter.Workbook.__init__(self, filename, options=options)

        # Initialize formats for the workbook
        self.header_fmt = self.add_format({"bold": True, "center_across": True,
                                          "bg_color": "#2CA918", "font_color": "white"})
        self.highlight_fmt = self.add_format({"bold": True, "center_across": True,
                                             "bg_color": "#FF0000", "font_color": "white",
                                             "text_wrap": True})
        self.std_fmt = self.add_format({"center_across": True, "text_wrap": True})

class XLSXDictWriter(object):
    """
        Class to handle the writing of dictionaries to disk as excel spreadsheets
    """

    def __init__(self, file_wb, file_ws_name, keys, dummy_values=None, write_header=True,
                 key_split="::", freeze_panes=None):
        """
        Init method for DictWriter. Primarily just used to initialize the worksheet,
        the header and the formatting
        """

        # Initialize new worksheet
        self.file_wb = file_wb
        if len(file_ws_name) > 30:
            file_ws_name = file_ws_name[:30]
        self.file_ws = self.file_wb.add_worksheet(file_ws_name)

        # Initialize counter
        self.n_written_lines = 0

        # Log whether or not there are dummy values defined for this set of keys
        self.header_keys = keys
        self.key_dummies = dummy_values
        # Initialize column widths
        self.column_widths = {header_key: 10 for header_key in self.header_keys}
        self.n_header_lines = self.initialize_header(dummy_values=dummy_values,
                                                     write_header=write_header,
                                                     key_split=key_split)

        # Freeze panes
        self.freeze_panes = None
        if freeze_panes:
            self.freeze_panes = freeze_panes
            self.file_ws.freeze_panes(*freeze_panes)

    def initialize_header(self, dummy_values=None, write_header=True, key_split=None,
                          header_fmt=None):
        """
        Method to initialize the header and the dummy values based on the keys
        """
        self.key_dummies = {key: " " for key in self.header_keys}
        if dummy_values:
            for key in self.key_dummies:
                if key in dummy_values:
                    self.key_dummies[key] = dummy_values[key]
                else:
                    logging.warning("Key %s has no defined dummy value so defaulting to empty",
                                    key)
        else:
            logging.warning("No dummy values set for any headers so defaulting to empty")

        logging.info("Initialized CSV Writer with headers %s and corresponding dummies %s",
                     json.dumps(self.header_keys), json.dumps(self.key_dummies))

        if not header_fmt:
            header_fmt = self.file_wb.header_fmt

        row_num = 0
        if write_header:
            logging.info("Writing header %s to file", json.dumps(self.header_keys))
            if key_split:
                key_list = [key.split(key_split) for key in self.header_keys]
                key_list = map(list, zip_longest(*key_list))
                for row_num, key_row in enumerate(key_list):
                    xls_row = []
                    last_key = ""
                    for key in key_row:
                        if key == None:
                            xls_row.append(" ")
                        elif key == last_key:
                            xls_row.append("")
                        else:
                            xls_row.append(key)
                            last_key = key
                    xls_key_dict = {header_key: key_name
                                    for header_key, key_name in zip(self.header_keys, xls_row)}
                    self.write_line(xls_key_dict, cell_format=header_fmt)
            else:
                row_num = 0
                self.write_line({key_name: key_name for key_name in self.header_keys},
                                cell_format=header_fmt)

        return row_num

    def update_width(self, header_key, value):
        """
        Function to update column widths based on a header key and a value
        """

        if isinstance(value, six.text_type):
            updated_width = len(value)
        else:
            updated_width = len(str(value))
        if updated_width > self.column_widths.get(header_key, 0):
            self.column_widths[header_key] = updated_width

    def autofit(self, max_freeze_width=150, max_col_width=50):
        """
        Function to autofit columns
        """
        pane_width = 0
        for col_num, header_key in enumerate(self.header_keys):
            col_width = self.column_widths.get(header_key, 10)
            if col_width > max_col_width:
                col_width = max_col_width
            pane_width += col_width
            if self.freeze_panes and pane_width > max_freeze_width:
                if self.freeze_panes[1] > col_num:
                    self.freeze_panes = (self.freeze_panes[0], col_num)
                    self.file_ws.freeze_panes(*self.freeze_panes)
                    pane_width = -1.0e10
            self.file_ws.set_column(col_num, col_num, 1.25 * col_width)

    def write_line(self, row_dict, cell_format=None):
        """
        Function to write a row i.e a single dict to file
        """

        if not cell_format:
            cell_format = self.file_wb.std_fmt

        for col_num, header_key in enumerate(self.header_keys):
            dummy_value = self.key_dummies.get(header_key, " ")
            written_value = row_dict.get(header_key, dummy_value)
            self.file_ws.write(self.n_written_lines, col_num, written_value,
                               cell_format)
            self.update_width(header_key, written_value)
        self.n_written_lines += 1
