"""
    Modification of csvwriter to work better with unicode
"""

from __future__ import absolute_import
import json
import logging
import csv
import codecs
import sys
import itertools

from six.moves import map
if sys.version_info[0] < 3:
    import cStringIO
    from itertools import izip_longest as zip_longest
else:
    from io import StringIO
    from itertools import zip_longest

class UTF8Recoder(object):
    """
        Iterator that reads an encoded stream and reencodes the input to UTF-8
    """
    def __init__(self, f, encoding):
        """
        Init for UTF8 Recorder
        """
        self.reader = codecs.getreader(encoding)(f)

    def __iter__(self):
        """
        Iter for UTF8 Recorder
        """
        return self

    def next(self):
        """
        Next for UTF8 Recorder
        """
        return self.reader.next().encode("utf-8")

class UnicodeReader(object):
    """
        A CSV reader which will iterate over lines in the CSV file "f",
        which is encoded in the given encoding.
    """

    def __init__(self, f, dialect=csv.excel, encoding="utf-8", **kwds):
        """
        Init for Unicode Reader
        """
        f = UTF8Recoder(f, encoding)
        self.reader = csv.reader(f, dialect=dialect, **kwds)

    def next(self):
        """
        Next for Unicode Reader
        """
        row = self.reader.next()
        if sys.version_info[0] < 3:
            return [str(s).encode("utf-8") for s in row]
        else:
            return [str(s, "utf-8") for s in row]

    def __iter__(self):
        """
        Iter for Unicode Reader
        """
        return self

class UnicodeWriter(object):
    """
        A CSV writer which will write rows to CSV file "f",
        which is encoded in the given encoding.
    """

    def __init__(self, f, dialect=csv.excel, encoding="utf-8", **kwds):
        """
        Init for Unicode Writer
        """
        # Redirect output to a queue
        if sys.version_info[0] < 3:
            self.queue = cStringIO.StringIO()
        else:
            self.queue = StringIO()
        self.writer = csv.writer(self.queue, dialect=dialect, **kwds)
        self.stream = f
        self.encoder = codecs.getincrementalencoder(encoding)()

    def writerow(self, row):
        """
        Modified writerow function for Unicode
        """
        if sys.version_info[0] < 3:
            self.writer.writerow([str(s).encode("utf-8") for s in row])
        else:
            self.writer.writerow([str(s) for s in row])
        
        # Fetch UTF-8 output from the queue ...
        data = self.queue.getvalue()
        #data = data.decode("utf-8")
        # ... and reencode it into the target encoding
        #data = self.encoder.encode(data)
        # write to the target stream
        self.stream.write(data)
        # empty queue
        self.queue.truncate(0)

    def writerows(self, rows):
        """
        Modified writerows function for Unicode
        """
        for row in rows:
            self.writerow(row)

class DictWriter(object):
    """
        Class to handle the writing of dictionaries to disk
    """

    def __init__(self, file_stream, keys, dialect=csv.excel,
                 encoding="utf-8", dummy_values=None, write_header=True,
                 key_split=None, **kwds):
        """
        Init method for DictWriter. Primarily just used to initialize the filename
        and create the file if it doesn't exist already
        """
        # Create a UnicodeWriter object
        if sys.version_info[0] < 3:
            self.writer = UnicodeWriter(file_stream, dialect=dialect, encoding=encoding, **kwds)
        else:
            self.writer = csv.writer(file_stream, dialect=dialect, **kwds)


        # Log whether or not there are dummy values defined for this set of keys
        self.header_keys = keys
        self.key_dummies = dummy_values
        self.initialize_header(dummy_values=dummy_values, write_header=write_header,
                               key_split=key_split)

        # Initialize counter
        self.n_written_lines = 0

    def initialize_header(self, dummy_values=None, write_header=True, key_split=None):
        """
        Method to initialize the header and the dummy values based on the keys
        """
        self.key_dummies = {key: "" for key in self.header_keys}
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

        if write_header:
            logging.info("Writing header %s to file", json.dumps(self.header_keys))
            if key_split:
                key_list = [key.split(key_split) for key in self.header_keys]
                key_list = list(map(list, zip_longest(*key_list)))
                for key_row in key_list:
                    self.writer.writerow(key_row)
            else:
                self.writer.writerow(self.header_keys)

    def write_line(self, line_dict):
        """
        Writes line specified by input file to disk using the predefined keys
        """

        line_list = [(line_dict.get(key, self.key_dummies.get(key, "")))
                     for key in self.header_keys]
        self.writer.writerow(line_list)
        self.n_written_lines += 1

class DictReader(object):
    """
        Class to handle the reading of dictionaries from disk
    """

    def __init__(self, file_stream, keys=None, dialect=csv.excel,
                 encoding="utf-8", dummy_values=None, read_header=True,
                 **kwds):
        """
        Init method for DictReader. Primarily just used to initialize the headers
        """
        # Create a UnicodeWriter object
        if sys.version_info[0] < 3:
            self.reader = UnicodeReader(file_stream, dialect=dialect, encoding=encoding,
                                        **kwds)
        else:
            self.reader = csv.reader(file_stream, dialect=dialect, **kwds)

        # Log whether or not there are dummy values defined for this set of keys
        self.header_keys = keys
        if not self.header_keys and read_header:
            self.initialize_header()

        # Initialize counter
        self.n_read_lines = 0

    def initialize_header(self):
        """
        Method to initialize the header based on the first line of the CSV
        """

        row = self.reader.next()
        self.header_keys = row

    def __next__(self):
        """
        Next for Dict Reader
        """
        row = next(self.reader)
        self.n_read_lines += 1
        return {key: item for key, item in zip(self.header_keys, row)}

    # Declaration of next for Py2 Compatibility
    next = __next__

    def __iter__(self):
        """
        Iter for Unicode Reader
        """
        return self
