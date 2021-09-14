"""
Purpose

Exceptions for the AWS Comprehend classes
"""

class LowUnitSizeError(Exception):
    def __init__(self, *args):
        self.message = None
        if args:
            self.message = args[0]

    def __str__(self):
        if self.message:
            return 'LowUnitSizeError, {0} '.format(self.message)
        else:
            return 'LowUnitSizeError raised'


class HighUnitSizeError(Exception):
    def __init__(self, *args):
        self.message = None
        if args:
            self.message = args[0]

    def __str__(self):
        if self.message:
            return 'HighUnitSizeError, {0} '.format(self.message)
        else:
            return 'HighUnitSizeError raised'


class HighByteSizeError(Exception):
    def __init__(self, *args):
        self.message = None
        if args:
            self.message = args[0]

    def __str__(self):
        if self.message:
            return 'HighByteSizeError, {0} '.format(self.message)
        else:
            return 'HighByteSizeError raised'