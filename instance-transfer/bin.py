from math import log, inf


class Bin(object):
    """
    bin class to hold occurences. For continouous attributes, the value ranges will be set.
    For nominal values, the value will be set
    """

    def __init__(self, min_val=None, max_val=None):
        self.min_val = min_val
        self.max_val = max_val

        self.count = 0

    def add_to_bin(self, weight):
        self.count += weight

    def fits_in_bin(self, value):
        return self.min_val <= value < self.max_val

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "<Min: {}, Max: {}, Count: {}>".format(
            self.min_val, self.max_val, self.count
        )
