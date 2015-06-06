__author__ = 'Danny Goldstein <dgold@berkeley.edu>'

'''autoScan helper classes + methods + constants.'''

from textwrap import dedent
import sys

# Class constants.
PIXEL_SCALE = 0.277  # arcesconds per pixel
FILTCAT_HEADER = dedent(
    '''# AUTOSCAN / FILTEROBJ ASSOCIATION CATALOG
       # 1 SNOBJID (ID Number)
       # 2 RA (deg)
       # 3 DEC (deg)
       # 4 MAG (mag)''')

# Exceptions.
class FilterObjError(Exception):
    '''Raised on badly formatted filterObj input.'''
    pass

class InvalidSequenceException(Exception):
    """ Thrown on an attempt to split a zero-length sequence."""
    pass

class DBException(Exception):
    '''Thrown when db_connect cannot connect to a database.'''
    pass

class NoDataException(Exception):
    """ Thrown when autoScan cannot find any data to process. """
    pass

# Helper methods.
def print_funcname(func):
    '''Decorator that traces the execution of feature computation
    functions in debug mode.'''

    def newfunc(oper, sci, fh, f, sex, stampx, impix):
        print 'executing {0}'.format(func.__name__)
        return func(oper, sci, fh, f, sex, stampx, impix)

    return newfunc

def keyed_file_to_dict(f):
    '''take a file object that looks like this:

         K1:   V1
         K2:   V2
         K3:   V3
         K4:   V4

      and return its dict form, like this:
         dict_form[K1] = V1, etc. '''

    dict_form = {}
    for row in f:
        split_row = [s.strip() for s in row.split()]
        for i, word in enumerate(split_row):
            if word.endswith(':'):
                key = word[:-1]
                value = split_row[i + 1]
                dict_form[key] = value
    # rewind file
    f.seek(0, 0)

    return dict_form

def center(string, ll):
    '''Center a string on a line of length `ll`.'''
    sl = len(string)
    if sl >= ll:
        return string
    else:
        diff = ll - sl
        pad = ' ' * (diff / 2)
        return pad + string + pad

def split_seq(seq, procs):
    """Split a sequence into `procs` subsequences."""
    newseq = []
    proc_load = len(seq) / procs
    for i in range(procs):
        if i == procs - 1:
            newseq.append(seq[i * proc_load:])
        else:
            newseq.append(seq[i * proc_load: (i + 1) * proc_load])
    return newseq
