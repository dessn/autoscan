__author__ = 'Danny Goldstein <dgold@berkeley.edu>'
__all__ = ['pix']

'''File I/O for autoScan.'''

import logging
from astropy.io import ascii, fits

def _try_to_open(filename, log=None):
    '''Try to read the fits file `filename`.
    Return the pixel data.'''

    if log is None:
        log = logging.getLogger('')
    log.debug('Reading %s.', filename)
    try:
        hdu = fits.open(filename)
    except IOError as e:
        log.warning('Could not read %s.', filename)
        raise e

    # Byte-swap to make pixel arrays little-endian.
    return hdu[0].data.byteswap(True).newbyteorder()


def pix(*args, **kwargs):
    ''' Open up files in 'args.' Extract and renormalize their
    pixels. Return pixel arrays.

    Parameters:
    -----------
    filenames: An iterable of filenames pointing to fits files that
    can be opened by pyfits.
    
    names: An iterable of strings corresponding to the names of the
    types of the files in the order the files appear in the iterable
    'filenames'.
    '''

    if 'log' not in kwargs:
        log = logging.getLogger('')
    else:
        log = kwargs['log']
    return [_try_to_open(fname, log) for fname in args]

