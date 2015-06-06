__author__ = 'Danny Goldstein <dgold@berkeley.edu>'
__ml_version__ = 3

'''Feature extraction methods for autoScan.'''

import os
from textwrap import dedent
from astropy.coordinates import SkyCoord
from astropy.table import Column
from astropy import units as u
from scipy.optimize import fmin_l_bfgs_b
import numpy as np
from scipy.ndimage.filters import median_filter
import logging
from joblib import Parallel, delayed
from itertools import chain
import sep
import sys
from numpy.lib import recfunctions

from . import io
from . import utils
from . import db_connect

# Numeric codes for bands.
bm = {'g':0, 'r':1, 'i':2, 'z':3}

# Features
FEATS = ['gflux', 'diffsumrn', 'numnegrn', 'mag',
         'a_image', 'b_image', 'flags', 'mag_ref', 
         'mag_ref_err', 'a_ref', 'b_ref', 'bandnum',
         'n2sig3', 'n3sig3', 'n2sig5', 'n3sig5',
         'n3sig5shift', 'n2sig5shift', 'n2sig3shift',
         'n3sig3shift', 'ellipticity', 'nn_dist_renorm',
         'maglim', 'min_distance_to_edge_in_new', 'ccdid',
         'snr', 'mag_from_limit', 'magdiff', 'spread_model',
         'spreaderr_model', 'flux_ratio', 'lacosmic',
         'gauss', 'scale', 'amp', 'colmeds', 'maskfrac',
         'l1']

FEATS_LEGACY = ['b_image',
                'mag_ref', 'mag_ref_err',
                'n2sig3', 'n3sig3', 'n2sig5',
                'n3sig5', 'ellipticity',
                'nn_dist_renorm',
                'seeing_ratio', 'good_cand_density',
                'extracted', 'obsaved',
                'gauss', 'scale', 'amp', 'snr',
                'magdiff', 'n2sig5shift',
                'n2sig3shift', 'n3sig3shift', 'n3sig5shift',
                'diffsum', 'numneg', 'chisq',
                'gflux']

# Feature extraction methods.
def gflux(img, cat, oper, sci, imgrn):
    '''Return the ratio of the flux of the object in the difference image
    plus the flux of the host galaxy in the search image to the flux of
    the object in the difference image.'''
    return (oper['galflux'] + oper['flux']) / oper['flux']


def diffsumrn(img, cat, oper, sci, imgrn):
    '''The sum of the pixel intensities in a 5x5 pixel box centered on the
    object in the difference image.'''
    return imgrn['diff'][10: 15, 10: 15].sum()

def numnegrn(img, cat, oper, sci, imgrn):
    '''The number of negative pixels in a 7x7 pixel box centered on the
    object in the difference image.'''
    return len(np.argwhere(imgrn['diff'][9: 16, 9: 16] < 0))

def mag(img, cat, oper, sci, imgrn):
    '''The object magnitude in difference image.'''
    return oper['mag']

def a_image(img, cat, oper, sci, imgrn):
    '''The semi-major axis (in pixels) of object in difference image.'''
    return oper['ellipa']

def b_image(img, cat, oper, sci, imgrn):
    '''The semi-minor axis (in pixels) of object in difference image.'''
    return oper['ellipb']

def flags(img, cat, oper, sci, imgrn):
    '''Numerical representation of sextractor extraction flags.'''
    return oper['sexflags']

def mag_ref(img, cat, oper, sci, imgrn):
    '''Magnitude of nearest object from SNAUTOSCAN_COADD_OBJECTS if less
    than 5'' from candidate.'''
    return sci['mag_auto']

def mag_ref_err(img, cat, oper, sci, imgrn):
    '''Error on magnitude of nearest object from SNAUTOSCAN_COADD_OBJECTS
    if less than 5'' from candidate.'''
    return sci['magerr_auto']

def a_ref(img, cat, oper, sci, imgrn):
    '''Semi-major axis of the reference source.'''
    return sci['a_image']

def b_ref(img, cat, oper, sci, imgrn):
    '''Semi-minor axis of the reference source.'''
    return sci['b_image']

def bandnum(img, cat, oper, sci, imgrn):
    '''Image filter.'''
    return bm[oper['band']]

def n2sig3(img, cat, oper, sci, imgrn):
    '''The number of at least negative 2 sigma pixels in a 5x5 box
    centered on the candidate.'''
    return len(np.argwhere(imgrn['diff'][10: 15, 10: 15] <= -2))

def n3sig3(img, cat, oper, sci, imgrn):
    '''The number of at least negative 3 sigma pixels in a 5x5 box
    centered on the candidate.'''
    return len(np.argwhere(imgrn['diff'][10: 15, 10: 15] <= -3))

def n2sig5(img, cat, oper, sci, imgrn):
    ''' The number of at least negative 2 sigma pixels in a 7x7 box
    centered on the candidate.'''
    return len(np.argwhere(imgrn['diff'][9: 16, 9: 16] <= -2))

def n3sig5(img, cat, oper, sci, imgrn):
    ''' The number of at least negative 3 sigma pixels in a 7x7 box
    centered on the candidate.'''
    return len(np.argwhere(imgrn['diff'][9: 16, 9: 16] <= -3))

def n3sig5shift(img, cat, oper, sci, imgrn):
    ''' The difference between the number of at least positive 3 sigma
    pixels in a 7x7 box centered on the candidate position in the
    difference image and the number of at least positive 3 sigma
    pixels in a 7x7 box centered on the candidate position in the
    template image.'''
    return len(np.argwhere(imgrn['diff'][9 : 16, 9 : 16] >= 3)) - \
        len(np.argwhere(imgrn['temp'][9: 16, 9: 16] >= 3))

def n2sig5shift(img, cat, oper, sci, imgrn):
    ''' The difference between the number of at least positive 2 sigma
    pixels in a 7x7 box centered on the candidate position in the
    difference image and the number of at least positive 2 sigma
    pixels in a 7x7 box centered on the candidate position in the
    template image.'''
    return len(np.argwhere(imgrn['diff'][9 : 16, 9: 16] >= 2)) - \
        len(np.argwhere(imgrn['temp'][9: 16, 9: 16] >= 2))

def n2sig3shift(img, cat, oper, sci, imgrn):
    '''The difference between the number of at least positive 2 sigma
    pixels in a 5x5 box centered on the candidate position in the
    difference image and the number of at least positive 2 sigma
    pixels in a 5x5 box centered on the candidate position in the
    template image.'''
    return len(np.argwhere(imgrn['diff'][10 : 15, 10 : 15] >= 2)) - \
        len(np.argwhere(imgrn['temp'][10: 15, 10: 15] >= 2))

def n3sig3shift(img, cat, oper, sci, imgrn):
    '''The difference between the number of at least positive 3 sigma
    pixels in a 5x5 box centered on the candidate position in the
    difference image and the number of at least positive 3 sigma
    pixels in a 5x5 box centered on the candidate position in the
    template image.'''
    return len(np.argwhere(imgrn['diff'][10 : 15, 10 : 15] >= 3)) - \
        len(np.argwhere(imgrn['temp'][10: 15, 10: 15] >= 3))

def ellipticity(img, cat, oper, sci, imgrn):
    ''' The ellipticity of the candidate using a image and b image.'''
    return 1 - (oper['ellipb'] / oper['ellipa']) if oper['ellipa'] != 0 else 1

def nn_dist_renorm(img, cat, oper, sci, imgrn):
    ''' The distance in arcseconds from the candidate to the reference
    source.'''
    return sci['distance']

def maglim(img, cat, oper, sci, imgrn):
    '''True if there is no nearby reference source, false otherwise.'''
    return sci['maglim']

def min_distance_to_edge_in_new(img, cat, oper, sci, imgrn):
    '''Distance in pixels to the nearest edge of the array in the
    difference image.'''

    x = oper['pixelx']
    y = oper['pixely']

    return min(4096 - y, 2048 - x, x, y)

def ccdid(img, cat, oper, sci, imgrn):
    ''' Return the numerical ID of the specific camera detector.'''
    return oper['ccdnum']

def snr(img, cat, oper, sci, imgrn):
    ''' Signal to noise of candidate in difference image.'''
    return oper['flux'] / oper['flux_err']

def mag_from_limit(img, cat, oper, sci, imgrn):
    '''Limiting magnitude minus candidate magnitude.'''
    return oper['limmag'] - oper['mag']

def magdiff(img, cat, oper, sci, imgrn):
    '''When a reference source is found nearby, the difference between the
    candidate magnitude and the SNAUTOSCAN_COADD_OBJECTS source
    magnitude. else, the difference between the candidate magnitude
    and the limiting magnitude of the image.'''
    return oper['limmag'] - \
        oper['mag'] if sci['maglim'] else sci['mag_auto'] - oper['mag']

def spread_model(img, cat, oper, sci, imgrn):
    """Spread model."""
    return oper['spread_model']

def spreaderr_model(img, cat, oper, sci, imgrn):
    """Spread model error."""
    return oper['spreaderr_model']

def flux_ratio(img, cat, oper, sci, imgrn):
    '''Ratio of the aperture flux of the candidate minus the aperture
    flux of the reference source relative to the absolute value of the
    aperture flux of the reference source.'''

    return (cat['diff']['flux_aper'] - cat['aper']['flux_aper']) \
           / np.abs(cat['aper']['flux_aper'])

def lacosmic(img, cat, oper, sci, imgrn):
    '''Maximum pixel on LA Cosmic "fine structure" image.'''
    lhs = median_filter(img['diff'], size=3)
    rhs = median_filter(median_filter(img['diff'], size=3), size=7)
    fsimg = lhs - rhs
    fsmax = fsimg.max()
    diffmax = img['diff'].max()
    return diffmax / fsmax

def gaussscaleamp(img, cat, oper, sci, imgrn):
    '''Fit a spherical, 2D Gaussian to a 15x15 pixel cutout around the
    object in the difference image. Return a tuple containing the
    Chi**2, Sigma, and Amplitude of the fit.
    '''

    diffpix = img['diff']
    x, y = np.mgrid[:51, :51]

    def sph_gauss(xy, meanx, meany, amp, sigma):
        '''A spherical 2D gaussian with mean (meanx, meany), scale sigma, and
        amplitude amp.'''
        x, y = xy
        return amp * \
               np.exp(-0.5 * ((x - meanx) ** 2 + (y - meany) ** 2) / sigma ** 2)

    def gaussfit_chisq(param_vec):
        pred = sph_gauss((x, y), *param_vec)
        mask = np.ma.getmask(diffpix)
        resid = pred.ravel()[~mask.ravel()] - diffpix.compressed().ravel()
        return np.sum(
            resid ** 2) if len(resid) >= len(param_vec) else np.sum(pred ** 2)

    guess = np.array([25, 25, 10, 10])  # good guess

    popt, fmin, d = fmin_l_bfgs_b(gaussfit_chisq,
                                  guess, approx_grad=True,
                                  bounds=[(24.5, 25.5),
                                          (24.5, 25.5),
                                          (None, None),
                                          (None, None)])

    chisq = gaussfit_chisq(popt)

    return (chisq, popt[-1], popt[-2])


def colmeds(img, cat, oper, sci, imgrn):
    '''A bad column search. Compute the median pixel value of each
    column on a stamp, returning the maximum.'''

    return np.ma.median(img['diff'], axis=0).max()

def maskfrac(img, cat, oper, sci, imgrn):
    '''The fraction of the difference image stamp that is masked.'''
    pix = img['diff']
    num_pix_unmasked = pix.compressed().size
    num_pix = pix.size
    return 1 - (float(num_pix_unmasked) / num_pix)


def l1(img, cat, oper, sci, imgrn):
    '''L1 norm of difference image postage stamp.'''
    norm_const_arg = img['diff'].sum()
    numerator = np.abs(img['diff']).sum()
    if norm_const_arg < 0:
        numerator = -numerator
        norm_const_arg = -norm_const_arg
    return numerator / np.sqrt(norm_const_arg)

def pixelx(img, cat, oper, sci, imgrn):
    return oper['pixelx']

def pixely(img, cat, oper, sci, imgrn):
    return oper['pixely']


def sci(cid, oper, session, log):
    '''
    Retrieve information about template objects from
    SNAUTOSCAN_COADD_OBJECTS.
    Arguments:
    ----------
    cid: The SNOBJID of the candidate to be processed.
    
    session: A db cursor pointing to DESOPER or DESTEST.
    oper: A dict of information for this CID generated by
    querying SNOBS.
    Returns:
    --------
    A dict with the results of a candidate query.  eg. sci['mag_auto']
    returns the magnitude of the nearest reference object from the
    SNAUTOSCAN_COADD_OBJECTS table.
    '''

    ra_oper = oper['ra']
    dec_oper = oper['dec']
    filt = oper['band'].upper()

    ra_hi = float(ra_oper + (5 * u.arcsec).to(u.deg).value)
    ra_lo = float(ra_oper - (5 * u.arcsec).to(u.deg).value)
    dec_hi = float(dec_oper + (5 * u.arcsec).to(u.deg).value)
    dec_lo = float(dec_oper - (5 * u.arcsec).to(u.deg).value)

    sciquery = dedent('''select MAG_AUTO_{0}, MAGERR_AUTO_{0}, 
                         A_IMAGE, B_IMAGE, ELLIPTICITY_{0}, ra, 
                         dec FROM SNAUTOSCAN_COADD_OBJECTS WHERE 
                         ra BETWEEN :ralo and :rahi and dec BETWEEN :declo 
                         and :dechi'''.format(filt))

    names = [
        'mag_auto',
        'magerr_auto',
        'a_image',
        'b_image',
        'ellipticity',
        'distance']

    
    log.debug('Executing %s.', sciquery)

    try:
        session.execute(sciquery, rahi=ra_hi, ralo=ra_lo,
                        dechi=dec_hi, declo=dec_lo)
    except db_connect.cx_Oracle.DatabaseError as e:
        log.error('DATABASE INTERNAL ERROR EXECUTING DESSCI SELECT STATEMENT: %s' % e)
        log.error('DUMPING RELEVANT STATE')
        log.error('rahi: %f' % float(ra_hi))
        log.error('ralo: %f' % float(ra_lo))
        log.error('dechi: %f' % float(dec_hi))
        log.error('declo: %f' % float(dec_lo))
        log.error('filt: %s' % filt)
        raise e        
    
    # Get closest galaxy from coadd catalog. 
    coadd_gals = list()
    distance = list()
    for entry in session.fetchall():
        log.debug('Found source: %s', entry)
        first = SkyCoord(*entry[-2:], unit='deg')
        second = SkyCoord(ra_oper, dec_oper, unit='deg')
        separation = first.separation(second).arcsec
        if separation <= 5:
            distance.append(separation)
            coadd_gals.append(entry)
    coadd_gals = np.array(coadd_gals)
    distance = np.array(distance)
    indices = np.argsort(distance)
    result = coadd_gals[indices]

    ret = {}

    if len(result) == 0:
        ret['maglim'] = 1
        log.debug('%d has no reference source.', cid)
        append = dict(zip(names, [np.nan] * len(names)))
    else:
        ret['maglim'] = 0
        log.debug('%d has a reference source.', cid)
        result = result[0]
        dist = distance[indices][0]
        result = list(result[:-2])
        result.append(dist)
        append = dict(zip(names, result))
        if append['ellipticity'] == 0:
            append['ellipticity'] = 1 - \
                (append['b_image'] / append['a_image']) if append['a_image'] != 0 else 1

    ret.update(append)
    return ret

def snobs(cid, session, season, log, objlist=None):
    '''
    retrieve candidate metadata from DESOPER / SNOBS /IMAGE.
    
    arguments:
    ----------
    cid: the SNOBJID of the candidate to be processed
    session: a db cursor pointing to desoper
    returns:
    --------
    
    a dict eg. oper['mag'] = the magnitude of the candidate from the snobs table
    '''

    if objlist is not None:
        try:
            retrow = objlist[objlist['snobjid'] == cid][0]
        except IndexError as e:
            logging.error('WARNING: %s DOES NOT CONTAIN %s.' % (objlist.name, cid))
            raise e
        return retrow
    else:
        snobs = 'snobs_legacy' if season == 'Y1' else 'snobs'
        key = 'coaddid' if season == 'Y1' else 'image_name_diff'
        imtab = 'o' 
        extra = 'join image i on o.coaddid = i.id' if season == 'Y1' else ''

        query = dedent('''select o.mag, o.flux, o.flux_err, o.band,
                          o.ccdnum, o.ellipa, o.ellipb, o.sexflags, o.ra, o.dec, 
                          o.pixelx, o.pixely, %s.%s, o.galflux, o.spread_model, 
                          o.spreaderr_model from %s o %s where o.SNOBJID = :objid''' % (imtab, key, snobs, extra))

        names = ['mag', 'flux', 'flux_err', 'band', 'ccdnum', 'ellipa', 'ellipb', 
                 'sexflags', 'ra', 'dec', 'pixelx', 'pixely', key, 'galflux', 'spread_model',
                 'spreaderr_model']


        log.debug('Executing %s.', query)
        session.execute(query, objid=cid)
        query_result = session.fetchone()
        if query_result is None:
            logging.error('WARNING: %s DOES NOT CONTAIN %s.' % (snobs, cid))
            raise Exception
        log.debug('Success.')
        ret = dict(zip(names, query_result))
        lim_mag_query = 'SELECT mag FROM %s WHERE %s  = :val' % (snobs, key)
        log.debug('Executing %s.', lim_mag_query)
        session.execute(lim_mag_query, val=ret[key])
        log.debug('Success.')
        ret['limmag'] = max([mag for mag, in session.fetchall()])
        return ret

def compress(pixels):
    """Compress 51x51 stamp to 25x25 stamp, ignoring last row and column."""
    compressed_pixels = np.array(
        [
            [0.25 * (pixels[2 * i, 2 * j] +
                     pixels[2 * i, 2 * j + 1] +
                     pixels[2 * i + 1, 2 * j] +
                     pixels[2 * i + 1, 2 * j + 1]
                     )
             for j in xrange(25)]
            for i in xrange(25)]
    )
    return compressed_pixels

def read_filterObj_file(objlist):

    '''Parse a filterObj output file, returning its tabular
    information as a numpy array.'''

    diffim_info = utils.keyed_file_to_dict(objlist)

    # Make sure the file is rewound
    objlist.seek(0, 0)

    skip_head = 0
    skip_delta = 0

    for i, line in enumerate(objlist):
        if line.startswith('VARNAMES_OBJ'):
            skip_head = i
        if line.startswith('NOBJ_TOTAL'):
            skip_delta = i

    skip_foot = i - skip_delta

    meta = np.atleast_1d(np.genfromtxt(
        objlist.name,
        dtype=None,
        names=True,
        skip_header=skip_head,
        skip_footer=skip_foot,
        case_sensitive='lower'
    ))

    # add limmag to rows

    # this just creates an array the same length as meta where every
    # entry has the image's limiting magnitude

    limmag_arr = np.zeros(len(meta)) + meta[meta['reject'] == 0]['mag'].max()

    # we only consider the unrejected entries to be consistent with
    # the database calculation of the limiting magnitude

    # now append it to the meta array as a column

    meta = recfunctions.append_fields(
        meta,
        'limmag',
        limmag_arr,
        '<f8',
        usemask=False)

    dtypenameslist = list(meta.dtype.names)

    ellipaind = dtypenameslist.index('a_image')
    ellipbind = dtypenameslist.index('b_image')
    flagsind = dtypenameslist.index('flags')
    sprmodind = dtypenameslist.index('sprmod')
    sprmoderrind = dtypenameslist.index('sprmoderr')
    ximgind = dtypenameslist.index('x_image')
    yimgind = dtypenameslist.index('y_image')

    dtypenameslist[ellipaind] = 'ellipa'
    dtypenameslist[ellipbind] = 'ellipb'
    dtypenameslist[flagsind] = 'sexflags'
    dtypenameslist[sprmodind] = 'spread_model'
    dtypenameslist[sprmoderrind] = 'spreaderr_model'
    dtypenameslist[ximgind] = 'pixelx'
    dtypenameslist[yimgind] = 'pixely'

    meta.dtype.names = tuple(dtypenameslist)

    # do the same thing for bands

    band_arr = np.array([diffim_info['BAND'].lower()
                         for i in range(len(limmag_arr))])

    meta = recfunctions.append_fields(
        meta,
        'band',
        band_arr,
        '|S1',
        usemask=False)

    # rewind file
    objlist.seek(0, 0)
    return meta


def renormalize(pixels, compression=True):

    ###################################################################
    # NMAD renormaliztion procedure.
    # Expresses each pixel as [number of absolute deviations from
    # the background median (calculated not including null pixels),
    # divided by the median absolute deviation of all nonnull pixels from
    # the background median]. All null pixels are recast as np.nan.
    ###################################################################

    if compression:
        pixels = compress(pixels)

    # 1d np.array of nonnull pixels used to calculate background median

    mp = np.ma.masked_where(np.absolute(pixels) < 1e-29, pixels)
    np.ma.set_fill_value(mp, np.nan)

    # stat_pixels = np.array([ pixel for pixel in pixels.flat if np.absolute(pixel) > 1e-29 ])
    # background_med = np.median(stat_pixels)

    bkgnd_med = np.ma.median(mp)
    phi = 1.4826  # NMAD factor
    # sigma = phi * np.median(np.absolute(stat_pixels - background_med))

    sigma = phi * np.ma.median(np.absolute(mp - bkgnd_med))

    # Renormalize (return all pixels)

    # return np.array([[(pixel - background_med) / sigma if np.absolute(pixel)
    # > 1e-29 else np.nan for pixel in row] for row in pixels])

    return (mp - bkgnd_med) / sigma


def get_closest(cat, tol=3, x=26, y=26):
    '''Given an astropy asciitable representation of a SExtractor
    output catalog, return the row that corresponds to the detection
    closest to a given pixel on a stamp. If no detection is found
    within a specified pixel tolerance, return None.

    Parameters
    ----------
    cat: numpy structured array. 
        The array representation of the SExtractor output catalog. 
        
    tol: float, int. 
        The maximum allowable distance from the center pixel of the
        stamp in pixels that a detection can have.
        
    x: int, default=26.  
        The x-coordinate of the pixel with respect to
        which to measure distances.
        
    y: int, default=26.  
        The y-coordinate of the pixel with respect to
        which to measure distances.
        
    Returns
    -------
    The row from `cat` that corresponds to the detection closest to the
    center of the stamp.
    '''

    dist = Column(name='dist',
                  data=np.sqrt((cat['x_image'] - x) ** 2
                               + (cat['y_image'] - y) ** 2))
    cat.add_column(dist)
    cat.sort('dist')
    try:
        cand = cat[0]
    except IndexError:
        return None
    shortest_dist = cand['dist']
    if shortest_dist <= tol:
        return cand
    else:
        return None


def renorm(stamp):
    '''Renormalize a postage stamp.'''
    mp = np.ma.masked_where(np.absolute(stamp) == 1e-30, stamp)
    np.ma.set_fill_value(mp, np.nan)
    bkgnd_med = np.ma.median(mp)
    return (mp - bkgnd_med) / np.abs(mp).max()


def setup_logger(logger_name, log_file, level):
    l = logging.getLogger(logger_name)
    l.setLevel(level)
    formatter = logging.Formatter('LOG {0} : %(levelname)s : %(asctime)s : %(message)s'.format(logger_name),
                                  datefmt='%m/%d/%Y %I:%M:%S %p')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setLevel(level)
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setLevel(logging.ERROR)
    streamHandler.setFormatter(formatter)
    l.propagate = False
    l.addHandler(fileHandler)
    l.addHandler(streamHandler)

def _extract_subseq(seq, i, level, objlist, des_services,
                    des_db_section, readlegacyDB, **kwargs):
    """Extract Features for a single detection."""

    # Initialize logger.
    setup_logger('%d' % i, '%d.log' % i, level)
    log = logging.getLogger('%d' % i)

    features = dict()
    cursor = db_connect.db_connect(des_services, des_db_section)
    season = 'Y1' if readlegacyDB else 'NotY1'
    
    for j, (snobjid, srch, temp, diff) in enumerate(seq):
 
        # Initial type enforcement.
        cid = int(snobjid)

        info_dict = dict()
        features[cid] = info_dict
        info_dict['snobjid'] = cid
        info_dict['ml_version'] = __ml_version__

        # Begin extraction.
        log.info('**Processing SNOBJID=%d.**', cid)

        # Get renormalized, uncompressed brink pixels.
        log.debug('Reading brink pixels for %d.', cid)
        try:
            spix, tpix, dpix = io.pix(srch, temp, diff, log=log)
        except IOError as e:
            log.warning("Failed to read brink pixels for %d.", cid, exc_info=True)
            log.warning("Adding %d to error list.", cid)
            info_dict['error_msg'] = 'brink pixel read failure'
            continue
        log.debug('Read brink pixels successfully.')

        spixr, tpixr, dpixr = map(renorm, (spix, tpix, dpix))

        # Label pixels.
        img = {'diff': dpixr,
               'ref': tpixr,
               'srch': spixr}
            
        # Get renormalized, compressed goldstein pixels.
        log.debug('Computing goldstein pixels for %d.', cid)
        try:
            imgrn = dict(zip(('srch', 'temp', 'diff'), map(renormalize, (spix, tpix, dpix))))
        except IOError as e:
            log.warning("Failed to compute goldstein pixels for %d.", cid, exc_info=True)
            log.warning("Adding %d to error list.", cid)
            info_dict['error_msg'] = 'goldstein pixel computation failure'
            continue

        log.debug('Computed goldstein pixels successfully.')

        # Do source extraction and photometry.
        cat = dict()
        log.debug('Performing source extraction / photometry on %s.', diff)

        try:
            dflux, dfluxerr, dflag = sep.apercirc(dpix, 25., 25., 5.)
        except Exception as e:
            log.warning('Failed to sep diffim on %d.', cid, exc_info=True)
            info_dict['error_msg'] = 'diffim sep failure'
            continue

        log.debug('SExtraction successful for %s.', diff)
        cat['diff'] = {'flux_aper': dflux}

        log.debug('Performing aperture photometry on %s', temp)
        try:
            tflux, tfluxerr, tflag = sep.apercirc(tpix, 25., 25., 5.)
        except Exception as e:
            log.warning('Failed to sep aper for %s.', cid, exc_info=True)
            info_dict['error_msg'] = 'aper sep failure'
            continue
            
        cat['aper'] = {'flux_aper':tflux}
        
        log.debug('Extracting information from SNOBS for %d.', cid)
        try:
            oper = snobs(cid, cursor, season, log, objlist)
        except:
            log.warning('Failed to extract SNOBS info for %d.', cid, exc_info=True)
            info_dict['error_msg'] = 'SNOBS extraction failure'
            continue
        
        log.debug('Extracint information from SNACOADDOBJECTS for %d.', cid)
        try:
            scidata = sci(cid, oper, cursor, log)
        except:
            log.warning('Failed to extract SNACOADDOBJECTS info for %d.', cid, exc_info=True)
            info_dict['error_msg'] = 'SNACOADD extraction failure'
            continue

        # Compute features now.

        log.debug('Computing features for %d.', cid)
        log.debug('Computing gaussscaleamp.')

        # feature args:
        
        args = (img, cat, oper, scidata, imgrn)

        try:
            gauss, scale, amp = gaussscaleamp(*args)
        except Exception as e:
            log.warning('Failed to gaussscaleamp %d.', cid, exc_info=True)
            info_dict['error_msg'] = 'gausscaleamp failed'
            continue
            
        det_features = list()
        for name in FEATS:
            log.debug('Computing %s.', name)
            if name not in ['gauss', 'scale', 'amp']:
                try:
                    det_features.append(eval(name)(*args))
                except Exception as e:
                    log.warning('Failed to compute %s for %d.', name, cid, exc_info=True)
                    info_dict['error_msg'] = 'could not compute %s' % name
                    continue
            else:
                det_features.append(eval(name))
        log.info('Success.')
        info_dict.update(dict(zip(FEATS, det_features)))
    return features


def extract_features(data_set, debug=False, n_jobs=1, objlist=None,
                     des_db_section=None, des_services=None, readlegacyDB=False, **kwargs):
    """Extracts autoScan features from a data set.  First, runs source
    extractor to perform object detection and photometry on difference
    image postage stamps. Then runs source extractor to perform
    reference image object detection and photometry on template image
    postage stamps. Then runs source extractor in dual image mode to
    perform aperture photometry on template postage stamp.

    Parameters:
    -----------

    data_set: a numpy structured array or astropy table that has the
    following one integer and three string columns (not necessarily in
    this order):

        ( 'id', 'srch', 'temp', 'diff' )

    where id is the SNOBJID of the object, and 'srch', 'temp', and
    'diff' are the names of its stamps.

    Equivalently, an autoscan.MLTrainingSet.


    Returns:
    --------

    feats_noref: array_like, shape (n_samples, n_features)
        Features of extracted objects for which no reference source was detected.

    feats_ref: array_like, shape (n_samples, n_features)
        Features of extracted objects for which a reference source was detected.
    """

    # TODO: Document

    if len(data_set) == 0:
        raise utils.NoDataException, 'Extract_features called on an empty data set.' \
                                     'Check that STAMPS.LIST is not empty.'

    # Build stamp path array.  This object is a spreadsheet that
    # points to the location of postage stamps at NCSA.
    stamp_array = data_set[['id', 'srch', 'temp', 'diff']]


    # Set loglevel.
    level = logging.DEBUG if debug else logging.INFO

    # Extract features in parallel.
    features = np.array(Parallel(n_jobs=n_jobs)(
        delayed(_extract_subseq)(seq, i, level, objlist, des_services, des_db_section, readlegacyDB)
        for i, seq in enumerate(utils.split_seq(stamp_array, n_jobs))))


    # Conslidate logs.
    try: 
        baselogger_file = open(logging.getLogger('').handlers[0].baseFilename, 'a')
    except AttributeError:
        baselogger_file = sys.stdout
    for i in range(n_jobs):
        with open('%d.log' % i, 'r') as l:
            name = l.name
            baselogger_file.write(l.read())
        os.remove(name)

    # Consolidate features.
    pooled = dict()
    for di in features:
        pooled.update(di)
    return pooled
