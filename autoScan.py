#!/usr/bin/env python

__author__ = 'Danny Goldstein <dgold@berkeley.edu>'
__ml_version__ = 3

prog_name = 'autoScan'
description = '''autoScan: DESSN's automated supernova candidate scanner.
takes in 51x51 pixel cutouts of objects produced by makeStamps and scores
them from 0 to 1. objects with higher scores are more likely to be SNe.
uses the ML3 algorithm.'''
prog = './autoScan.py'

# Line for automated diffim manual generation
# Define OracleTableName_SNAUTOSCAN // write object scores, ml version.

from argparse import FileType, ArgumentParser
import sys

def autoscan_arg_parser():
    '''Create the autoScan CLI argument parser.'''

    parser = ArgumentParser(description=description, prog=prog)

    # Collect input file arguments into a group.
    pred_file_info = '''Please do not use any regular expressions or
    shell environment variables in the filenames you pass to these
    arguments. They will not be parsed.'''
    pred_files = parser.add_argument_group('Mandatory Inputs', pred_file_info)

    # Collect database arguments into a group.
    db_args = parser.add_argument_group('Database Options')

    # Collect miscellaneous options into a group.
    misc_opts = parser.add_argument_group('Miscellaneous Options')

    # Input file arguments.
    pred_files.add_argument(
        '-inDir_stamps',
        metavar='STAMPDIR',
        dest='stamppath',
        required=True,
        help='Folder containing stamps.')
    pred_files.add_argument(
        '-inFile_stampList',
        metavar='STAMPLIST',
        dest='stamplist',
        required=True,
        help='Relative path to stamp list file in the directory tree'
             ' rooted at STAMPDIR.',
        type=str)
    pred_files.add_argument(
        '-outFile_results',
        dest='results',
        required=True,
        help='File in which to write classification results for processed objects.',
        type=FileType('w'))
    pred_files.add_argument(
        '-inFile_scaler',
        dest='scaler',
        required=True,
        help='Scaler binary.',
        type=FileType('rb'))
    pred_files.add_argument(
        '-inFile_imputer',
        dest='imputer',
        required=True,
        help='Imputer binary.',
        type=FileType('rb'))
    pred_files.add_argument(
        '-inFile_model',
        dest='model',
        required=True,
        help='Classifier binary.',
        type=str)

    # Database arguments.
    db_args.add_argument(
        '-writeDB',
        dest='writedb',
        help='If flagged, write results to database.',
        action='store_true')
    db_args.add_argument(
        '-inFile_desservices',
        dest='des_services',
        help='Use the specified file instead of $DES_SERVICES.',
        default=None,
        type=str)
    db_args.add_argument(
        "-des_db_section", 
        required=False, 
        dest="des_db_section",
        default=None,
        help="Section of .desservices file with connection info (db-desoper, db-destest)." \
             "If not specified, will use $DES_DB_SECTION.",
        type=str)
    db_args.add_argument(
        "-readlegacyDB",
        required=False,
        action='store_true',
        default=False,
        dest='readlegacyDB',
        help='If true, read object data from SNOBS_LEGACY instead of SNOBS.')
    db_args.add_argument(
        "-writelegacyDB",
        required=False,
        action='store_true',
        default=False,
        dest='writelegacyDB',
        help='If true, only write v2 columns to SNAUTOSCAN.')
    
    # Miscellaneous options.
    
    misc_opts.add_argument(
        '-inFile_objList',
        metavar='OBJLIST',
        dest='objlist',
        required=False,
        default=None,
        help='filterObj file for detections in STAMPLIST. Specify to speed up program execution.',
        type=FileType('r'))
    misc_opts.add_argument(
        '-outFile_stdout',
        metavar='logfile',
        dest='log',
        help='The logfile.',
        default=sys.stdout,
        type=str)
    misc_opts.add_argument(
        '-debug',
        dest='debug',
        required=False,
        help='Run classifier in debug (verbose) mode. ' \
             'Log useful error messages and program state to logfile.',
        action='store_true')
    misc_opts.add_argument(
        '-n_jobs',
        dest='n_jobs',
        required=False,
        help='Number of processes to use for feature extraction.',
        type=int,
        default=1)
    return parser

# Create argparser.
parser = autoscan_arg_parser()
ins = parser.parse_args()

def produce_failure(idlist):
    '''Write a results file for a complete program failure. This
    method is called when no detections are successfully processed.'''

    logging.error('Writing failure to %s', ins.results)
    
    snfake_ids = list()
    cursor = db_connect.db_connect(ins.des_services, ins.des_db_section)
    snobs_table = 'SNOBS_LEGACY' if ins.readlegacyDB else 'SNOBS'
    fake_id_query = 'SELECT snfake_id FROM %s WHERE snobjid = :id' % snobs_table

    for ob in idlist:
        cursor.execute(fake_id_query, id=int(ob))
        try:
            result = cursor.fetchone()[0]
            wrong_db = False
        except TypeError:
            logging.error('**Error writing failure!**')
            logging.error('%d is not present in %s!', ob, 
                          'SNOBS' if not ins.readlegacyDB else 'SNOBS_LEGACY')
            logging.error('This indicates that you may be connected to the wrong database.')
            logging.error('**MAKE SURE YOUR DES_DB_SECTION IS CONFIGURED PROPERLY!**')
            wrong_db = True
            break
        snfake_ids.append(result)

    ins.results.write('ML_VERSION: %d\n' % __ml_version__)

    if not wrong_db:
        ins.results.write('NVAR_OBJ: 3\n')
        ins.results.write('VARNAMES_OBJ:\tSNOBJID\t\tSCORE\tSNFAKE_ID\n')
        for objid, snfid in zip(idlist, snfake_ids):
            ins.results.write('OBJ:{3}{0}\t{1}\t{2}\n'.format(objid, -9., snfid, ' ' * 12))
    else:
        ins.results.write('NVAR_OBJ: 2\n')
        ins.results.write('VARNAMES_OBJ:\tSNOBJID\t\tSCORE\n')
        for objid in idlist:
            ins.results.write('OBJ:{2}{0}\t{1}\n'.format(objid, -9., ' ' * 12))
    exit(1)

def publish_results(data_dict):
    """Write results to results file."""

    snfake_ids = list()
    fluxes = list()
    logging.debug('Connecting to db...')
    cursor = db_connect.db_connect(ins.des_services, ins.des_db_section)
    logging.debug('Success.')
    snobs_table = 'SNOBS_LEGACY' if ins.readlegacyDB else 'SNOBS'
    query = 'SELECT snfake_id, flux FROM %s WHERE snobjid = :id' % snobs_table
    snobjids = data_dict.keys()

    for ob in data_dict:
        logging.debug('Executing query')
        cursor.execute(query, id=int(ob))
        logging.debug('Executed %s ' % query)
        result = cursor.fetchone()
        if result is None:
            try:
                result = [ins.objlist[ins.objlist['snobjid'] == ob][key][0] for key in ('snfake_id', 'flux')]
            except KeyError:
                produce_failure(snobjids)
        snfake_ids.append(result[0])
        fluxes.append(result[1])

    ins.results.write('ML_VERSION: {0}\n'.format(__ml_version__))
    ins.results.write('NVAR_OBJ: 4\n')
    colnames = 'VARNAMES_OBJ:\tSNOBJID         SCORE\tFLUX\t        SNFAKE_ID\n'
    ins.results.write(colnames)
    result_string = 'OBJ:{4}{0}\t{1}\t{2}\t{3}\n'
    for objid, flux, snfake_id in zip(snobjids, fluxes, snfake_ids):
        if 'ml_score' in data_dict[objid].keys():
            score = data_dict[objid]['ml_score']
        else:
            score = -9.
        if flux is None:
            flux = -9.
        outtup = (objid, round(score, 3), flux, snfake_id,' ' * 12)
        ins.results.write(result_string.format(*outtup))

# update SNautoscan
def write_db(data_dict):
    cursor = db_connect.db_connect(ins.des_services, ins.des_db_section)
    taskid = os.getenv("DESDMFW_TASKID", None)
    cursor.callproc("setDesCtx", [taskid])
    insert = 'INSERT INTO SNAUTOSCAN (%s) VALUES (%s)'
    update = 'UPDATE SNAUTOSCAN SET %s WHERE SNOBJID=:snobjid AND ML_VERSION=:ml_version' 

    for objid in data_dict:
        names = data_dict[objid].keys()
        if ins.writelegacyDB:
            names = [name for name in names if name in FEATS_LEGACY + ['snobjid', 'ml_version', 'ml_score']]
        values = [':' + name for name in names]
        this_insert = insert % (', '.join(names), ', '.join(values))

        equalities = ', '.join(['='.join(elem) for elem in zip(names, values)])
        this_update = update % equalities
        
        safestr = lambda x: str(x) if not np.isnan(x) else np.nan
        strdict = {key: safestr(data_dict[objid][key]) for key in data_dict[objid] if key in names}
        try:
            logging.info('Attempting insert for %s.', objid)
            logging.debug('Query is %s' % this_insert)
            logging.debug('Strdict is %s' % strdict)
            cursor.execute(this_insert, **strdict)
            logging.info('Success.')
        except cx_Oracle.IntegrityError as e:
            logging.info('%s is already present in the DB.', objid)
            logging.info('Attempting update for %s.', objid)
            cursor.execute(this_update, **strdict)
            logging.info('Success.')
            logging.debug('%d rows affected.', cursor.rowcount)

    cursor.connection.commit()
    cursor.close()
    cursor.connection.close()

def classify(X, clf, imputer, scaler):
    """Classify an autoScan feature array with a trained model. Return the
    IDs and scores of the classified detections.

    Parameters
    ----------
    X: array_like
        Record array with named fields (one of which must be 'SNOBJID')
        and all features to be used for classification.

    clf: sklearn.RandomForestClassifier
        The model to use for classification.
    """

    if len(X) == 0:
        logging.error('Empty X passed to classify.')
        raise NoDataException

    X_feats_unscaled = X[FEATS]
    X_snobs = X['snobjid']

    # Imputer, scaler.
    logging.info('Imputing and scaling features...')
    x = scaler.transform(imputer.transform(np.asarray(X_feats_unscaled.tolist())))

    # Classify.
    logging.info('Scoring.')
    clf.n_jobs = 1
    probs = clf.predict_proba(x)
    rb_scores = probs[:, 1]

    return X_snobs, rb_scores

# Main routine.

if __name__ == '__main__':

    import os
    import numpy as np
    from autoscan import *
    import pickle
    import joblib
    import logging
    from time import time
    import cx_Oracle

    start = time()

    # Setup logging.
    log_kw = dict()
    if ins.log != sys.stdout:
        log_kw = {'filename': ins.log, 
                  'filemode': 'w'}
        log_kw['level'] = logging.INFO
    else:
        log_kw['level'] = logging.INFO
    if ins.debug:
        log_kw['level'] = logging.DEBUG

    logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', **log_kw)

    # Summarize inputs.
    logging.info('Received inputs: %s.', ins)

    # Read list of stamps.
    stamps_fullpath = os.path.join(ins.stamppath, ins.stamplist)
    logging.info('Acquiring stamps from %s.', stamps_fullpath)
    ps = lambda stamppath: os.path.join(ins.stamppath, stamppath)
    stamp_array = np.atleast_1d(np.genfromtxt(stamps_fullpath,
                                              dtype=None,
                                              usecols=[1, 2, 3, 4],
                                              names=(
                                                  'id',
                                                  'srch',
                                                  'temp',
                                                  'diff'),
                                              converters={
                                                  'srch':ps,
                                                  'temp':ps,
                                                  'diff':ps,
                                              },
                                              autostrip=True))

    if len(stamp_array) == 0:
        logging.error('Input file %s is empty.', ins.stamplist)
        produce_failure(stamp_array['id'])

    # Load classifiers.
    logging.info('Loading classifier.')
    try:
        clf = joblib.load(ins.model)
    except Exception as e:
        logging.error('', exc_info=True)
        raise e
    logging.debug('Loaded %s.', ins.model)    

    # Load binaries.
    imputer = pickle.load(ins.imputer)
    scaler = pickle.load(ins.scaler)

    # Load objlist if specified.
    if ins.objlist is not None:
        ins.objlist = features.read_filterObj_file(ins.objlist) 
    
    # Extract features.
    logging.info('Beginning core feature extraction loop.')
    try:
        data = features.extract_features(stamp_array, **vars(ins))
    except Exception as e:
        logging.error('', exc_info=True)
        logging.error('Wrote failure to %s.', ins.log)
        produce_failure(stamp_array['id'])
        raise e

    # Success?
    if len(data) == 0:
        produce_failure(stamp_array['id'])

    # Munge output of `extract_features`.
    names = ['snobjid'] + FEATS
    formats = ['<i8'] + ['<f8'] * len(FEATS)
    X_dtype = np.dtype(zip(names, formats))

    X = np.array([tuple([data[key][name] for name in names])
                  for key in data if 'error_msg' not in  
                  data[key].keys()], dtype=X_dtype)

    # Classify objects.
    try:
        snobjids, scores = classify(X, clf, imputer, scaler)
    except NoDataException:
        produce_failure(stamp_array['id'])

    for objid, score in zip(snobjids, scores):
        data[objid]['ml_score'] = score

    # Write results to file.
    publish_results(data)
    
    # And to database.
    if ins.writedb:
        write_db(data)

    end = time()
    logging.info('Program terminated successfully.')
    logging.info('Execution time: %.2f seconds.', end - start)
