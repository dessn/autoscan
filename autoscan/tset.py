__author__ = 'Danny Goldstein <dgold@berkeley.edu>'

import pickle
import numpy as np
from pkg_resources import resource_string

from . import db_connect


class MLTrainingSet:

    '''
    object that assembles and packages
    a training set.
    '''

    def __init__(self, size, rbfrac, reprocessed=False, n_jobs=60):
        '''
size: int, size of training set
        rbfrac: float, 0 - 1, fraction of
        training set composed of reals.
        '''

        if rbfrac >= 1 or rbfrac <= 0:
            raise ValueError('rbfrac must be less than 1 and greater than 0')

        if size <= 0 or not isinstance(size, type(1)):
            raise ValueError('size must be a positive integer')

        self.rbfrac = rbfrac
        self.size = size

        self.n_jobs = n_jobs

        self.reprocessed = reprocessed

        self.cursor = db_connect.y1legacy()
        self.reals = self.make_reals()
        self.bogus = self.make_bogus()
        self.set = np.concatenate((self.reals, self.bogus))

    def __getitem__(self, key):
        return self.set[key]

    def __repr__(self):
        return self.set.__repr__()

    def __str__(self):
        return str(self.set)

    def __len__(self):
        return len(self.set)

    def make_reals(self):
        '''
        generate the real detections
        to populate the
        training set.

        currently this
        is set to just pick out rick's
        fakes.
        '''

        query = resource_string('autoscan', 'database/real_query.sql')
        self.cursor.execute(query)
        query_results = self.cursor.fetchmany(
            self.size - int(self.size * self.rbfrac))
        dtype = [
            ('SNOBJID',
             '<i8'),
            ('OBJECT_TYPE',
             '<i8'),
            ('DIFF',
             'S88'),
            ('SRCH',
             'S88'),
            ('TEMP',
             'S88'),
            ('NITE',
             '<i8'),
            ('MAG',
             '<f8'),
            ('CCDID',
             '<i8'),
            ('FIELD',
             'S6'),
            ('BAND',
             'S6')]
        real_set = np.array(query_results, dtype=dtype)
        return real_set

    def make_bogus(self):
        '''
        generate the bogus detections
        with which to populate the
        training set.
        '''

        query = resource_string('autoscan', 'database/bogus_query.sql')
        self.cursor.execute(query)
        query_results = self.cursor.fetchmany(
            self.size - int(self.size * self.rbfrac))
        dtype = [
            ('SNOBJID',
             '<i8'),
            ('OBJECT_TYPE',
             '<i8'),
            ('DIFF',
             'S88'),
            ('SRCH',
             'S88'),
            ('TEMP',
             'S88'),
            ('NITE',
             '<i8'),
            ('MAG',
             '<f8'),
            ('CCDID',
             '<i8'),
            ('FIELD',
             'S6'),
            ('BAND',
             'S6')]
        bogus_set = np.array(query_results, dtype=dtype)

        return bogus_set

    def save(self, filename):
        """pickle training set to file"""

        pickle.dump(self.set, open(filename, 'wb'))
