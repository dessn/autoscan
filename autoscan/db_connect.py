__author__ = 'Danny Goldstein <dgold@berkeley.edu>'
__all__ = ['db_connect']

'''DES database services for autoscan.'''

import logging
import os
import base64
import cx_Oracle
import coreutils

from .utils import DBException

def db_connect(des_services=None, des_db_section=None): 
    """Connect to the DES database using coreutils and cx_Oracle. Return
    an active cursor to the database.
     
     Parameters
     ----------
     des_services: str,
         The name of the DES_SERVICES file to use, which contains
         information about specific DES database services to connect
         to. If None, use os.getenv('DES_SERVICES').
    
     des_db_section: str,
         The section of the DES_SERVICES file to read for DB
         connection information.  If none, use
         os.getenv('DES_DB_SECTION').
    """

    if des_services is None:
        des_services = os.getenv('DES_SERVICES')
            
    if des_db_section is None:
        des_db_section = os.getenv('DES_DB_SECTION')

    info_dict = coreutils.serviceaccess.parse(des_services, des_db_section)

    user = info_dict['user'] 
    password = info_dict['passwd'] 
    db_name = info_dict['name']
    server  = info_dict['server']
        
    host = '/'.join([server, db_name])
    db = cx_Oracle.connect(user, password, host, threaded=True)
    cursor = db.cursor()

    logging.info('%s is connected to %s', user, db_name)

    return cursor


def rows_to_dict_list(result, cursor, lower=False):
    columns = [i[0] for i in cursor.description] if not lower else [i[0].lower() for i in cursor.description]
    return [dict(zip(columns, row)) for row in result]

