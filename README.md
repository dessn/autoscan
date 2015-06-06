# autoscan 

*Python package for artifact rejection in Dark Energy
 Survey (DES) supernova difference imaging.*
 
 Currently *only* usable in the DES difference imaging pipeline. Code is provided as a reference implementation with **no guarantee of usability for other surveys** at this time.

## Contents

This repository contains:
    
1. **Core classes and methods for**
        
  * *Computing classification features using DES difference image
  object postage stamps, the DES database @ NCSA, and,
  optionally `filterObj` output files.*

  * *Reading from, writing to, and connecting to the DES database.*
      
  * *Scoring and classifying difference image objects.*
	    
  * *Publishing classification results.*
    
2. **A readymade script** for the production DES diffim pipeline that
reads information about objects from stamps, the database, and
(optionally, for speed) files produced by previous pipeine steps,
computes classification features for each object, scores each
object using a trained classifier, writes the results to an
SNANA-formatted file, and, optionally, writes the results to the
`SNAUTOSCAN` table in the database.

This repository does *not* contain:
     
1. Trained classifier binaries. (For these, email dgold@berkeley.edu).
2. A script to retrain on the fly.

## Installation

Clone this repository, then do:

    cd autoscan
    python setup.py install

to build the package in your python interpreter's `site-packages`
folder, which is in your interperter's module search path by default.
To build the package in another location, do:

    python setup.py install --prefix=build/dir/prefix

then modify your `$PYTHONPATH` environment variable to include
`build/dir/prefix`.

## Dependencies

`autoscan` requires the following python packages:

* [sep](https://github.com/kbarbary/sep) `>= 0.1.0`
* [astropy](http://www.astropy.org/) `>= 0.4.2`
* [numpy](http://www.numpy.org/) `>=1.9.0`
* [cx_Oracle](http://cx-oracle.sourceforge.net/) `>=5.1.3`
* [coreutils](https://dessvn.cosmology.illinois.edu/websvn/desdm/devel/CoreUtils/) 
* [scikit-learn](http://scikit-learn.org/stable/) `==0.14`
* [scipy](http://www.scipy.org) `>=0.13.0`
* [joblib](https://pythonhosted.org/joblib) `>=0.8.3-r1`

## Using autoscan

This package exposes two pathways for using `autoscan`. 

* `autoScan.py` is a readymade script for scoring detections
of variability produced by the DES `diffim` pipeline. Invoke it
from the command line. 

````
usage: ./autoScan.py [-h] -inDir_stamps STAMPDIR -inFile_stampList
                         STAMPLIST -outFile_results RESULTS -inFile_scaler
                         SCALER -inFile_imputer IMPUTER -inFile_model MODEL
                         [-writeDB] [-inFile_desservices DES_SERVICES]
                         [-des_db_section DES_DB_SECTION] [-readlegacyDB]
                         [-inFile_objList OBJLIST] [-outFile_logFile logfile]
                         [-debug] [-n_jobs N_JOBS]

autoScan: DESSN's automated supernova candidate scanner. takes in 51x51 pixel
cutouts of objects produced by makeStamps and scores them from 0 to 1. objects
with higher scores are more likely to be SN. uses the ML3 algorithm.

optional arguments:
  -h, --help            show this help message and exit

Mandatory Inputs:
  Please do not use any regular expressions or shell environment variables
  in the filenames you pass to these arguments. They will not be parsed.

  -inDir_stamps STAMPDIR
                        Folder containing stamps.
  -inFile_stampList STAMPLIST
                        Relative path to stamp list file in the directory tree
                        rooted at STAMPDIR.
  -outFile_results RESULTS
                        File in which to write classification results for
                        processed objects.
  -inFile_scaler SCALER
                        Scaler binary.
  -inFile_imputer IMPUTER
                        Imputer binary.
  -inFile_model MODEL   Classifier binary.

Database Options:
  -writeDB              If flagged, write results to database.
  -inFile_desservices DES_SERVICES
                        Use the specified file instead of $DES_SERVICES.
  -des_db_section DES_DB_SECTION
                        Section of .desservices file with connection info (db-
                        desoper, db-destest).If not specified, will use
                        $DES_DB_SECTION.
  -readlegacyDB         If true, read object data from SNOBS_LEGACY instead of
                        SNOBS.

Miscellaneous Options:
  -inFile_objList OBJLIST
                        filterObj file for detections in STAMPLIST. Specify to
                        speed up program execution.
  -outFile_logFile logfile
                        The logfile.
  -debug                Run classifier in debug (verbose) mode. Log useful
                        error messages and program state to logfile.
  -n_jobs N_JOBS        Number of processes to use for feature extraction.
````

* If you are writing your own python code and want it to call
  `autoscan` methods, you can import autoscan into your script and
  call its methods directly.

## How does it work? 

See [this paper](http://arxiv.org/abs/1504.02936) 
for a description of `autoscan` and its role in the 
DES supernova pipeline. 

## FAQ

* Q: **Can I use this for surveys other than DES?**
* A: Not yet, but stay tuned. 

