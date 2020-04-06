#!/usr/bin/env python
'''
Test suite for the Validate package.
'''

import os
import numpy as np
import datetime as dt
import unittest
import validator as vd

# All test classes in this file:
__all__ = []

def read_mag(filename):
    '''
    Read an ascii magnetometer file so that we can use the data
    here for testing.

    Parameters
    ==========
    filename : string
       Path to the file to open and parse.

    Other Parameters
    ================
    None

    Returns
    =======
    data : dict
       A dictionary containing numpy arrays of the values in the file,
       including "time", "bx", "by", "bz", etc.

    Examples
    ========
    >>> data = read_mag('data/ott_obs.txt')
    >>> data['time']
    '''

    import datetime as dt
    import numpy as np

    with open(filename, 'r') as f:
        # Skip header lines.
        trash='garbage'
        while '[year]' not in trash:
            trash = f.readline()
        # Slurp in remaning lines:
        lines = f.readlines()

    # Count lines, produce empty container.
    nLines = len(lines)
    data = {'time':np.zeros(nLines, dtype=object)}
    for v in ['bx','by','bz']:
        data[v] = np.zeros(nLines)

    # Now, loop through dem lines:
    for i, line in enumerate(lines):
        # Break into parts:
        parts = line.split()

        # Extract and convert time:
        t_raw = '_'.join(parts[:6])
        data['time'][i] =  dt.datetime.strptime(
            t_raw, '%Y_%m_%d_%H_%M_%S')

        # Extract values:
        for v, x in zip( ['bx','by','bz'],parts[-3:]):
            data[v][i] = x

    # Calculate time derivatives.  Follow the forward difference
    # method and definition of dB_H used in Pulkkinen et al. 2013.
    # Get dt values:
    dt = np.array([x.total_seconds() for x in np.diff(data['time'])])

    # Loop through variables.  Start by 
    data['dBdt'] = np.sqrt( (data['bx'][1:] - data['bx'][:-1])**2 +
                            (data['by'][1:] - data['by'][:-1])**2 ) / dt
            
    return data

class TestBinaryEventTable(unittest.TestCase):
    '''
    Test to see if the BinaryEventTable class operates
    as required.
    '''
    # For convenience:
    varnames = ['falseP','hit','miss','n','trueN']
    
    # Set true values to test against.    
    knownCount1 = {'falseP': 1,'hit': 3,'miss':5,'n': 12,'trueN': 3}
    knownCount2 = {'falseP': 1,'hit': 3,'miss':5,'n': 12,'trueN': 3}
    
    def setUp(self):
        '''
        Pre-test set up work.  If this fails, then there is a 
        syntax or other fundamental issue with BinaryEventTables.
        '''
        # Pathing information worth hanging on to:
        self.pth = os.path.dirname(os.path.abspath(__file__))
        
        # Create some "dummy data" to produce an artificial test case.
        # Simple time arrays:
        start = dt.datetime(2000,1,1,12,0,0)
        t_obs = np.array([start + dt.timedelta(minutes=4*x)
                          for x in range(15)])
        t_mod = np.array([start + dt.timedelta(minutes=4*x+2)
                          for x in range(15)])
        
        # Simple yes/no data:
        d_obs = np.zeros( t_obs.size )
        d_mod = np.zeros( t_obs.size )
        d_obs[::2] = 1. # every other one is true.
        d_mod[::4] = 1. # every forth one is true.
        
        # Binary event table for dummy data:
        self.t1 = vd.BinaryEventTable(t_obs, d_obs, t_mod, d_mod, .5, 300)

        # Now load "real world" data.  Thresholds from Pulkkinen et al., 2013;
        # Values from SWMF-SWPC test using Event 1 Ottawa station.
        obs = read_mag(self.pth+'/data/ott_obs.txt')
        mod = read_mag(self.pth+'/data/ott_mod.txt')
        self.t2 = vd.BinaryEventTable(obs['time'][:-1], obs['dBdt'],
                                      mod['time'][:-1], mod['dBdt'], .3, 60*20)
        # This fails, need to restrict time: '2003_Oct29_0600-Oct30_0600'
        
    def test_hitmiss(self):
        '''
        Test that we can get the basic numbers correct: hits, misses,
        total counts, etc.  Test the simple table and real-world table.
        '''

        for v in self.varnames:
            self.assertEqual(self.knownCount1[v], self.t1[v])

        for v in self.varnames:
            self.assertEqual(self.knownCount2[v], self.t2[v])
            
if __name__=='__main__':
    unittest.main()

