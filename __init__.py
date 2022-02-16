#!/usr/bin/env python
'''
A module containing data-model validation tools.
'''

from re import S
import unittest

# ------------------- METRICS --------------------------


def bias(o, m):
    '''
    For paired data arrays o (observed) and m (model), calculate the
    bias or mean systematic error.  See Jolliffe and Stephenson, Chapter 5,
    page 99 for details.

    A positive bias indicates that the model is consistently overpredicting
    the observations; a negative value indicates underprediction.
    '''

    from numpy import nansum

    # Check input values.
    assert o.shape == m.shape, 'Input arrays must have same shape!'
    assert o.size == m.size,   'Input arrays must have same size!'

    return nansum( m - o )/m.size


def mse(o, m):
    '''
    For paired data arrays o (observed) and m (model), calculate the
    mean squared error.
    It is a popular metric and can be found easily in review literature.

    Both arguments must be numpy arrays of the same shape with the
    same number of values.

    '''

    from numpy import nansum, sqrt

    # Check input values.
    assert o.shape == m.shape, 'Input arrays must have same shape!'
    assert o.size == m.size,   'Input arrays must have same size!'

    mse = nansum((o-m)**2.0)/o.size

    return mse


def rmse(o, m):
    '''
    For paired data arrays o (observed) and m (model), calculate the
    root mean squared error (the square root of MSE.)  The units of the
    returned value is the same as the input units.
    It is a popular metric and can be found easily in review literature.

    Both arguments must be numpy arrays of the same shape with the
    same number of values.

    '''

    from numpy import sqrt

    # Check input values.
    assert o.shape == m.shape, 'Input arrays must have same shape!'
    assert o.size == m.size,   'Input arrays must have same size!'

    rmse = sqrt(mse(o, m))

    return rmse


def nrmse(o, m, factor=None):
    '''
    For paired data arrays *o* (observed) and *m* (model), calculate the
    normalized root-mean-squared error.  This is the rmse value divided by
    the range of the data values.  This value is the more widely accepted
    version of nRMSE.

    Alternatively, the normalization factor can be overrided by setting the
    *factor* keyword, which defaults to **None** (and, therefore, the RMSE
    is normalized to the range of the observations).
    '''

    from numpy import nanmin, nanmax

    # Check input values.
    assert o.shape == m.shape, 'Input arrays must have same shape!'
    assert o.size == m.size,   'Input arrays must have same size!'

    if not factor:
        factor = (nanmax(o)-nanmin(o))

    return rmse(o, m)/factor


def nrmse_old(o, m):
    '''
    For paired data arrays o (observed) and m (model), calculate the
    normalised root-mean-squared error.  The resulting value is zero for
    a perfect prediction and infinity for an infinitely bad prediction.
    A score of 1 means that the model has the same predictive power
    as a persistance forecast with an amplitude equal to the mean of the
    observations.

    Both arguments must be numpy arrays of the same shape with the
    same number of values.

    This value is described in detail by Welling and Ridley, 2010.
    '''

    from numpy import sum, sqrt

    # Check input values.
    assert o.shape == m.shape, 'Input arrays must have same shape!'
    assert o.size == m.size,   'Input arrays must have same size!'

    nrmse = sqrt(sum((o-m)**2.0) / sum(o**2.0))

    return nrmse


def p_corr(o, m):
    '''
    For paired time series vectors o (observed) and m (model), calculate
    the Pearson correlation coefficient.  Note that this function is
    just a convenience wrapper for the Scipy function that does the same.

    Pearson correlation coefficient is a well documented value that is
    easily found in the literature.  A value of zero implies no correlation
    between the data and the model.  A value of [-]1 implies perfect
    [anti-]correlation.
    '''

    from scipy.stats import pearsonr

    # Check input values.
    assert o.size == m.size,   'Input arrays must have same size!'
    assert (len(o.shape) == 1) and (len(m.shape) == 1), \
        'Input arrays must be vectors!'

    r = pearsonr(o, m)

    return r[0]


def predicteff(o, m):
    '''
    For paired time series vectors o (observed) and m (model), calculate
    the prediction efficiency.  This metric is a measure of skill and is
    defined as 1 - (MSE/theta**2) where MSE is the Mean Square Error
    and theta**2 is the variance of the observations.  A value of 1
    indicates a perfect forecast.

    For more information, see
    http://www.swpc.noaa.gov/forecast_verification/Glossary.html#skill
    '''

    from scipy.ndimage.measurements import variance

    # Check input values.
    assert o.size == m.size,   'Input arrays must have same size!'
    assert (len(o.shape) == 1) and (len(m.shape) == 1), \
        'Input arrays must be vectors!'

    var  = variance(o)
    peff = 1.0 - mse(o,m)/var

    return peff


def pairtimeseries_linear(time1, data, time2, **kwargs):
    '''
    Use linear interpolation to pair two timeseries of data.  Data set 1
    (data) with time t1 will be interpolated to match time set 2 (t2).
    The returned values, d3, will be data set 1 at time 2.
    No extrapolation will be done; t2's boundaries should encompass those of
    t1.

    This function will correctly handle masked functions such that masked
    values will not be considered.

    **kwargs** will be handed to scipy.interpolate.interp1d
    A common option is to set fill_value='extrapolate' to prevent
    bounds errors.
    '''

    from numpy import bool_
    from numpy.ma import MaskedArray
    from scipy.interpolate import interp1d
    from matplotlib.dates import date2num, num2date

    # Dates to floats:
    t1=date2num(time1); t2=date2num(time2)

    # Remove masked values (if given):
    if type(data) == MaskedArray:
        if type(data.mask) != bool_:
            d =data[~data.mask]
            t1=t1[~data.mask]
        else:
            d=data
    else:
        d=data
    func=interp1d(t1, d, **kwargs)
    return func(t2)


class BinaryEventTable(object):
    '''
    For two unpaired timeseries, observations *Obs* with datetimes *tObs*,
    and model values *Mod* with datetimes *tMod*, create a binary event
    table using timewindow *window* (a datetime.timedelta object) using value
    threshold *cutoff*.  Such tables are powerful for creating validation
    metrics for predictive models.

    Time windows are handled so that the start of the window is inclusive
    while the end of the window is exclusive (i.e., [winStart, winStop) ).
    The start and stop times of windows are not the exact start/stop
    times of the data but relative to universal time.  For example, if your
    data starts at 5:24UT but a 20 minute time window is selected, the first
    window will start at 5:20, the second at 5:40, etc. until every data point
    (from both model and observations) lies in a window.  If any time window
    is devoid of model and/or observation data, it is discarded.

    The default behavior is that the time windows are created to span the
    entire period covered by the data and model.  However, this can be
    over-ridden using the *trange* keyword argument.  If set to a two-element
    list/tuple/array of datetimes, the time windows will span
    [trange[0], trange[1]) -- i.e., the final time window will go up to
    but not include trange[1].

    This class is fairly robust in that it can handle several non-standard
    situations.  Data gaps spanning a width greater than *window* are
    dropped from the calculation altogether (i.e., the number of windows
    evaluated is reduced by one).  If either *Mod* and *Obs* are masked
    arrays, masked values are removed.

    The returned object stores information in its attributes.
    The counts for each category (hit, miss, etc.) can be accessed using
    dictionary-like syntax.  Values can be dereferenced by using either their
    name or letter designations following Jollife & Stephenson:

    | Name     | Letter | Meaning         |
    |----------|:------:|-----------------|
    | 'hit'    |   a    | Hits            |
    | 'falseP' |   b    | False positives |
    | 'miss'   |   c    | Misses          |
    | 'trueN'  |   d    | True negatives  |
    | 'n'      |   n    | Total windows   |

    Expanded information is stored in object attributes:

    | Attribute | Use/Meaning                                               |
    |-----------|-----------------------------------------------------------|
    | trange    | The period covered, end exclusive [start, stop).          |
    | epochs    | A list of window starting times for every hit, miss, etc. |
    | nWindow   | Number of windows used.                                   |
    | Obs, Mod  | The raw observed, model data used.                        |
    | tObs,tMod | The raw time corresponding to Obs, Mod.                   |
    | threshold | The event threshold value.                                |
    | window    | A datetime.timedelta of the window size.                  |

    Finally, critical timeseries are stored as follows:

    | Attribute | Use/Meaning                                               |
    |-----------|-----------------------------------------------------------|
    | time      | An array of datetimes at bin centers.                     |
    | obsmax    | An array of max observed values for each bin.             |
    | modmax    | An array of max predicted/modeled values for each bin.    |
    | bool      | An array of booleans indicating event/non-event per bin.  |

    Parameters
    ==========
    tObs : array-like
       List or array of :class:`datetime.datetime` objects corresponding
       to the observed values.

    obs : array-like
       List or numpy array of observed values.

    tMod : array-like
       Same as tObs, but for the model/forecast times.

    Mod : array-like
       Same as Obs, but for model/forecast values.

    cutoff : real
       Threshold for value to be counted as an event.

    window : real
       Size of time window in seconds.

    Other Parameters
    ================
    trange : 2-element list-like of datetime objects
       Restricts the start and stop time of binning to the first and
       last element of *trange*.  Default is to generate the time range
       of the min and max times of *tMod* and *tObs*.

    Examples
    ========

    >>> import datetime as dt
    >>> import numpy as np
    >>> import validator as vd
    >>> # Simple time arrays:
    >>> start = dt.datetime(2000,1,1,12,0,0)
    >>> t_obs = np.array([start + dt.timedelta(minutes=4*x) for x in range(15)])
    >>> t_mod = np.array([start + dt.timedelta(minutes=4*x+2) for x in range(15)])
    >>> # Simple yes/no data:
    >>> d_obs = np.zeros( t_obs.size )
    >>> d_mod = np.zeros( t_obs.size )
    >>> d_obs[::2] = 1. # every other one is true.
    >>> d_mod[::4] = 1. # every forth one is true.
    >>> # Binary event table for dummy data:
    >>> t1 = vd.BinaryEventTable(t_obs, d_obs, t_mod, d_mod, .5, 300)

    '''

    from numpy import nan

    def __repr__(self):
        return 'Binary Event Table with {:.0f} entries'.format(self['n'])

    def __str__(self):
        return '{:.0f} hits, {:.0f} misses,'.format(self['a'], self['c']) + \
            ' {:.0f} false positives, {:.0f} true negatives.'.format(self['b'],
                                                                     self['d'])

    def __getitem__(self, key):
        return self.counts[key]

    def __setitem__(self, key, value):
        self.counts[key] = value

    def __iadd__(self, table):
        '''
        Add two tables together, combining hits and misses such that
        the resulting metrics and skill scores reflect an analysis
        using a broad data range (as opposed to, e.g., just averaging
        two Heidke skill scores togther).
        '''

        from numpy import append

        # Only add to similar objects:
        if type(table) != type(self):
            raise TypeError(
                "unsupported operand type(s) for +: {} and {}".format(
                    type(self), type(table)))

        # Only add if window and cutoff are identical.
        if (self.window != table.window) or (self.threshold !=table.threshold):
            raise ValueError("Threshold and window must be equivalent")

        # Ensure no overlapping times: DISABLED.
        # if (self.start<table.end and self.start>=table.start) or \
        #   (self.end>table.start and self.end<=table.end):
        #    raise ValueError(
        #        "Cannot add two temporally overlapping data sets in place.")

        # Combine observations, predictions, and times:
        self.Obs = append(self.Obs,  table.Obs)
        self.Mod = append(self.Mod,  table.Mod)
        self.tObs = append(self.tObs, table.tObs)
        self.tMod = append(self.tMod, table.tMod)

        # Combine timeseries:
        self.time = append(self.time, table.time)
        self.obsmax = append(self.obsmax, table.obsmax)
        self.modmax = append(self.modmax, table.modmax)
        self.bool = append(self.bool, table.bool)

        # Combine timings:
        self.nWindow += table.nWindow
        self.trange[0] = min(self.trange[0], table.trange[0])
        self.trange[1] = min(self.trange[1], table.trange[1])

        # Combine hits/misses/etc.
        self['hit'] += table['hit']
        self['miss'] += table['miss']
        self['falseP'] += table['falseP']
        self['trueN'] += table['trueN']

        self['n'] += table['n']

        # Append epoch lists:
        for category in ['hit', 'miss', 'falseP', 'trueN']:
            self.epochs[category] += table.epochs[category]

        # Update letter-designated values:
        self['a'], self['b'] = self['hit'],  self['falseP']
        self['c'], self['d'] = self['miss'], self['trueN']

        return self

    def __init__(self, tObs, Obs, tMod, Mod, cutoff, window,
                 trange=None):
        '''
        Build binary event table from scratch.
        '''

        from datetime import timedelta

        from numpy import array, ndarray
        from numpy.ma.core import MaskedArray, bool_
        from numpy import min, max, ceil, where, zeros, logical_not
        from matplotlib.dates import date2num, num2date

        # If window not a time delta, assume it is seconds.
        if type(window) is not timedelta:
            window = timedelta(seconds=window)

        # If list type data, convert to numpy arrays:
        if type(tObs) is list:
            tObs = array(tObs)
        if type(tMod) is list:
            tMod = array(tMod)
        if type(Obs) is list:
            Obs = array(Obs)
        if type(Mod) is list:
            Mod = array(Mod)

        # If handed masked arrays, collapse them to remove bad data.
        if type(Obs) == MaskedArray:
            if type(Obs.mask) != bool_:
                mask = logical_not(Obs.mask)
                Obs = Obs.compressed()
                tObs = tObs[mask]
        if type(Mod) == MaskedArray:
            if type(Mod.mask) != bool_:
                mask = logical_not(Mod.mask)
                Mod = Mod.compressed()
                tMod = tMod[mask]

        # Build start and stop times for binning.  If not provided by
        # "trange" kwarg, they must be built.
        # Using the start and stop time of the file, obtain the start and stop
        # times of our analysis (time rounded up/down according to *window*).
        # DoInclusive sets if the last window is inclusive/exclusive
        # (i.e., mathematical [start,end] vs. [start,end) behavior.)
        if not trange:
            start = date2num(min([tObs.min(), tMod.min()]))
            end = date2num(max([tObs.max(), tMod.max()]))
            DoInclusive = True
        else:
            if not isinstance(trange, (list, tuple, ndarray)):
                raise TypeError(
                    "trange must be two element list, tuple, or array")
            start, end = date2num(trange[0]), date2num(trange[-1])
            DoInclusive = False
        dT = window.total_seconds()

        # Now, adjust start and end such that they begin on round number
        # times- for example, if the time window is 5 minutes, and the
        # raw start time is 6:02 UT, the first window should start at 6:00UT.
        # Offsets for start and end times to make time windows align
        # correctly.  Round to nearest second to avoid precision issues.
        start_offset = timedelta(seconds=round(start*24*3600) % dT)
        end_offset = timedelta(seconds=round(end*24*3600) % dT)

        # Generate start and stop time.
        winstart = (num2date(start) - start_offset).replace(tzinfo=None)
        winend = (num2date(end) - end_offset).replace(tzinfo=None) \
            + DoInclusive*window

        # With start and stop times, create window information.
        nWindow = int(ceil((date2num(winend)-date2num(winstart))
                           / (dT/3600. / 24.)))
        nTime = nWindow+1

        # Create boundaries of time windows.
        time = [winstart+i*window for i in range(int(nTime))]
        #self.time = array(time)[:-1] + timedelta(seconds=int(dT/2))

        # Store these values in the object.
        self.Obs, self.tObs = Obs, tObs
        self.Mod, self.tMod = Mod, tMod
        self.window = window
        self.trange = [winstart, winend]
        self.nWindow = nWindow
        self.threshold = cutoff

        # Create timeseries arrays. Initiate as lists as there
        # may be entries with no data that we will not want to include.
        self.time = []
        self.obsmax = []
        self.modmax = []
        self.bool = []

        # Convert data to binary format: +1 for above or equal to threshold,
        # -1 for below threshold.
        # A "hit" is 2*Obs+Model=3.
        # A "true negative" is 2*Obs+Model=-3.
        # A "miss" is 2*Obs+Model=1.
        # A "false positive" is 2*Obs+Model=-1
        Obs = where(Obs >= cutoff, 1, -1)
        Mod = where(Mod >= cutoff, 1, -1)

        # Create an empty table and a dictionary of keys:
        table = {'hit':0., 'miss':0., 'falseP':0., 'trueN':0., 'n':0.}
        result = {3:'hit', 1:'miss', -1:'falseP', -3:'trueN'}

        # Create dictionary to store epochs for each
        # event type (i.e., the time for each "hit", etc.)
        self.epochs = {'hit':[], 'miss':[], 'falseP':[], 'trueN':[]}

        # Perform binary analysis.
        for i in range(nWindow):
            # print('Searching from {} to {}'.format(time[i], time[i+1]))
            # Get points inside window:
            obs_loc = (tObs >= time[i]) & (tObs < time[i+1])
            mod_loc = (tMod >= time[i]) & (tMod < time[i+1])
            subObs = Obs[obs_loc]
            subMod = Mod[mod_loc]

            # No points?  No metric!
            if not subObs.size*subMod.size:
                # print('NO RESULT from {} to {}'.format(time[i], time[i+1]))
                continue

            # Store timeseries information:
            self.time.append(time[i] + timedelta(seconds=int(dT/2)))
            self.obsmax.append(self.Obs[obs_loc].max())
            self.modmax.append(self.Mod[mod_loc].max())
            self.bool.append(self.modmax[-1] >= cutoff)

            # Determine contigency result and increment it.
            val = 2*int(subObs.max()) + int(subMod.max())
            table[result[val]] += 1
            table['n'] += 1

            # Save the current epoch into the epoch dictionary.
            self.epochs[result[val]].append(time[i])

            # print('{} from {} to {}'.format(result[val],time[i], time[i+1]))

        # Convert timeseries into arrays:
        self.time = array(self.time, dtype=object)
        self.obsmax, self.modmax = array(self.obsmax), array(self.modmax)
        self.bool = array(self.bool, dtype=bool)

        # For convenience, use the definitions from Jolliffe and Stephenson.
        table['a'], table['b'] = table['hit'],  table['falseP']
        table['c'], table['d'] = table['miss'], table['trueN']

        # Place results into object.
        self.counts = table
        self.n = table['n']

    def latex_table(self, value='values', units=''):
        '''
        Return a string that, if printed into a LaTeX source file,
        would yield the results in tabular format.

        The kwarg *value* should be set to the variable being investigated,
        e.g., tornado occurence, 40$keV$ proton flux, etc.

        If kwarg *units* is provided, add units to the threshold value
        in the table caption.
        '''

        table = r'''
        \begin{table}[ht]
        \centering
        \begin{tabular}{r|c c}
        \hline \hline
        \multicolumn{1}{c|}{Event} & \multicolumn{2}{c}{Event}\\
        \multicolumn{1}{c|}{Forecasted?} & \multicolumn{2}{c}{Observed?}\\
        & Yes & No\\
        \hline
        '''

        table += '''
        Yes   & {a:.0f}  &  {b:.0f} \\\\
        No    & {c:.0f}  &  {d:.0f} \\\\
        \\hline
        Total & {n:.0f}\\\\
            '''.format(**self.counts)

        table += r'''
        \end{tabular}
        \caption{
        '''

        table += '''Binary event table for predicted {0} using a threshold of
        {1.threshold:G}{2}.  Under these conditions, the model yielded a
        Hit Rate of {3:05.3f}, a False Alarm Rate of {4:05.3f}, and a
        Heidke Skill Score of {5:05.3f}.'''.format(
            value, self, units, self.calc_HR(),
            self.calc_FARate(), self.calc_heidke())

        table += '''}
        \end{table}
        '''

        return table

    def add_timeseries_plot(self, target=None, loc=111, xlim=None, ylim=None,
                            doLog=False):
        '''

        '''
        pass

    def calc_s(self):
        '''
        Calculate and return the base rate, which is a sample estimate
        of the marginal probability of the event occurring.
        '''
        return (self['a']+self['c'])/self['n']

    def calc_r(self):
        '''
        Calculate and return the forecast rate, which is a sample estimate
        of the marginal probability of the event being forecast.
        '''
        return (self['a']+self['b'])/self['n']

    def calc_B(self):
        '''
        Calculate frequency bias, or the ratio of the number of forecasts
        of occurrence to the number of actual occurrences.  Results range
        from [0,$\inf$]; naive of economic and forecast goals, a value of 1
        is desireable.
        This is sometimes referred to as just "bias", though it is not a
        true measure of forecast bias in the traditional sense.
        '''
        return self.calc_r()/self.calc_s()

    def calc_ar(self):
        '''
        Calculate and return a$_r$, which is the expected "a" (number of hits)
        for a random forecast with the same base rate and forecast rate.
        '''
        return (self['a']+self['b'])*(self['a']+self['c'])/self['n']

    def calc_dr(self):
        '''
        Calculate and return d$_r$, which is the expected "d" (number of
        true negatives) for a random forecast with the same base and forecast
        rate.
        '''
        return (self['b']+self['d'])*(self['c']+self['d'])/self['n']

    def calc_PC(self):
        '''
        Calculate and return Proportion Correct, or, the proportion of
        correct forecasts, defined as "hits" plus "true negatives" divided
        by total number of occurrences.
        '''
        return (self['a']+self['d'])/self['n']

    def calc_HR(self):
        '''
        Calculate and return Hit Rate, or the proportion of occurrences that
        were correctly forecast.  This is also known as probability of
        detection, and is the "hits" divided by "hits"+"misses".
        '''
        from numpy import nan

        if self['a']+self['c'] > 0:
            return self['a']/(self['a']+self['c'])
        else:
            return nan

    def calc_FARate(self):
        '''
        Calculate False Alarm Rate, also known as Probability of False
        Detection (POFD), definded as "False Positives" divided by
        "False Positives" plus "True Negatives".  It is the proportion of
        the total non-events incorrectly forecasted as events.
        '''

        from numpy import nan

        if (self['falseP']+self['trueN']) > 0:
            return self['falseP']/(self['falseP']+self['trueN'])
        else:
            return nan

    def calc_PCE(self):
        '''
        Calculate and return the Proportion Correct for a random
        forecast.  This value is a baseline, unskilled PC and is the
        basis for the Heidke Skill Score calculation.
        '''

        return \
            (self['a']+self['c'])/self['n'] * \
            (self['a']+self['b'])/self['n'] + \
            (self['b']+self['d'])/self['n'] * \
            (self['c']+self['d'])/self['n']

    def calc_heidke(self):
        '''
        Calculate and return the Heidke Skill Score, a measure of proportion
        correct adjusted for the number of forecasts that would be correct by
        random chance (i.e., in the absence of skill.)  The value ranges
        from [-1,1], where zero is no skill (the model performs as well as
        one that relies on random chance), 1 is a perfect forecast.
        A negative value, or negative "skill", is not worse than a score
        of zero; rather, it implies positive skill if the binary event
        categories were rearranged.
        '''
        from numpy import nan

        PC = self.calc_PC()
        E = self.calc_PCE()

        if (1-E) != 0:
            return (PC-E)/(1.-E)
        else:
            return nan

    def calc_bias(self):
        '''
        Calculate the event bias of the forecast. Bias indicates if the model
        overpredicts (bias>1) or underpredicts (bias<1) the frequency of
        event (i.e., threshold crossing) occurrence.
        '''

        return (self['a']+self['b'])/(self['a']+self['c'])
###############################################################################
# TEST SUITE #
###############################################################################
class TestBinaryTable(unittest.TestCase):
    '''
    Test building binary event tables, combining them with others, and
    calculating final metrics.
    '''
    import datetime as dt

    # Create some time vectors:

    start = dt.datetime(2000, 5, 2, 23, 56, 0)
    # UPDATED: LIMITED SCOPE FOR LIST COMP & EXEC WITHIN CLASS DEF
    # t1 = [start+dt.timedelta(minutes=i) for i in range(8)]
    # t2 = [t + dt.timedelta(seconds=10)  for t in t1]
    # t3 = [t + dt.timedelta(minutes=10)for i, t in enumerate(t1)]
    t1, t2, t3 = [], [], []
    for i in range(8):
        t1.append(start+dt.timedelta(minutes=i))
        t2.append(t1[-1] + dt.timedelta(seconds=10))
        t3.append(t1[-1] + dt.timedelta(minutes=10))

    # Artificial data vectors: every category (hit, miss, etc.) is used.
    # This will give us a final binary table where every value is "2".
    d1 = [0, 1, 0, 1, 0, 1, 0, 1]
    d2 = [0, 0, 1, 1, 0, 0, 1, 1]

    # Alternative "observation" that yields better scores:
    d3 = [0, 1, 0, 1, 0, 0, 0, 0]

    # Cutoff and time window for tables:
    cutoff = .5
    window = dt.timedelta(minutes=1)

    def testTable(self):
        '''
        Build a table, test results.
        '''

        tab = BinaryEventTable(self.t1, self.d1, self.t2, self.d2,
                               self.cutoff, self.window)

        # Test values in table:
        for x in 'ac':
            self.assertEqual(2, tab[x])

        # Test metric calculation:
        self.assertEqual(0.5, tab.calc_PC())
        self.assertEqual(0.5, tab.calc_HR())
        self.assertEqual(0.5, tab.calc_FARate())
        self.assertEqual(0.5, tab.calc_PCE())
        self.assertEqual(0.0, tab.calc_heidke())

    def testTableExactTime(self):
        '''
        Build a table where the observed and model times are identical.
        Test results.  This was creating issues in some applications.
        This error was caused by precision limitations when converting
        datetimes to floats using Matplotlib's dates.date2num function.
        Very small (1E-5s) errors was causing a shift in the time windows
        and missing some data-model comparisons.
        '''

        tab = BinaryEventTable(self.t3, self.d1, self.t3, self.d2,
                               self.cutoff, self.window)

        # Test values in table:
        for x in 'abcd':
            self.assertEqual(2, tab[x])

        # Test metric calculation:
        self.assertEqual(0.5, tab.calc_PC())
        self.assertEqual(0.5, tab.calc_HR())
        self.assertEqual(0.5, tab.calc_FARate())
        self.assertEqual(0.5, tab.calc_PCE())
        self.assertEqual(0.0, tab.calc_heidke())

    def testTableMetric(self):
        '''
        Test metrics when skill is nonzero.
        '''
        tab = BinaryEventTable(self.t1, self.d1, self.t2, self.d3,
                               self.cutoff, self.window)

        # Test values in table:
        for x in 'ac':
            self.assertEqual(2, tab[x])
        self.assertEqual(0, tab['b'])
        self.assertEqual(4, tab['d'])

        # Test metric calculation:
        self.assertEqual(0.75, tab.calc_PC())
        self.assertEqual(0.5, tab.calc_HR())
        self.assertEqual(0.0, tab.calc_FARate())
        self.assertEqual(0.5, tab.calc_PCE())
        self.assertEqual(0.5, tab.calc_heidke())

    def testTableAddInPlace(self):
        '''
        Test our ability to add to existing table in-place.
        '''

        # Two individual tables.  These are tested individually above.
        tab1 = BinaryEventTable(self.t1, self.d1, self.t2, self.d2,
                                self.cutoff, self.window)
        tab2 = BinaryEventTable(self.t3, self.d1, self.t3, self.d3,
                                self.cutoff, self.window)

        # Combined table (manual):
        tab3 = BinaryEventTable(self.t1+self.t3, 2*self.d1,
                                self.t2+self.t3, self.d2+self.d3,
                                self.cutoff, self.window)

        # Combined table (via sum):
        tab1 += tab2

        # Table 3 should be equivalent to tab1+tab2.
        for x in 'abcd':
            self.assertEqual(tab1[x], tab3[x])

        self.assertEqual(tab1.calc_PC(),     tab3.calc_PC())
        self.assertEqual(tab1.calc_HR(),     tab3.calc_HR())
        self.assertEqual(tab1.calc_FARate(), tab3.calc_FARate())
        self.assertEqual(tab1.calc_PCE(),    tab3.calc_PCE())
        self.assertEqual(tab1.calc_heidke(), tab3.calc_heidke())

    # def testTableAddOverlap(self):
    #    '''
    #    Ensure that adding temporally overlapping tables raises exception.
    #    '''
    #
    #    tab1=BinaryEventTable(self.t1, self.d1, self.t2, self.d2,
    #                          self.cutoff, self.window)
    #    tab2=BinaryEventTable(self.t1, self.d1, self.t2, self.d3,
    #                          self.cutoff, self.window)
    #
    #    self.assertRaises(ValueError, tab1.__iadd__, tab2)


if __name__ == '__main__':
    print(10*'=' + 'TESTING VALIDATION PACKAGE' + 10*'=')
    unittest.main()
