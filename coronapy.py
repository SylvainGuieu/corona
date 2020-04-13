""" Tools to load and handle the Johns Hawkins data on the covid-19 

Data are located here : https://github.com/CSSEGISandData/COVID-19/
"""
from pandas import  DataFrame, read_csv, concat, Series

from matplotlib.pylab import plt
from datetime import date, timedelta
import numpy as np

_min = min 
_max = max

# default urls for specfic cases
default_urls = {
#'confirmed' : 'https://github.com/CSSEGISandData/COVID-19/raw/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv',
'confirmed' : 'https://github.com/CSSEGISandData/COVID-19/raw/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv', 
#'death' : 'https://github.com/CSSEGISandData/COVID-19/raw/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Deaths.csv',
'death' : 'https://github.com/CSSEGISandData/COVID-19/raw/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv',
'recovered' : 'https://github.com/CSSEGISandData/COVID-19/raw/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv',
}

PROVINCE, COUNTRY = "Province/State", "Country/Region"
STATE, REGION = PROVINCE, COUNTRY
LAT, LONG = "Lat", "Long"

# type of the columns considered as "data" instead of header
DATA_COL_TYPE = (date, int, np.int32, np.int64)
DAY_COL_TYPE = (int, np.int32, np.int64)
###############################################################################
#          ____  _   _ ____  _     ___ ____   _____ _   _ _   _  ____         #
#         |  _ \| | | | __ )| |   |_ _/ ___| |  ___| | | | \ | |/ ___|        #
#         | |_) | | | |  _ \| |    | | |     | |_  | | | |  \| | |            #
#         |  __/| |_| | |_) | |___ | | |___  |  _| | |_| | |\  | |___         #
#         |_|    \___/|____/|_____|___\____| |_|    \___/|_| \_|\____|        #
#                                                                             #
###############################################################################
_default_styles = {}
def set_default_styles(styles):
    """ Set default styles, by state/country name for plots 
    
    Args
    ----
    styles : dict, 
             dict like {'France' : {'color':'blue'}, 'Italy':{'color':'green', 'marker':'+'}}
    """
    global _default_styles
    _default_styles = styles

def concatenate(lst):
    """ Concatenate a list of Series or DataFrame into a DataFrame """
    if not len(lst):
        return _DataSet()    
        
    lst2 = []   
    constructor = None
    for d in lst:
        if isinstance(d, Series):
            d = d.to_frame().T
        if constructor and d._constructor is not constructor:
            raise ValueError("Cannot mixt different class of data, use pd.concat instead")
        else:
            constructor = d._constructor
        lst2.append(d)
    #return constructor(concat(lst, axis=1, sort=False).T)
    #constructor = lst[0]._constructor
    return _reordered(constructor(concat(lst2)))
        
def load_data(names=("confirmed", "death", "recovered"), urls=None):
    """ Load data from Johns Hopkins University GitHub (internet coneeection requiered)
    
    Inputs
    ------
    names : str or iterable
            name of data to retrieve, default is ("confirmed", "death", "recovered")
            if an iterable is given output is a list of data
    urls: optional, a dictionary of name/url pairs default are in default_urls      
    
    Output:
        data in a DataFrame like object  or a list of Data
        data indexes are State/Region name or Country name or "State Country" for 
              embigous state with country name        
        The fours first columns is Region/Countries information and the rest are 
        data time serie cases with columns as datetime.date object.                  
    
    >>> confirmed = load_data("confirmed")
    >>> death = load_data("death")
    Or in one call : 
    >>> confirmed, death  = load_data(("confirmed", "death"))
    >>> confirmed, death, recoverd = load_data() # load the three by default
    
    """
    if not names: names = ("confirmed", "death", "recovered")
    if urls is None:
        urls = default_urls
    
    if isinstance(names, str):
        return _load_one_data(urls[names]) 
    else:
        return [_load_one_data(urls[dt]) for dt in names]    


def patch(data, patch):
    """ patch data with patch dataFrame or dict 
    
    If a data frame must have the columns 'name' 'date' and 'value'
    If a dict the items must be name,dictionary and dictionary a date/value pairs
    If a list must contains the (name,date,value) triplet
    
    e.i.: 
    { 'France' : {'2020-03-18':244 }}
    or 
    [('France', '2020-03-18', 244)]
    or 
    DataFrame([('France', '2020-03-18', 244)], columns=['name', 'date', 'value'] )        
    """
    
    if isinstance(patch , dict):
        for name, d in patch.items():            
            for date, value in d.items():
                date = _parse_date(date)
                data.loc[name, date] = value
    elif isinstance(patch, DataFrame):
        for _, p in patch.iterrows():
            date = _parse_date(p['date'])
            data.loc[p['name'], date] = p['value']
    else:
        for name, date, value in patch:
            date = _parse_date(date)
            data.loc[name, date] = value
# legacy 
patch_data = patch

def today(days=0):
    """ return the today date or shifted be an optional number of days 
    
    today_date = today()
    yesterday = today(-1) 
    """
    return date.today() + timedelta(days)

def subset(data, names=None, state=None, country=None, start=None, end=None, ndays=None):
    """ Return a new dataset with selected rows and date boundaries 
    
    Inputs
    ------
    data : DateFrame or Series like object 
    names : optional, str or iterable
            index name or names
            names are state name or country names 
            if str return a Series
    state : optional, str or iterable
            choose from slected state 
    country : optional, str or iterable 
            choose a given country/ies 
    
    start : optional, starting date
           * if the input data is indexed by dates:
             start can be : 
                - a string "yyyy-mm-dd"
                - a datetime.date object 
                - an integer if positive is offset from min date of data
                             if negative offset from max date -1 beeing the last day
                - a data frame with the column 'date' and name index, this works only if the output 
                  rerult is a Serie, ei, if `name` is str
                - None 
            * if the input data is indexed by days:
               start must be an integer matching the day index or None
    end : optional, end date
           same as start 
    ndays : optional, integer 
           number of days of the sample from the start 
           if given `end` is ignored
           
    Output
    ------
       Series like object if name is str
       DataFrame like object otherwise             
      
    Exemple
    -------
    >>> subset(confirmed, ['France', 'Italy'], start="2020-03-01")
    # last 10 days:
    >>> subset(confirmed, ['France', 'Italy'], start=-10)
    
    """
    
    if names is not None:
        data = _subset_by_names(data, names)
    elif state is not None:
        data = _subset_by_states(data, state)
    elif country is not None:
        data = _subset_by_countries(data, country)
        
    return _get_subdata_range(data, start=start, end=end, ndays=ndays)
# legacy
get_data_subset = subset

def get_cases(data):
    """ Return only the time serie data part from the DataFrame """
    _, DSLICE = _header_cases_slices(data)
    if isinstance(data, Series):
        return data[DSLICE]
    return data[data.columns[DSLICE]]


def get_header(data):
    """ Return only the header part from the DataFrame 
    (Province/State Country/Region Lat Long)
    """
    HSLICE,_ = _header_cases_slices(data)
    if isinstance(data, Series):
        return data[HSLICE]
    return data[data.columns[HSLICE]]

def get_dates(data):
    """ Return the dates from the DataFrame """
    _, DSLICE = _header_cases_slices(data)
    if isinstance(data, Series):
        return _strdate2date(data.index[DSLICE])
    return _strdate2date(data.columns[DSLICE])

def when_case_exceed(data, n):
    """ return DataFrame or a Series of the date when case start to exceed N
    
    Input
    -----
    data : DataFrame or Serie time serie like object 
    n : number of cases to match 
    
    Output
    ------
    DataFrame or Series according to input 
    
    """
    dates = get_dates(data)
    if isinstance(data, Series):
        ou = np.where( get_cases(data)>= n)[0]
        if not len(ou):
            return _max(dates)+timedelta(1)
    else:
        days = []
        for h, row in _itercases(data):                        
            ou = np.where(row>= n)[0]
            if not len(ou):
                days.append( list(h)+[_max(dates)+timedelta(1)] )
            else:
                days.append( list(h)+[dates[ou[0]]] )
        
        HSLICE, _ = _header_cases_slices(data)
        return DataFrame(days, index=data.index, columns=list(data.columns[HSLICE])+[ 'date'])    
                    
    if not len(ou):
        return _min(dates)
    return dates[ou[0]]

def is_day_indexed(data):
    """ True if data is indexed by day number instead of date """
    if 'day_zero' in data: 
        return True
    dates = get_dates(data)
    if len(dates):
        return isinstance(dates[0], DAY_COL_TYPE)
    return False
    #return 'day_zero' in data
def is_model(data):
    """ True if time sery data was created by a model """
    return ('T' in data) and ('start' in data)

def get_day_indexed(data, day_zero=None, days=None, ):
    """ Reindex data indexed by dates to day numbers 
    
    Input
    -----
    data : DataFrame or Serie time serie like object
    day_zero : DataFrame or list of day_zero dates see get_days
               if days is given, day_zero is ignored 
    days : optional, DataFrame of day numbers as returned by 
           get_days(data, day_zero). Use only if days has been previously 
           computed. if `days` is given `day_zero` argument is ignored
    Output
    ------
    DataFrame or Series according to input
      columns are day numbers instead of dates
    
    """
    if is_day_indexed(data):
        raise ValueError("Input data is already indexed by days")
    
    if days is None and day_zero is None:
        raise ValueError('need one of days or day_zero')
    if days is None:
        days = get_days(data, day_zero)
    
    hslice, dslice = _header_cases_slices(data)
    
    if isinstance(data, Series):
        row_days = _get_row(data.name, days)
                
        index = list(data.index[hslice])+['day_zero']+list(get_cases(row_days))
        sdata  = list(data[hslice])+[row_days.loc['day_zero']]+list(data.loc[get_dates(row_days)])
        return TimeSeries(sdata, index=index, name=data.name)
    
    lstdata = [] 
    for name, row in data.iterrows():
        lstdata.append(get_day_indexed(row, days=days))
    return concatenate(lstdata)    
        

def get_days(data, day_zero=None, dates=None):
    """ Get day numbers from a list of origin dates
    
    Input
    -----
    data : DataFrame or Series time serie like object
    day_zero : DataFrame or list 
            if list must contains (name,date) pairs 
            if DateFrame must contain column date and index should be state/country
    
    Output
    ------
    DataFrame or Series  according to output. The data represent the day number
    
    See Also
    --------
    get_day_indexed     
    """
    day_zero = _parse_date_arg(day_zero)
    if dates is None:
        dates = get_dates(data)
        
    hslice, dslice = _header_cases_slices(data)
    
            
    if isinstance(data, Series):
        if isinstance(day_zero, DataFrame):            
            zd = day_zero.loc[data.name]['date']
        else: 
            zd = day_zero
        
        days = _get_days(dates,zd)
        index = list(data.index[hslice])+['day_zero']+list(dates)
        ddata = list(data[hslice])+[zd]+days 
        return TimeSeries( ddata, index=index, name=data.name )        
    else:
        ddata = []
        columns = list(data.columns[hslice])+['day_zero']+list(dates)   
        
        # compute it ones only
        if not isinstance(day_zero, DataFrame): 
            days = _get_days(dates,zd)
            
        for c,row in data.iterrows():
            if isinstance(day_zero, DataFrame):            
                zd = day_zero.loc[c]['date']
                days = _get_days(dates,zd)
            else: 
                zd = day_zero
                        
            ddata.append( list(row[data.columns[hslice]])+[zd]+days) 
        return TimeFrame(ddata, columns=columns, index=data.index)
    
def date_range(start, end, step=1):
    """ date iterator frrm start, end dates and step 
    
    Args
    ----
    start : the start date, muste be a str 'yyyy-mm-dd' or a datetime.date or an integer
            if an integer this is the offset to `end`
    end : the end date, muste be a str 'yyyy-mm-dd' or a datetime.date or an integer
            if an integer this is the offset to `start`
         start and end cannot be both integer
    step : int, step in days
    
    Exemple
    -------
    >>> list(date_range("2020-03-01", "2020-03-05"))
    [datetime.date(2020, 3, 1),
    datetime.date(2020, 3, 2),
    datetime.date(2020, 3, 3),
    datetime.date(2020, 3, 4),
    datetime.date(2020, 3, 5)]
    
    >>> list(date_range("2020-03-01", +2))
    [datetime.date(2020, 3, 1),
    datetime.date(2020, 3, 2),
    datetime.date(2020, 3, 3)]
    """
    if isinstance(start, np.int):
        end = _parse_date(end)
        if not isinstance(end, date):
            raise ValueError("start and end cannot be both integers")
        start = end + timedelta(start)
          
    if isinstance(end, np.int):
        start = _parse_date(start)
        if not isinstance(start, date):
            raise ValueError("start and end cannot be both integers")
        end = start + timedelta(end)
    
    end = _parse_date(end)  
    start = _parse_date(start)  
    for d in range(0,(end-start).days+1,step):
        yield start + timedelta(d)




def intervals(data, start=None, end=None,  window=None, step=1, nwindows=None,  mindays=1):
    """ generate intervals from data 
    
    A slidding days window define the intervals from start to end
    
    Inputs
    ------
    data: dataFrame or Series like object 
    start, end: optional, None, date, DataFrame with 'date' column
                if None the min date and max date are taken respectively
    
    window : optional, int. The window size in days
             if not given window is end-start
    step : day step for the slidding window 
    nwindows : optional, max number of intervals per data row (per state/countries)
    mindays : optional, The minimum number of days tolerated per window. This can be usefull 
              if the intervals is used for fits 
    
    Output
    ------
    DataFrame : with columns 'start' en 'end' to define intervals 
    """
    intervals = []
    indexes = []
                
    if start is None:
        start = _min(get_dates(data))
    if end is None:
        end = _max(get_dates(data))
    
    if isinstance(data, Series):
        iterator = [(data.name, data)]
    else:
        iterator = data.iterrows()
    
    dx = is_day_indexed(data)
    
    if window is None and nwindows is None:
        for c,row in iterator:
            s, e = _compute_date_range(row, start, end)
            me = _max( get_dates(row) )
            if dx:
                x = get_cases(row, subset(start=start,end=end))
                days = len(np.where(x==x)[0])
                #days = (min(me,end)-start) + 1
            else:
                days = (_min(me,end)-start).days + 1
            if days>=mindays:                                    
                intervals.append[(c, start, end, days)]
    else:            
                        
        for c, rowdata in iterator:
            s, e = _compute_date_range(rowdata, start, end)
            if window is None:
                if dx:
                    w = int(np.ceil((e-s)/nwindows))
                else:
                    w = int(np.ceil((e-s).days/nwindows))
                n = nwindows
            else:
                w = window
                n = 99999999 if nwindows is None else nwindows
            
            d = s
            i=0
            while (d<=e) and (i<n):
                ss,ee = _compute_date_range(rowdata, d, ndays=w)
                me = _max( get_dates(rowdata) )
                if dx:
                    x = get_cases(subset(rowdata, start=ss,end=ee))
                    days = len(np.where(x==x)[0])  
                    #days = (min(me,ee)-ss) +1       
                else:           
                    days = (_min(me,ee)-ss).days +1
                if days>=mindays:
                    intervals.append([c, ss, ee, days])
                    i+=1
                if dx:
                    d = d+step
                else: 
                    d = d + timedelta(days=step)
    return DataFrame(intervals, columns=['name', 'start', 'end', 'days'])    
#legacy 
gen_intervals = intervals

def split(data, intervals):
    """ split data into a list of Series with given date intervals 
    
    Inputs
    ------
    data : DataFrame or Series like object
    intervals :     
       - a DateFrame with columns 'name', 'start', 'end'
       - a list with tripplet (name, start, end)
    
    Output
    ------
    list : list containing Series
    
    see also intervals
    """
    output = []    
    intervals = _parse_intervals_arg(intervals)
    dataiterator = [(data.name, data)] if isinstance(data, Series) else data.iterrows()
            
    for c, rowdata in dataiterator:
        it = intervals[intervals['name']==c]
        if not len(it): continue
        iterator = [(None,it)] if isinstance(it, Series) else it.iterrows()        
        output.extend((subset(rowdata, start=i['start'], end=i['end']) for _,i in iterator))
   
    return output
#legacy 
split_cases = split

def sum(data, name=None, axis=0, **kwargs):
    """ Sum a TimeFrame cases along rows 
    
    Inputs
    ------
    data : TimeFrame data
    name : optional, name of the time_series
           If not given try to guess one from country or state if not mixed
    axis : optional, axis=0 
           change only to an other value to have the normal DataFrame behavior
    
    Output
    ------
    TimeSeries : Series like object 
    """
    return _run_operator(data, np.sum, axis, name, kwargs)

def mean(data, name=None, axis=0, **kwargs):
    """ mean of a TimeFrame cases along rows 
    
    Inputs
    ------
    data : TimeFrame data
    name : optional, name of the time_series
           If not given try to guess one from country or state if not mixed
    axis : optional, axis=0 
           change only to an other value to have the normal DataFrame behavior
    
    Output
    ------
    TimeSeries : Series like object 
    """
    return _run_operator(data, np.mean, axis, name, kwargs)

def median(data, name=None, axis=0, **kwargs):
    """ median of a TimeFrame cases along rows 
    
    Inputs
    ------
    data : TimeFrame data
    name : optional, name of the time_series
           If not given try to guess one from country or state if not mixed
    axis : optional, axis=0 
           change only to an other value to have the normal DataFrame behavior
    
    Output
    ------
    TimeSeries : Series like object 
    """
    return _run_operator(data, np.median, axis, name, kwargs)

def std(data, name=None, axis=0, **kwargs):
    """ standard deviation of a TimeFrame cases along rows 
    
    Inputs
    ------
    data : TimeFrame data
    name : optional, name of the time_series
           If not given try to guess one from country or state if not mixed
    axis : optional, axis=0 
           change only to an other value to have the normal DataFrame behavior
    
    Output
    ------
    TimeSeries : Series like object 
    """
    return _run_operator(data, np.std, axis, name, kwargs)

def minimum(data, name=None, axis=0, **kwargs):
    """ min of a TimeFrame cases along rows 
    
    Inputs
    ------
    data : TimeFrame data
    name : optional, name of the time_series
           If not given try to guess one from country or state if not mixed
    axis : optional, axis=0 
           change only to an other value to have the normal DataFrame behavior
    
    Output
    ------
    TimeSeries : Series like object 
    """
    return _run_operator(data, np.min, axis, name, kwargs)

def maximum(data, name=None, axis=0, **kwargs):
    """ min of a TimeFrame cases along rows 
    
    Inputs
    ------
    data : TimeFrame data
    name : optional, name of the time_series
           If not given try to guess one from country or state if not mixed
    axis : optional, axis=0 
           change only to an other value to have the normal DataFrame behavior
    
    Output
    ------
    TimeSeries : Series like object 
    """
    return _run_operator(data, np.max, axis, name, kwargs)




def fit(data, ftype='2'):
    """ fit a time series cases
    
    Input
    -----
    data : DataFrame or Series like of time serie data
    ftype : fits type must be:
            - '2' (default) to fit $A 2^{t/T}$
            - 'exp' to fit A \exp^{t/T}
          When T is representing a period (doubling time in case of '2')
          t=0 at the begining of the data set, so A is the number of cases at t=0
    
    Output
    ------
    DataFrame or Series like object according to input
        The output fit result data contains 'T', 'A' 'ftype' 'start' and 'end' columns 
    """
    if ftype == '2':
        fit_func =  _sfit
        fit_repr = '$A 2^{t/T}$'
    if ftype == 'exp':
        fit_func =  _efit
        fit_repr = '$A \exp^{t/T}$', 'exp'
    
    dates = get_dates(data)
    mindate = _min(dates)
    if is_day_indexed(data):
        days = dates.astype(np.int64)        
        # days has to be positives
        days = np.asarray(days-mindate)
    else:        
        days =  np.asarray([(d - mindate).days for d in dates])
     
    columns = ['T', 'A', 'rep', 'ftype', 'start', 'end', 'label', 'days']
    hslice, dslice = _header_cases_slices(data)
    
    if isinstance( data, Series):
        cdata = get_cases(data)
        if not len(cdata):
            return FitSeries(index=list(data.index[hslice])+columns, name=data.name)
        
        # remove nan
        t = cdata == cdata
        
        A, T = fit_func(days[t], cdata[t])
        
        result = list(data[hslice]) + [T, A, fit_repr, ftype, mindate, _max(dates),
                          '%s T=%.2f'%(data.name, T), len(cdata)]
        return FitSeries(result, index=list(data.index[hslice])+columns, name=data.name)
    
    else:
        results = []
        for header, cdata in _itercases(data):           
            # remove nans 
            t = cdata == cdata
            A, T = fit_func(days[t], cdata[t])  
            results.append( list(header) + [        
            T, A, fit_repr, ftype,
            mindate, _max(dates),
            '%s T=%.2f'%(cdata.name, T), len(cdata)
            ]
            )        
        return FitDataFrame(results, columns=list(data.columns[hslice])+columns, index=data.index)
#legacy 
fit_cases = fit

def fit_severals(data_list, ftype='2'):
    """fit a list of Series and return result in a single DataFrame 
    
    see fit
    """
    result = [fit(data, ftype) for data in data_list]
    return concatenate(result)

def fit_intervals(data, intervals, ftype='2'):
    """fit the data from intervals and  return result in a single DataFrame 
    
    see fit for the fits 
    and intervals and split function for the interval argument 
    """
    data_list = split(data, intervals)
    return fit_severals(data_list, ftype)


def make_model(fit_result, dates=None):
    """ make a time series data from a fit result and dates
    
    Inputs
    ------
    fit_result : Series or DataFrame like as returned by fit
    dates: optional, None, 2xtuple of min,max range, list of dates
            default is None: min,max is taken from 'strat', 'end' columns of fit_result
    
    Output
    ------
    Series or DataFrame according to input
         time series data like the John H ones
    """
    if not isinstance(fit_result, Series):
        # vectorialize the function to DataFrame
        return _concatenate([make_model(r,dates) for _,r in fit_result.iterrows()])
    
    if dates is None:
        if isinstance(fit_result, Series):        
            dates  = (fit_result['start'], fit_result['end'])
        else:
            # same range for all the DataFrame
            dates = (_min(fit_result['start']), _max(fit_result['end']))
    
    if isinstance(dates, tuple):
        mindate, maxdate = dates
        if is_day_indexed(fit_result):
            dates = list(range(mindate, maxdate+1))
        else:    
            dates = list(date_range(mindate, maxdate))            
        
    model = _get_fit_model_func(fit_result['ftype'])
    cases = model(dates, fit_result['A'], fit_result['T'], fit_result['start'])
    index = list(fit_result.index)
    index += list(dates)
    data = list(fit_result)
    data += list(cases)
    return TimeSeries(data, index=index, name=fit_result.name)


###############################################################################
#           ____  _     ___ _____   ____  _   _ ____  _     ___ ____          #
#          |  _ \| |   / _ \_   _| |  _ \| | | | __ )| |   |_ _/ ___|         #
#          | |_) | |  | | | || |   | |_) | | | |  _ \| |    | | |             #
#          |  __/| |__| |_| || |   |  __/| |_| | |_) | |___ | | |___          #
#          |_|   |_____\___/ |_|   |_|    \___/|____/|_____|___\____|         #
#                                                                             #
###############################################################################

def clean_date_xticks(dates=None, axes=None, **kwargs):
    """ make nicer readable label if xaxis is a date 
    Input
    -----
    dates : optional, None, or list of dates
            if None taken from axes
    axes : matplotlib axes, if not taken from plt.gca()
    **kwargs :  kwargs for axes.set_xticklabels method
    """        
    if axes is None:
        axes = plt.gca()
                
    if dates is None:
        if axes.xaxis.converter is None:
            return 
        l1, l2 = axes.get_xlim()
        dates = (date.fromordinal(int(np.floor(l1))), date.fromordinal(int(np.ceil(l2))))
        
    kwargs.setdefault('rotation', 75)    
    ticks, labels = _dates2ticks(dates)
    axes.set_xticks(ticks)
    labels = axes.set_xticklabels(labels, **kwargs)        
    for l in labels:
        l.update(kwargs)

def plot_cases(data, log=False, axes=None, style=None, styles=None, fit_result=None, **kwargs):
    """ Plot data cases for each data rows 
    Input
    -----
    data : DataFrame or Series time series data
    log : optional, False, True, 2,  10
          plot the log, log2, or log10 of cases instead
          Note: do not confuse with the yscale="log" axes parameter
    axes : optional, matplotlib axes. If None axes=plt.gca() 
    style : style dictionary for the plt.plot function
    styles : dictionary of name/style dictionary to style for a given country 
             One can set the default styles with the public 
             set_default_styles(styles) function
             **styles** dictionary is updated to at least color 
    **kwargs : additional kwargs to plot function, concatenate to style 
    
    Output
    ------
    axes : matplotlib axes
    """
    iterator, axes, style, styles = _parse_plot_args(data, axes, style, styles)    
    
    labeler = _labeler(data)    
    for name, row in iterator:
        dates = get_dates(row)
        
        st = _styler(name, style, styles)
        st = dict(st, **kwargs)
        
        cases = get_cases(row)
        
        st.setdefault('label', labeler(row))        
        st.setdefault('linestyle', _next_linestyle(name, axes))
        
        # apply log formulae is needed
        y = _parse_log(log, cases)
        
        axes.plot(dates, y, **st)
        _update_styles(axes, name, styles)        
        
    if fit_result is not None:
        plot_cases(make_model(fit_result), log=log, style=style, styles=styles, axes=axes)
    
    axes.legend()
    # make nicer ticks when date         
    clean_date_xticks(axes=axes, rotation=75)                
    return axes

def plot_model(fit_result, dates=None, **kwargs):    
    """ convenient function make a model from result of a fit and plot 
    
    plot_model(fit_result, dates, **kwargs)
    is equivalent to 
    plot_cases(make_model(fit_result, dates), **kwargs)
    """
    return plot_cases(make_model(fit_result, dates), **kwargs)
    
    
def plot_proportion(numerator, denominator, scale=1.0, log=False, axes=None, style=None, styles=None, **kwargs):
    """ convenient function plot a fraction from numerator and denominator data
    
    Input
    -----
    numerator, denominator:  DataFrame or Series time series data
    scale : float, fraction scale default is 1.0, set it to 100 for percentils 
    axes : optional, matplotlib axes. If None axes=plt.gca() 
    style : style dictionary for the plt.plot function
    styles : dictionary of name/style dictionary to style for a given country 
             One can set the default styles with the public 
             set_default_styles(styles) function
             **styles** dictionary is updated to at least color 
    **kwargs : additional kwargs to plot function, concatenate to style 
    
    Output
    ------
    axes : matplotlib axes
    """
    iterator, axes, style, styles = _parse_plot_args(numerator, axes, style, styles)        
    
    labeler = _labeler(numerator)
                    
    for (name, row) in iterator:               
        dates = get_dates(row)
        
        n = get_cases(row)
        # get the denominator of same name
        d = _get_row(name, denominator)                
        d = d.loc[n.index]
        
        st = _styler(name, style, styles)
        st = dict(st, **kwargs)
                
        st.setdefault('label',labeler(n))        
        st.setdefault('linestyle', _next_linestyle(name, axes))
        
        y =_parse_log(log, n/d*scale)       
        axes.plot(dates, y, **st)
        _update_styles(axes, n.name, styles)

    axes.legend()             
    clean_date_xticks(axes=axes, rotation=75)        
    return axes
    

def plot_fit_result(result, datekey="end", ykey="T", axes=None, styles=None, style=None, **kwargs):
    """ Plot fit results, one line per state/country 
    
    Input
    -----
    result : Series or DataFrame as returned by fit
    datekey : optional, 'start', 'end' (default), 'middle' 
              which date to plot on xaxis the start date of the fit end or middle 
    ykey : optional, str, The column to lot in y default is 'T'
    axes : optional, matplotlib axes. If None axes=plt.gca() 
    style : style dictionary for the plt.plot function
    styles : dictionary of name/style dictionary to style for a given country 
             One can set the default styles with the public 
             set_default_styles(styles) function
             **styles** dictionary is updated to at least color 
    
    **kwargs : additional kwargs to plot function, concatenate to style 
    
    Output
    ------
    axes : matplotlib axes
    """    
    _, axes, style, styles = _parse_plot_args(result, axes, style, styles)    
            
    
    names = set(result.index)
        
    def get_key(r, key):
        if key in ["center", "middle"]:
            return r['start'] + (r['end'] - r['start'])/2
        return r[key]
    
    for name in names:
        res = result.loc[name]
        st = _styler(name, style, styles)
        st = dict(st, **kwargs)
        
        if is_day_indexed(result):
            label = "%s %s"%(name, res.iloc[0]['day_zero'])
        else:
            label = name                
        st.setdefault('label', label)
        
        x = get_key(res, datekey)
        axes.plot(x, res[ykey], **st)
    
    axes.legend()
    #axes.set_xlabel('Date' if day_zero is None else 'Days')
    if not axes.get_ylabel():
        axes.set_ylabel(ykey)
            
    clean_date_xticks(axes=axes, rotation=75)
    return axes    


###############################################################################
#                   ____ _        _    ____ ____  _____ ____                  #
#                  / ___| |      / \  / ___/ ___|| ____/ ___|                 #
#                 | |   | |     / _ \ \___ \___ \|  _| \___ \                 #
#                 | |___| |___ / ___ \ ___) |__) | |___ ___) |                #
#                  \____|_____/_/   \_\____/____/|_____|____/                 #
#                                                                             #
###############################################################################


class _DataSet(DataFrame):
    """ A DataFrame agremented to some usefull functions """
    @property
    def _constructor(self):
        return _DataSet
        
    @property
    def _constructor_sliced(self):
        return _DataRow
    
    
    def subset(self, *args, **kwargs):
        return subset(self, *args, **kwargs)
        
class _DataRow(Series):
    @property
    def _constructor(self):
        return _DataRow

    @property
    def _constructor_expanddim(self):
        return _DataSet
    
    def subset(self, *args, **kwargs):
        return subset(self, *args, **kwargs)
    
    
class TimeFrame(_DataSet):
    """ DataFrame like object to handle Cases TimeSeries of John Hopkins University
    
    Properties
    ----------
    - dates:  return the column dates as datetime.date
    - cases:  Return only the time series parts of the data (for math operation) and
              drop the "header" part. 
    - header: Return only the "header" part
    - daily_cases: number of cases per day TimeFrame
    
    Methods
    -------
    get_days(day_zero): return a DataFrame of days from given origin Date
    get_day_indexed(day_zero): return a TimeFrame DataFrame with day number as
                               columns instead of date
    
    is_day_indexed() : True if columns are day numbers instead of date 
    split(intervals) : Split the TimeFrame Data into list of TimeSeries 
    intervals(...) : Generate intervals DataFrame
    fit(ftype='2') : Fit the data and return A FitDataFrame containing fit results
                     for each rows
    fit_intervals(intervals, ftype='2'): Fit the data for several intervals
    plot(...) :  plot all the cases, one line per row
    plot_proportion(denominator) : plot self over denominator 
    when_case_exceed(n) : return a DataFrame of date matching the date when case exceed n
    patch(patch) : patch the data 
    
    sum(), mean(), median(), std() :
         Apply operation along rows and return a TimeSeries with guessed header 
    
                    
    """
    @property
    def _constructor(self):
        return TimeFrame
    
    @property
    def _constructor_sliced(self):
        return TimeSeries
    
    @property
    def dates(self):
        return get_dates(self)    
    
    @property
    def cases(self):
        return get_cases(self)
        
    @property
    def header(self):
        return get_header(self)
    
    @property
    def daily_cases(self):
        cases = self.cases
        return cases.iloc[:,1:] - np.asarray(cases.iloc[:,0:-1])
        
    def get_days(self, day_zero=None):
        """ Get day numbers from a list of origin dates
        
        Input
        -----
        day_zero : DataFrame or list 
                if list must contains (name,date) pairs 
                if DateFrame must contain column date and index should be state/country
        
        Output
        ------
        DataFrame : The data represent the day number
        
        See Also
        --------
        get_day_indexed method     
        """
        return get_days(self, day_zero=day_zero)    
    
    def get_day_indexed(self, day_zero=None, days=None):
        """ Reindex data indexed by dates to day numbers
        
        Input
        -----
        day_zero : DataFrame or list of day_zero dates see get_days
                   if days is given, day_zero is ignored 
        days : optional, DataFrame of day numbers as returned by 
               get_days(data, day_zero). Use only if days has been previously 
               computed. if `days` is given `day_zero` argument is ignored
        Output
        ------
        TimeFrame (DataFrame like) 
          where columns are day numbers instead of dates
        
        """
        return get_day_indexed(self, day_zero=day_zero, days=days)
    
    def is_day_indexed(self):
        """ True if the TimeFrame data if indexed by day number instead of dates """
        return is_day_indexed(self)
    
    def split(self, intervals):
        """ split data into a list of Series with given date intervals 
        
        Inputs
        ------
        intervals :     
           - a DateFrame with columns 'name', 'start', 'end'
           - a list with tripplet (name, start, end)
                
        Output
        ------
        list : list containing TimeSeries
        
        see also intervals method 
        """
        return split(self, intervals)
    
    def intervals(self, *args, **kwargs):
        """ generate intervals 
        
        A slidding days window define the intervals from start to end
        
        Inputs
        ------
        start, end: optional, None, date, DataFrame with 'date' column
                    if None the min date and max date are taken respectively
        
        window : optional, int. The window size in days
                 if not given window is end-start
        step : day step for the slidding window 
        nwindows : optional, max number of intervals per data row (per state/countries)
        mindays : optional, The minimum number of days tolerated per window. This can be usefull 
                  if the intervals is used for fits 
        
        Output
        ------
        DataFrame : with columns 'start' en 'end' to define intervals 
        """
        return intervals(self, *args, **kwargs)
    
    
    def fit_intervals(self, intervals, ftype='2'):
        """fit the data from intervals and  return result in a single DataFrame 
        
        see .fit method for the fits 
        and .intervals and .split method for the interval argument 
        """
        return fit_intervals(self, intervals, ftype=ftype)
    
    def fit(self, ftype='2'):
        """ fit the cases for each rows 
        
        Input
        -----
        ftype : fits type must be:
                - '2' (default) to fit $A 2^{t/T}$
                - 'exp' to fit A \exp^{t/T}
              When T is representing a period (doubling time in case of '2')
              t=0 at the begining of the data set, so A is the number of cases at t=0
        
        Output
        ------
        FitDataFrame 
            The output fit result data contains 'T', 'A' 'ftype' 'start' and 'end' columns 
        """
        return fit(self, ftype)
    
    def plot(self, *args, **kwargs):
        """ Plot cases for each rows 
        
        Input
        -----
        log : optional, False, True, 2,  10
              plot the log, log2, or log10 of cases instead
              Note: do not confuse with the yscale="log" axes parameter
        axes : optional, matplotlib axes. If None axes=plt.gca() 
        style : style dictionary for the plt.plot function
        styles : dictionary of name/style dictionary to style for a given country 
                 One can set the default styles with the public 
                 set_default_styles(styles) function
                 **styles** dictionary is updated to at least color 
        **kwargs : additional kwargs to plot function, concatenate to style 
        
        Output
        ------
        axes : matplotlib axes
        """
        return plot_cases(self, *args, **kwargs)
    
    def plot_proportion(self, denominator, *args, **kwargs):
        """ convenient function plot a fraction of self over denominator cases
        
        Input
        -----
        denominator:  TimeFrame or TimeSeries (must share same columns than self)
        scale : float, fraction scale default is 1.0, set it to 100 for percentils 
        axes : optional, matplotlib axes. If None axes=plt.gca() 
        style : style dictionary for the plt.plot function
        styles : dictionary of name/style dictionary to style for a given country 
                 One can set the default styles with the public 
                 set_default_styles(styles) function
                 **styles** dictionary is updated to at least color 
        **kwargs : additional kwargs to plot function, concatenate to style 
        
        Output
        ------
        axes : matplotlib axes
        """
        return plot_proportion(self, denominator, *args, **kwargs)
    
    def when_case_exceed(self, n):
        """ return DataFrame the date when case start to exceed N
        
        Input
        -----
        n : number of cases to match 
        
        Output
        ------
        DataFrame  with column 'date'
        
        """
        return when_case_exceed(self, n)
        
    def patch(self, patch_data):
        patch(self, patch_data)
    
    def sum(self, name=None, axis=0, **kwargs):
        """ Sum cases along rows 
        
        Inputs
        ------
        name : optional, name of the time_series
               If not given try to guess one from country or state if not mixed
        axis : optional, axis=0 
               change only to an other value to have the normal DataFrame behavior
        
        Output
        ------
        TimeSeries : Series like object 
        """
        return _run_operator(self, np.sum, axis, name, kwargs)

    def mean(self, name=None, axis=0, **kwargs):
        """ maen cases along rows 
        
        Inputs
        ------
        name : optional, name of the time_series
               If not given try to guess one from country or state if not mixed
        axis : optional, axis=0 
               change only to an other value to have the normal DataFrame behavior
        
        Output
        ------
        TimeSeries : Series like object 
        """
        return _run_operator(self, np.mean, axis, name, kwargs)

    def median(self, name=None, axis=0, **kwargs):
        """ Smedian cases along rows 
        
        Inputs
        ------
        name : optional, name of the time_series
               If not given try to guess one from country or state if not mixed
        axis : optional, axis=0 
               change only to an other value to have the normal DataFrame behavior
        
        Output
        ------
        TimeSeries : Series like object 
        """
        return _run_operator(self, np.median, axis, name, kwargs)

    def std(self, name=None, axis=0, **kwargs):
        """ cases standard deviation along rows 
        
        Inputs
        ------
        name : optional, name of the time_series
               If not given try to guess one from country or state if not mixed
        axis : optional, axis=0 
               change only to an other value to have the normal DataFrame behavior
        
        Output
        ------
        TimeSeries : Series like object 
        """
        return _run_operator(self, np.std, axis, name, kwargs)
    
    def min(self, name=None, axis=0, **kwargs):
        """ min cases along rows 
        
        Inputs
        ------
        name : optional, name of the time_series
               If not given try to guess one from country or state if not mixed
        axis : optional, axis=0 
               change only to an other value to have the normal DataFrame behavior
        
        Output
        ------
        TimeSeries : Series like object 
        """
        return _run_operator(self, np.min, axis, name, kwargs)
    
    def max(self, name=None, axis=0, **kwargs):
        """ max cases along rows 
        
        Inputs
        ------
        name : optional, name of the time_series
               If not given try to guess one from country or state if not mixed
        axis : optional, axis=0 
               change only to an other value to have the normal DataFrame behavior
        
        Output
        ------
        TimeSeries : Series like object 
        """
        return _run_operator(self, np.max, axis, name, kwargs)


    
    
class TimeSeries(_DataRow):
    """Series like object to handle Cases Time Series of John Hopkins University
    
    Properties
    ----------
    - dates:  return the column dates as datetime.date
    - cases:  Return only the time series parts of the data (for operation) and
              drop the "header" part. 
    - header: Return only the "header" part
    - daily_cases : number of cases per day TimeSeries
    Methods
    -------
    get_days(day_zero): return a Series of days from given origin Date
    get_day_indexed(day_zero): return a TimeSeries with day number as
                               columns instead of date
    
    is_day_indexed() : True if columns are day numbers instead of date 
    split(intervals) : Split the TimeSeries Data into list of TimeSeries 
    intervals(...) : Generate intervals DataFrame
    fit(ftype='2') : Fit the data and return A FitDataFrame containing fit results
                     for each rows
    fit_intervals(intervals, ftype='2'): Fit the data for several intervals
    plot(...) :  plot the cases
    plot_proportion(denominator) : plot self over denominator 
    when_case_exceed(n) : return a Series of date matching the date when case exceed n
    patch(patch) : patch the data 
    """
    @property
    def _constructor(self):
        return TimeSeries

    @property
    def _constructor_expanddim(self):
        return TimeFrame
    
    @property
    def dates(self):
        return get_dates(self)
    
    @property
    def cases(self):
        return get_cases(self)
        
    @property
    def header(self):
        return get_header(self)
    
    @property
    def daily_cases(self):
        cases = self.cases
        return cases[1:] - np.asarray(cases[0:-1])
    
    
    def get_days(self, day_zero=None):
        """ Get day numbers from a list of origin dates
        
        Input
        -----
        day_zero : DataFrame or list 
                if list must contains (name,date) pairs 
                if DateFrame must contain column date and index should be state/country
        
        Output
        ------
        Series   The data represent the day number
        
        See Also
        --------
        get_day_indexed method    
        """
        return get_days(self, day_zero=day_zero)    
    
    def get_day_indexed(self, day_zero=None, days=None):
        """ Reindex data indexed by dates to day numbers 
        
        Input
        -----
        data : DataFrame or Serie time serie like object
        day_zero : DataFrame or list of day_zero dates see get_days
                   if days is given, day_zero is ignored 
        days : optional, DataFrame of day numbers as returned by 
               get_days(data, day_zero). Use only if days has been previously 
               computed. if `days` is given `day_zero` argument is ignored
        Output
        ------
        TimeSeries 
          indexes are day numbers instead of dates
        
        """
        return get_day_indexed(self, day_zero=day_zero, days=days)
    
    def is_day_indexed(self):
        """ True if the TimeFrame data if indexed by day number instead of dates """
        return is_day_indexed(self)
    
    def split(self, intervals):
        """ split data into a list of Series with given date intervals 
        
        Inputs
        ------
        intervals :     
           - a DateFrame with columns 'name', 'start', 'end'
           - a list with tripplet (name, start, end)
                
        Output
        ------
        list : list containing TimeSeries
        
        see also intervals method 
        """
        return split(self, intervals)
    
    def intervals(self, *args, **kwargs):
        """
        generate intervals 
        
        A slidding days window define the intervals from start to end
        
        Inputs
        ------
        start, end: optional, None, date, DataFrame with 'date' column
                    if None the min date and max date are taken respectively
        
        window : optional, int. The window size in days
                 if not given window is end-start
        step : day step for the slidding window 
        nwindows : optional, max number of intervals per data row (per state/countries)
        mindays : optional, The minimum number of days tolerated per window. This can be usefull 
                  if the intervals is used for fits 
        
        Output
        ------
        DataFrame : with columns 'start' en 'end' to define intervals 
        """
        return intervals(self, *args, **kwargs)
    
    def fit_intervals(self, intervals, ftype='2'):
        """fit the data from intervals and  return result in a single DataFrame 
        
        see .fit method for the fits 
        and .intervals and .split method for the interval argument 
        """
        return fit_intervals(self, intervals, ftype=ftype)
    
    def fit(self, ftype='2'):
        """ fit the cases 
        
        Input
        -----
        ftype : fits type must be:
                - '2' (default) to fit $A 2^{t/T}$
                - 'exp' to fit A \exp^{t/T}
              When T is representing a period (doubling time in case of '2')
              t=0 at the begining of the data set, so A is the number of cases at t=0
        
        Output
        ------
        FitDataFrame 
            The output fit result data contains 'T', 'A' 'ftype' 'start' and 'end' columns 
        """
        return fit(self, ftype)
    
    def plot(self, *args, **kwargs):
        """ Plot cases 
        
        Input
        -----
        log : optional, False, True, 2,  10
              plot the log, log2, or log10 of cases instead
              Note: do not confuse with the yscale="log" axes parameter
        axes : optional, matplotlib axes. If None axes=plt.gca() 
        style : style dictionary for the plt.plot function
        styles : dictionary of name/style dictionary to style for a given country 
                 One can set the default styles with the public 
                 set_default_styles(styles) function
                 **styles** dictionary is updated to at least color 
        **kwargs : additional kwargs to plot function, concatenate to style 
        
        Output
        ------
        axes : matplotlib axes
        """
        return plot_cases(self, *args, **kwargs)
    
    def plot_proportion(self, denominator, *args, **kwargs):
        """ convenient function plot a fraction of self over denominator cases
        
        Input
        -----
        denominator:  TimeFrame or TimeSeries (must share same columns than self)
        scale : float, fraction scale default is 1.0, set it to 100 for percentils 
        axes : optional, matplotlib axes. If None axes=plt.gca() 
        style : style dictionary for the plt.plot function
        styles : dictionary of name/style dictionary to style for a given country 
                 One can set the default styles with the public 
                 set_default_styles(styles) function
                 **styles** dictionary is updated to at least color 
        **kwargs : additional kwargs to plot function, concatenate to style 
        
        Output
        ------
        axes : matplotlib axes
        """
        return plot_proportion(self, denominator, *args, **kwargs)
    
    def when_case_exceed(self, n):
        """ return a Series the date when case start to exceed N
        
        Input
        -----
        n : number of cases to match 
        
        Output
        ------
        Series  with index 'date'
        
        """
        return when_case_exceed(self, n)    
    
    
        
class FitDataFrame(_DataSet):
    """ DataFrame handling fit results of cases 
    
    methods
    -------
    plot(...):  
            plot the results xaxis as date and 'T' as yaxis by default 
            One plot line per state/country 
    plot_model(dates=None, ...): 
            plot the model (cases function to date) according
            optional dates or dates range
    make_model(dates=None):
            Make a TimeFrame with date as columns and theoritical cases
                                       
    """
    @property
    def _constructor(self):
        return FitDataFrame
    
    @property
    def _constructor_sliced(self):
        return FitSeries
    
    def plot(self, *args, **kwargs):
        """ Plot the fit results, one plot line per state/country 
        
        Input
        -----
        datekey : optional, 'start', 'end' (default), 'middle' 
                  which date to plot on xaxis the start date of the fit end or middle 
        ykey : optional, str, The column to lot in y default is 'T'
        axes : optional, matplotlib axes. If None axes=plt.gca() 
        style : style dictionary for the plt.plot function
        styles : dictionary of name/style dictionary to style for a given country 
                 One can set the default styles with the public 
                 set_default_styles(styles) function
                 **styles** dictionary is updated to at least color 
        
        **kwargs : additional kwargs to plot function, concatenate to style 
        
        Output
        ------
        axes : matplotlib axes
        """ 
        return plot_fit_result(self, *args, **kwargs)
    
    def plot_model(self, *args, **kwargs):
        """ convenient function make a model from result of a fit and plot 
        
        
        fit_result.plot_model(dates, **kwargs)
        is equivalent to 
        fit_result.make_model(dates).plot(**kwargs)
        """
        return plot_model(self,  *args, **kwargs)
    
    def make_model(self, dates=None):
        """ make a TimeFrame data from dates
        
        Inputs
        ------
        dates: optional, None, 2xtuple of min,max range, list of dates
                default is None: min,max is taken from 'strat', 'end' columns of fit_result
        
        Output
        ------
        TimeFrame (DataFrame like)
             Time series data like the John H ones
        """
        return make_model(self, dates)
    
    
class FitSeries(_DataRow):
    """
    methods
    -------
    plot(...):  
            plot the results xaxis as date and 'T' as yaxis by default 
    plot_model(dates=None, ...): 
            plot the model (cases function to date) according
            optional dates or dates range
    make_model(dates=None):
            Make a TimeSeries with date as index and theoritical cases
    """            
    @property
    def _constructor(self):
        return FitSeries

    @property
    def _constructor_expanddim(self):
        return FitDataFrame
    
    def plot(self, *args, **kwargs):
        """ Plot the fit results, one single point 
        
        Input
        -----
        datekey : optional, 'start', 'end' (default), 'middle' 
                  which date to plot on xaxis the start date of the fit end or middle 
        ykey : optional, str, The column to lot in y default is 'T'
        axes : optional, matplotlib axes. If None axes=plt.gca() 
        style : style dictionary for the plt.plot function
        styles : dictionary of name/style dictionary to style for a given country 
                 One can set the default styles with the public 
                 set_default_styles(styles) function
                 **styles** dictionary is updated to at least color 
        
        **kwargs : additional kwargs to plot function, concatenate to style 
        
        Output
        ------
        axes : matplotlib axes
        """ 
        return plot_fit_result(self, *args, **kwargs)
    
    def plot_model(self, *args, **kwargs):
        """ convenient function make a model from result of a fit and plot 
        
        
        fit_result.plot_model(dates, **kwargs)
        is equivalent to 
        fit_result.make_model(dates).plot(**kwargs)
        """
        return plot_model(self,  *args, **kwargs)
    
    def make_model(self, dates=None):
        """ make a TimeSeries data from dates
        
        Inputs
        ------
        dates: optional, None, 2xtuple of min,max range, list of dates
                default is None: min,max is taken from 'strat', 'end' columns of fit_result
        
        Output
        ------
        TimeSeries (Series like)
             Time series data like the John H ones
        """
        return make_model(self, dates)

###############################################################################
#                    ____  ____  _____     ___  _____ _____                   #
#                   |  _ \|  _ \|_ _\ \   / / \|_   _| ____|                  #
#                   | |_) | |_) || | \ \ / / _ \ | | |  _|                    #
#                   |  __/|  _ < | |  \ V / ___ \| | | |___                   #
#                   |_|   |_| \_\___|  \_/_/   \_\_| |_____|                  #
#                                                                             #
###############################################################################    

def _run_operator(data, op, axis, name, kwargs):
    if not isinstance(data, TimeFrame):
        return op(data, axis=axis, **kwargs)
    if axis!=0 :
        return op(DataFrame(get_cases(data)), axis=axis, **kwargs)
    headers = get_header(data)
    cases = get_cases(data)
    index = cases.columns
    collapsed_cases = op(np.asarray(cases), axis=0, **kwargs)
    return _collaps_op(headers, collapsed_cases, index, name)

def _collaps_op(headers, collapsed_cases, index, name):
    countries =  set(headers[COUNTRY])
    if len(countries)>1:
        country = "_mixed_"
    else:
        country, = countries
    
    states = set(headers[STATE])
    if len(states)>1:
        state = "_mixed_"
    else:
        state, = states
    
    if name is None:
        if state is not "_mixed_" and state==state:
            name = state 
        elif country is not "_mixed_":
            name = "All "+country 
    
        
    return TimeSeries(
    [state, country]+list(collapsed_cases), 
    index = [STATE,COUNTRY]+list(index), 
    name=name    
    )
    

def _reindex_data(data, data_col=4):
    # reindex to state name or country name if state is NaN
    # note a==a return False if a is nan
    data.index = np.where(data[PROVINCE] == data[PROVINCE], data[PROVINCE], data[COUNTRY])
    # "header" column slice and data (time seri) slice
    HSLICE, DSLICE = slice(0,data_col), slice(data_col, None)
    # convert the string columns to date object 
    data.columns = list(data.columns[HSLICE])+list(_strdate2date(data.columns[DSLICE]))
    
    # some states can have name of country which would make duplicate 
    # for those we replace the state by "state country" to avoid duplicate keys
    duplicates = np.array([ isinstance(data.loc[n], DataFrame) for n,_ in data.iterrows()]) 
    new_index = np.where( duplicates & (data[PROVINCE]==data[PROVINCE]),  data[PROVINCE]+" "+data[COUNTRY], data.index )
    data.index = new_index
    
    return TimeFrame(data)

def _load_one_data(url):
    return _reindex_data(read_csv(url))

                
def _strdate2date(strdates):
    """ convert str JH mm/dd/yy dates format to datetime.date """
    # assume all date are iddentical and return as is 
    if not len(strdates):  return  np.array([])
    
    if isinstance(strdates[0], DATA_COL_TYPE):
        return strdates
    
    out = []
    for d in strdates:
        m,d,y = d.split('/')
        out.append(date(int(y)+2000,int(m),int(d)))
    return np.array(out)

def _date2strdate(dates):
    """ convert a list of datetime.date to str JH mm/dd/yy dates format """    
    return np.array( ['{month}/{day}/{year}'.format( month=d.month, day=d.day, year=d.year-2000) for d in dates])

def _subset_by_countries(data, countries):
    """ subset the data from country name or list """
    if countries is None:
        return data
    
    if isinstance(countries, str):
        return data.loc[data[COUNTRY]==countries]
    
    t = None
    for country in countries:
        if t is None: 
            t = data[COUNTRY]==country
        else:
            t |= data[COUNTRY]==country
    return data.loc[t]    

def _subset_by_states(data, states):
    """ subset the data from state name or list """

    if states is None:
        return data
    
    if isinstance(states, str):
        return data.loc[data[PROVINCE]==states]
    
    t = None
    for state in states:
        if t is None: 
            t = data[PROVINCE]==state
        else:
            t |= data[PROVINCE]==state
    return data.loc[t]  

def _subset_by_names(data, names):
    if names is None:
        return data
    return data.loc[names]

def _parse_date(d):
    if isinstance(d, str):
        return date.fromisoformat(d)
    return d

def _parse_date_arg(arg):
    if isinstance(arg, (list,)):
        indexes = []
        data = []
        for name, d in arg:
            indexes.append(name)
            data.append([_parse_date(d)])
        return DataFrame( data, index=indexes, columns=['date'])
    if isinstance(arg, str):
        return _parse_date(arg)
    return arg
    
def _parse_intervals_arg(arg):
    if isinstance(arg, (list,)):        
        data = []
        for name, start, end in arg:            
            data.append( [name, _parse_date(start),  _parse_date(end)])
        return DataFrame(data, columns=['name', 'start', 'end'])
    return arg

def _concatenate(lst):
    """ use only if every items is a Series """
    if not len(lst):
        return _DataSet()  
    constructor = lst[0]._constructor_expanddim
    return constructor(concat(lst, axis=1).T)

    
def _compute_date_range(data, start=None, end=None, ndays=None):   
    
    start = _parse_date_arg(start)
    end = _parse_date_arg(end)
    
    if is_day_indexed(data):
        if start is None:
            start = _min(get_dates(data))
        if ndays is not None:
            end = start+ndays-1
        elif end is None:
            end = _max(get_dates(data)) 
        return start, end
            
        
     
    if isinstance(start, DataFrame):
        if not isinstance(data, Series):
            raise ValueError("start can be a DataFrame only if data is a Series")
        start = start.loc[data.name]['date']
    if isinstance(end, DataFrame):
        if not isinstance(data, Series):
            raise ValueError("end can be a DataFrame only if data is a Series")
        end = end.loc[data.name]['date']
    
    if start is None:
        start= _min(get_dates(data))
    if end is None:
        end = _max(get_dates(data)) 
    
    if isinstance(end, np.int):
        if (end)>=0:
            end = _min(get_dates(data)) + timedelta(days=end)
        else:
            end = _max(get_dates(data)) + timedelta(days=end+1)        
    else:
        end = _parse_date(end)
    
    if isinstance(start, np.int):
        if start < 0:
            start = _max(get_dates(data)) + timedelta(days=start+1)
        else:
            start = _min(get_dates(data)) + timedelta(days=start)        
    else:
        start = _parse_date(start)
    
    if ndays is not None:
        end = start + timedelta(days=ndays-1)
    
    if start > end:
        raise ValueError('start > end')
    return start, end
    
def _get_subdata_range(data, start=None, end=None, ndays=None):
                    
    start, end = _compute_date_range(data, start=start, end=end, ndays=ndays)
    
    d = get_dates(data) 
    if start is None: start = _min(d)
    if end is None: end = _max(d)
       
    t = (d>=start) & (d<=end)
    
    hslice, dslice = _header_cases_slices(data)
    if isinstance(data, Series):        
        index = list(range(hslice.start, hslice.stop))
        index.extend( np.arange(len(t))[t]+dslice.start )
        return data[index]
    else:
        cols = list(data.columns[hslice])
        cols.extend(data.columns[dslice][t]) 
        return data[cols]
    
def _header_cases_slices(data):
    itr = data.index if isinstance(data, Series) else data.columns
    for i,v in enumerate(itr):
        if isinstance(v, (date, int, np.int64, np.int32)):
            return slice(0,i), slice(i, None)
    return slice(0, len(data.index)), slice(len(data.index), None)
    #raise ValueError("Could not found time seri data in data frame")




def _get_days(dates, since=None, name=None):
    since = _parse_date(since)
    if since is None:        
        since = _min(dates)
    
    elif isinstance(since, np.integer): 
        # this is an index
        since = dates[since]
    days = [(d-since).days for d in dates]
    return days
    #return Series(days, index=dates, name=name+" "+str(since))

def _reordered(data):
    if isinstance(data, Series):
        cls = data.index
    else:
        cls = data.columns
    
    hcol = []    
    dcol = []
    for c in cls:
        if isinstance(c, DATA_COL_TYPE):
            dcol.append(c)
        else:
            hcol.append(c)    
    dcol.sort()
    cols = hcol+dcol
    return data[cols]
    
def _itercases(data):
    """ iter over DataFrame rows 
    
    each iterations are returning (header, data)
    where header is the country information part (Province/State Country/Region Lat Long)
    and data is the time serie
    """
    hslice, dslice = _header_cases_slices(data)
    for _, sdata in data.iterrows():
        yield sdata[hslice], sdata[dslice]
  
def _dates2ticks(dates, step=None):
    """ convert dates to stick label for plotting """
    if isinstance(dates, tuple):
        dates = _gen_dates(*dates)              
    
    if step is None:
        amp = (_max(dates) - _min(dates)).days
        step = _max(1, amp//10)
        
    ticks = []
    pos_dates = []
    first = True
    for date in dates:
        if date.day==1 or first:
            ticks.append(date.strftime('%y-%m-%d'))
            pos_dates.append(date)                    
        else:
            if not date.day%step:
                ticks.append("%d"%date.day)
                pos_dates.append(date)
        first = False
    return pos_dates, ticks    

#####
# fit function 
#####
def _efit(days, cases):
    y = np.log(cases.astype(float))
    pol = np.polyfit(days,y,1)
    T = 1./pol[0]
    A = np.exp(pol[1])
    return A, T

def _sfit(days, cases):    
    y = np.log2(cases.astype(float))
    pol = np.polyfit(days,y,1)
    T = 1./pol[0]
    A = 2**pol[1]
    return A, T


def _gen_dates(mindate, maxdate):
    """ Generate a date list borned by mindate and maxdate 
    
    Inputs
    ------
    mindate, maxdate : yyyy-mm-dd str or datetime.date 
    
    Output
    ------ 
    list of datetime.date object 
    """
    mindate, maxdate = (_parse_date(d) for d in (mindate, maxdate))
    dates = []
    for i in range((maxdate-mindate).days+1):
        dates.append( mindate + timedelta(days=i) )
    return dates

def _model_date_to_days(dates, start):
    if start is None:
        start = _min(dates)
    if isinstance(dates[0], date):
        days = np.array([(d-start).days for d in dates])
    else:
        days = np.asarray(dates)-start
    return days

def _efit_model(dates, A, T, start=None):        
    days = _model_date_to_days(dates, start)
    return A*np.exp(days/T)

def _sfit_model(dates, A, T, start=None):            
    days = _model_date_to_days(dates, start)
    return A*2**(days/T)

def _get_fit_model_func(ftype):
    if ftype is '2':
        return _sfit_model        
    if ftype is 'exp':
        return _efit_model
    raise ValueError("unknonw fits type %r"%ftype)
    



def _update_styles(ax, name, styles):
    if not ax.lines:
        return 
    st = styles.setdefault(name, {})
    st.setdefault('color', ax.lines[-1].get_color())

def _parse_plot_args(data, axes, style, styles):
    
    if axes is None:
        axes = plt.axes()    
    
    styles = _default_styles if styles is None else styles
    style = style or {}
        
    if isinstance(data, Series):
        iterator = [(data.name, data)]
    else:
        iterator = data.iterrows()
    
    return iterator, axes, style, styles

def _get_row(name, data):
    """ get row from name from a DataFrame or Series """
    if isinstance(data, Series):
        if data.name and data.name!=name:
            raise ValueError("Series %r does not bellong to data row %r"(data.name, name))
        return data
    else:
        return  data.loc[name]

def _parse_log(log, value):
    value = np.asarray(value)
    if log==2:
        return np.log2(value.astype(float))
    if log==10:
        return np.log10(value.astype(float))
    if log==True:
        return np.log(value.astype(float))
    if not log:    
        return value
    raise ValueError('log must be 2, 10, True or False got %s'%log)

def _labeler(data):
    if is_day_indexed(data):
        if is_model(data):
            if "day_zero" in data:
                return lambda r:"{r.name} {r[day_zero]} {r[start]} -> {r[end]} T={r[T]:.2f}".format(r=r)
            else:
                return lambda r:"{r.name} {r[start]} -> {r[end]} T={r[T]:.2f}".format(r=r)                
        else:
            if "day_zero" in data:
                return lambda r:"{r.name} {r[day_zero]}".format(r=r)
            else:
                return lambda r:"{r.name}".format(r=r)
    else:
        if is_model(data):
            return lambda r:"{r.name} {r[start]} -> {r[end]} T={r[T]:.2f}".format(r=r)
        else:
            return lambda r:"{r.name}".format(r=r)

def _styler(name, style, styles):
    return dict(style, **styles.get(name, style))

_lsts = ["solid", "dotted", "dashed", "dashdot", (0, (1, 10)), (0, (5, 10)), (0, (3, 10, 1, 10))][::-1]
def _next_linestyle(name, axes):
    try:
        linestyle = axes._linestyles
    except AttributeError:
        linestyle = {}
        axes._linestyles = linestyle
        
    try:
        lsts = linestyle[name]
    except KeyError:
        lsts = list(_lsts)
        linestyle[name]  = lsts
    if not lsts:
        return "solid"
    return lsts.pop()
    
####
#Garbage 

def _fit_model(dates, result):
    """ Return the model value of fits from dates 
    
    dates :  array like dates
    result:  result of a fit 
    """    
                            
    if len(result.shape) == 1:
        r = result
        return Series(_get_fit_model_func(r['ftype'])(dates, r['A'], r['T'], r['start']), index=dates)
    else:
        return DataFrame(
        [_get_fit_model_func(r['ftype'])(dates, r['A'], r['T'], r['start']) for _,r in result.iterrows()], 
        columns=dates
        )

if __name__=="__main__":
    c,d,r = load_data() 
        
    origin = c.when_case_exceed(200)  
    print( get_day_indexed(c.loc['France'], day_zero=origin))
    
    print(c.loc[['France','Italy','Hubei']].header)
    print( get_day_indexed(c.loc[['France','Italy','Hubei']], day_zero=origin))
    
    #print(get_days(c.iloc[154], origin))
    #print(get_days(c.iloc[158:160], origin))
    
    