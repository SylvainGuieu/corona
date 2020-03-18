from pandas import  DataFrame, read_csv, concat, Series
from matplotlib.pylab import plt
from datetime import date, timedelta
import numpy as np


urls = {
'confirmed' : 'https://github.com/CSSEGISandData/COVID-19/raw/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv',
'death' : 'https://github.com/CSSEGISandData/COVID-19/raw/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Deaths.csv',
'recovered' : 'https://github.com/CSSEGISandData/COVID-19/raw/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Recovered.csv',
}


def load_data(dtags=("confirmed", "death", "recovered")):
    """ Load data from Johns Hopkins University GitHub (internet coneeection requiered)
    
    >>> confirmed = load_data("confirmed")
    >>> death = load_data("death")
    Or in one call : 
    >>> confirmed, death, recoverd = load_data("confirmed", "death", "recovered")
    >>> confirmed, death, recoverd = load_data() # load the three by default
    """
    if not dtags: dtags = ("confirmed", "death", "recovered")
    if len(dtags)>1:
        return [read_csv(urls[dt]) for dt in dtags]    
    else:
        return read_csv(urls[dtags]) 

def strdate2date(strdates):
    """ convert str dates to datetime.date """
    out = []
    for d in strdates:
        m,d,y = d.split('/')
        out.append(date(int(y)+2000,int(m),int(d)))
    return np.array(out)

def find_indexes(data, countries):
    """ find row indexes for several countries """
    if isinstance(countries, str):
        countries = [countries]
        
    indexes = []
    for country in countries:
        ou = np.where(data['Province/State']==country)[0]
        if len(ou):
            indexes.extend(ou)
        else:            
            ou = np.where(data['Country/Region']==country)[0]
            if len(ou)>1:
                ou = np.where((data['Province/State']==country) & (data['Country/Region']==country))[0]
            indexes.extend(ou)
    return indexes

def today(days=0):
    return date.today() + timedelta(days)

def _parse_date(d):
    if isinstance(d, str):
        return date.fromisoformat(d)
    return d
    

def get_subdata(data, countries=None, date_min=date(1970,1,1), date_max=date(2050,1,1)):
    """ Restric data from a list of countries and date boundaries """
    
    if date_min is None:
        date_min= date(1970,1,1)
    if date_max is None:
        date_max = date(2050,1,1)
    
    date_min, date_max = (_parse_date(d) for d in [date_min, date_max])
        
    if countries is not None:
        data = data.iloc[find_indexes(data, countries)]
    cols = list(data.columns[:4])
    d = get_dates(data)        
    t = (d>=date_min) & (d<=date_max)
    cols.extend(data.columns[4:][t])
    return data[cols]
    

def get_data(data):
    """ Return only the data part from the DataFrame """
    return data[data.columns[4:]]
def get_header(data):
    """ Return only the header part from the DataFrame 
    (Province/State Country/Region Lat Long)
    """
    return data[data.columns[:4]]
def get_dates(data):
    """ Return the dates from the DataFrame """
    return strdate2date(data.columns[4:])

def iter_rows(data):
    """ iter over DataFrame rows 
    
    each iterations are returning (header, data)
    where header is the country information part (Province/State Country/Region Lat Long)
    and data is the time seri
    """
    for _, sdata in data.iterrows():
        yield sdata[:4], sdata[4:]

def dates2ticks(dates, step=None):
    if step is None:
        amp = (max(dates) - min(dates)).days
        step = max(1, amp//10)
        
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

def efit_data(data):
    dates = get_dates(data)
    mindate = min(dates)
    days =  [(d - mindate).days for d in dates]
    results = {}
    for header, cdata in iter_rows(data):
        country = header[1] 
        y = np.log(cdata.astype(float))
        pol = np.polyfit(days,y,1)
        T = 1./pol[0]
        A = np.exp(pol[1])     
        results[country] = {
        'country':country,
        'T':T, 'A':A, 
        'rep': '$A \exp^{t/T}$', 
        'ftype':'exp',
        'start_date':mindate,
        'end_date':max(dates),
        'label': '%s T=%.2f'%(country, T)
        }
    return DataFrame.from_dict(results, orient='index')

def _efit_model(dates, A, T, start_date=None):
    
    if start_date is None:
        start_date = min(dates)
    days = np.array([(d-start_date).days for d in dates])
    return A*np.exp(days/T)

def sfit_data(data):
    """ fit A.2^(t/T) with t in days for each data rows 
    
    result is returned 
    """
    dates = get_dates(data)
    mindate = min(dates)
    days =  [(d - mindate).days for d in dates]
    results = {}    
    for header, cdata in iter_rows(data):
        country = header[1]        
        y = np.log2(cdata.astype(float))
        pol = np.polyfit(days,y,1)
        T = 1./pol[0]
        A = 2**pol[1]
        results[country] = {
        'country':country,
        'T':T, 'A':A, 'rep':'$A 2^{t/T}$', 
        'ftype':'2', 
        'start_date':mindate, 
        'end_date':max(dates),
        'label': '%s T=%.2f'%(country, T)
        }        
    return DataFrame.from_dict(results, orient='index')
    
def _sfit_model(dates, A, T, start_date=None):        
    if start_date is None:
        start_date = min(dates)
    days = np.array([(d-start_date).days for d in dates])
    return A*2**(days/T)

def get_fit_func(ftype):
    if ftype == '2':
        return sfit_data
    if ftype == 'exp':
        return efit_data
    raise ValueError('unknonw fit type %s'%ftype)

def fit_data(data, ftype='2'):
    """ from a data subset make a fit and return it in a single DataFrame
    
    data:  data (time sery DataFrame of covid-19) 
    ftype : type of fits '2' or 'exp'
    """
    fit = get_fit_func(ftype)
    return fit(data)

def get_fit_model_func(ftype):
    if ftype is '2':
        return _sfit_model        
    if ftype is 'exp':
        return _efit_model
    raise ValueError("unknonw fits type %r"%ftype)
    
def fit_model(dates, result):
    """ Return the model value of fits from dates 
    
    dates :  array like dates
    result:  result of a fit 
    """    
                            
    if len(result.shape) == 1:
        r = result
        return Series(get_fit_model_func(r['ftype'])(dates, r['A'], r['T'], r['start_date']), index=dates)
    else:
        return DataFrame(
        [get_fit_model_func(r['ftype'])(dates, r['A'], r['T'], r['start_date']) for _,r in result.iterrows()], 
        columns=dates
        )
    

def date_ranges(window=6, step=1, start_date=None, end_date=None):
    """generate a list of (date_min, date_max) boundary from a day window and day steps 
    
    window : size of the window in days default is 6 
    step  :  slidding step of the window default is 1
    start_date :  start date to considers default is "2020-01-01"
    end_date   :  end date default is date.today() 
    """
    if start_date is None:
        start_date = "2020-01-01"
    else:
        start_date = _parse_date(start_date)
    
    if end_date is None:
        end_date = date.today()
    else:
        end_date = _parse_date(end_date)
         
    d = start_date
    end_date = end_date - timedelta(days=window)
    dates = []
    while d<(end_date):
        d2 = d+timedelta(days=window) 
        dates.append( (d,d2))
        d = d + timedelta(days=step)
    return dates

def split_data_by_date(data, dates):
    return [get_subdata(data, date_min=d1, date_max=d2) for d1, d2 in dates]

def fit_severals(data_list, ftype='2'):
    """ from a list of data subset make a fit and return it in a single DataFrame
    
    data_list :  list of data (time sery DataFrame of covid-19) 
    ftype : type of fits '2' or 'exp'
    """
    results = []
    for data in data_list:
        results.append(fit_data(data, ftype))
    return concat(results)
    
def slidding_fits(data, window=6, step=1, start_date=None, end_date=None, fit_func=sfit_data):
    """ Make fits with a slidding time period defined by : 
    
    window : time window in days 
    step : incremental step of windows 
    start_date : starting date (default is the min date of data)
    end_date : en date (default is the max date of data) 
    fit_func : the fit function, default is `sfit_data` function must take one single 
               argument, the data
    """
    if start_date is None:
        d = get_data(data)
        start_date = min(d)
    if end_date is None:
        d = get_data(data)
        end_date = max(d)
    fits = []
    d = start_date
    while d<(end_date):
        d2 = d+timedelta(days=window)
        fits.append(fit_func(get_subdata(data, date_min=d, date_max=d2)))
        d = d + timedelta(days=step)
    return concat(fits)


def set_date_xticks(dates, ax=None, **kwargs):
    kwargs.setdefault('rotation', 75)
    if ax is None:
        ax = plt.gca()
    ticks, labels = dates2ticks(dates)
    ax.set_xticks(ticks)
    labels = ax.set_xticklabels(labels, **kwargs)        
    for l in labels:
        l.update(kwargs)


def plot_data(data, title="", log=False, ylabel="", fit_result=None, ax=None, styles=None, style=None):
    """ Plot data for each countries """
    plt.figure()
    if ax is None:
        ax = plt.axes()
    styles = styles or {}
    style = style or {}
    dates = get_dates(data)
    
    colors = {}
    linestyles = {}
    lsts = ["dotted", "dashed", "dashdot", "losely dotted", "losely dashed", "losely dashdotted"][::-1]
    for header, sdata in iter_rows(data):
        st = dict(styles.get(header[1], style))
        st.setdefault('label', header[1])                        
        ax.plot(dates, sdata, **st)
        
        colors[header[1]] = ax.lines[-1].get_color()
        linestyles[header[1]] = list(lsts)
    
    if fit_result is not None:
        if isinstance(fit_result, DataFrame):
            iterator = fit_result.iterrows
        else:
            iterator = fit_result.items    
        for country, r in iterator():
            y = fit_model(dates, r)
            
            lst = linestyles.get(country, ['dotted'])
            if not lst:
                lst = "dotted"
            else:
                lst = lst.pop()              
            ax.plot(dates, y,  color=colors.get(country, None), linestyle=lst, label="%s T=%.2f"%(country, r['T']))
        
    ax.legend()
    ax.set_xlabel("date")    
    ax.xaxis.set_tick_params(rotation=75)
    
    ticks, labels = dates2ticks(dates)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if log:
        ax.set_yscale('log')        
    return ax
        
def plot_proportion(numerator, denominator, title="", ylabel="", ax=None, styles=None, style=None):
    
    plt.figure()
    if ax is None:
        ax = plt.axes()
    
    styles = styles or {}
    style = style or {}
    dates = get_dates(numerator)
    
    for (header, n), (_, d) in zip(iter_rows(numerator), iter_rows(denominator)):
        st = dict(styles.get(header[1], style))
        st.setdefault('label', header[1])                     
        ax.plot(dates, n/d*100, **st)
    ax.legend()
    ax.set_xlabel("date") 
       
    ticks, labels = dates2ticks(dates)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.xaxis.set_tick_params(rotation=75)
    
    
    ax.set_ylabel(ylabel)
    return ax
    
if False:#__name__=="__main__":
    rec1 = dict(countries = ["France", "Italy", "Spain", "Germany", "Japan"],
                date_min=date(2020,2, 20),
                ) 
    
    confirmed, death, recovered = load_data()
    c1, d1, r1 = (get_subdata(data, **rec1) for data in [confirmed, death, recovered])
    
    plot_data(c1, title="Confirmed", log=True, ylabel="# Confirmed", fit=efit_data(c1))
    plot_proportion(d1, c1, ylabel="Death/Confirmed %")
    #plot_case(subdata(depth, ["France", "Italy", "Spain", "Japan"]), title="Depth")
    #plot_case(subdata(depth, ["France", "Italy", "Spain", "Japan"]), title="Recovered")
    plt.show()
    print(efit_data(c1))
