from pandas import  DataFrame, read_csv
from matplotlib.pylab import plt
from datetime import date
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
        ou = np.where(data['Country/Region']==country)[0]
        if len(ou)>1:
            ou = np.where((data['Province/State']==country) & (data['Country/Region']==country))[0]
        indexes.extend(ou)
    return indexes

def get_subdata(data, countries=None, date_min=date(1970,1,1), date_max=date(2050,1,1)):
    """ Restric data from a list of countries and date boundaries """
    
    date_min, date_max = ((date.fromisoformat(d) if isinstance(d, str) else d) for d in [date_min, date_max])
        
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
        'rep':'$%.2f \exp^{t/%.2f}$'%(A,T), 
        'type':'exp',
        'start_date':mindate,
        'end_date':max(dates),
        'label': '%s T=%.2f'%(country, T)
        }
    return DataFrame.from_dict(results, orient='index')

def efit_model(dates, A, T, start_date=None):
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
        'T':T, 'A':A, 'rep':'$%.2f 2^{t/%.2f}$'%(A,T), 
        'type':'2', 
        'start_date':mindate, 
        'end_date':max(dates),
        'label': '%s T=%.2f'%(country, T)
        }        
    return DataFrame.from_dict(results, orient='index')
    
def sfit_model(dates, A, T, start_date=None):        
    if start_date is None:
        start_date = min(dates)
    days = np.array([(d-start_date).days for d in dates])
    return A*2**(days/T)

def fit_model(dates, r):    
    A, T, start_date, mtype = (r[k] for k in ['A','T','start_date', 'type'])
    if mtype is '2':
        return sfit_model(dates, A, T, start_date)
    if mtype is 'exp':
        return efit_model(dates, A, T, start_date)
    raise ValueError('Bug unknown fit type')

def set_xticks(ax, ticks, labels):
    
    locs = ax.set_xticks(ticks)
    labels = ax.set_xticklabels(labels, **kwargs)
    for l in labels:
        l.update(kwargs)


def plot_data(data, title="", log=False, ylabel="", fit=None, ax=None):
    """ Plot data for each countries """
    plt.figure()
    if ax is None:
        ax = plt.axes()
    
    dates = get_dates(data)
    
    colors = {}
    linestyles = {}
    lsts = ["dotted", "dashed", "dashdot", "losely dotted", "losely dashed", "losely dashdotted"][::-1]
    for header, sdata in iter_rows(data):                
        ax.plot(dates, sdata, label=header[1])
        colors[header[1]] = ax.lines[-1].get_color()
        linestyles[header[1]] = list(lsts)
    
    if fit is not None:
        if isinstance(fit, DataFrame):
            iterator = fit.iterrows
        else:
            iterator = fit.items    
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
        
def plot_proportion(numerator, denominator, title="", ylabel="", ax=None):
    
    plt.figure()
    if ax is None:
        ax = plt.axes()
    
    dates = get_dates(numerator)
    
    for (header, n), (_, d) in zip(iter_rows(numerator), iter_rows(denominator)):                
        ax.plot(dates, n/d*100, label=header[1])
    ax.legend()
    ax.set_xlabel("date") 
       
    ticks, labels = dates2ticks(dates)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.xaxis.set_tick_params(rotation=75)
    
    
    ax.set_ylabel(ylabel)
    return ax
    
if __name__=="__main__":
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
