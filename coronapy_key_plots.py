from coronapy import load_data, set_default_styles
import  matplotlib 
from matplotlib.pylab import plt
import numpy as np 
import os 
import sys 

matplotlib.use('Agg')
plt_list = list(range(20))
#plt_list = [9]

if len(sys.argv) > 1:
    root = sys.argv[1]
else:
    root = "./img"

matplotlib.rcParams['figure.figsize'] = 12, 6 
matplotlib.rcParams['axes.grid'] = True
matplotlib.rcParams['axes.grid.which'] = 'both'
matplotlib.rcParams['grid.alpha'] = 0.3

         
styles = {"France":{"color":"blue"}, 
          "Italy":{"color":"green"}, 
          "Spain":{"color":"red"}, 
          "Germany":{"color":"black"}, 
          "Hubei":{"color":"orange"}, 
          "US":{"color":"cyan"}
         }
set_default_styles(styles) # styles will be default for all graph

def save(name, fig=None):
    fig = plt.gcf() if fig is None else fig 
    path = os.path.join(root,name+'.png')
    fig.savefig(path)
    print(path)


confirmed, death = load_data(('confirmed', 'death'))
patch = [
    ('France', '2020-03-17', 173),
    ('France', '2020-03-18', 244),
    ('France', '2020-03-19', 372)
]
death.patch(patch)

names1 = ['France', 'Italy', 'Spain', 'Hubei', 'US']
names2 = ['France', 'Italy', 'Spain', 'US']



if 1 in plt_list:
    plt.figure()
    subset = confirmed.subset(names1, start="2020-02-20")
    axes = subset.plot()
    axes.set(yscale="log", ylabel='Confirmed cases', xlabel="Date")
    save('confirmed', axes.figure)

if 2 in plt_list:
    plt.figure()
    origins = confirmed.when_case_exceed(200) # the dates from when at least 100 case detected
    subset = confirmed.subset(names1)
    axes = subset.get_day_indexed(day_zero=origins).subset(start=-10, end=50).plot()
    axes.set(yscale="log", ylabel="Confirmed case", xlabel="Days since at least 200 cases")
    save('confirmed_days_200', axes.figure)

if 3 in plt_list:
    plt.figure()
    origins = death.when_case_exceed(20)
    data = death.subset(names1).get_day_indexed(origins)
    data = data.subset(start=0, ndays=50) # 30 days max 
    axes = data.plot()
    axes.set(xlabel='Days since at least 20 Cases', ylabel='Number of Death', yscale="log")
    save('death_days_20', axes.figure)

if 4 in plt_list:
    
    plt.figure()
    subset = death.subset(names2, start="2020-02-20")
    fit_result = subset.subset(start=-10).fit('2') # fit the last 10 days
    
    axes = subset.plot()
    fit_result.plot_model((fit_result.loc['France','start'], "2020-04-10"), axes=axes)
    axes.set(yscale='log', xlabel="Date", ylabel="Death", title = "Death cases = A $2^{t/T}$")
    save('death_fit', axes.figure)

if 5 in plt_list:
    plt.figure()
    subset = death.subset(names1)
    s, e, c = 6,13, 20
    origin = death.when_case_exceed(c)
    subset = subset.get_day_indexed(origin).subset(start=0, end=30)
    
    fit_result = subset.subset(start=s, end=e).fit('2') # fit the last 10 days
    axes = subset.plot()
    fit_result.plot_model( (s, 25), axes=axes)
    axes.set(yscale='log', xlabel="Days since %s cases"%c, ylabel="Death", title = "Death = A $2^{t/T}$ Fit between day %s and day %s since %s cases"%(s,e,c))
    [axes.axvline(x, color='k', linestyle=":") for x in (s,e)]
    save('death_fit_days', axes.figure)


if 6 in plt_list:
    plt.figure()
    subset = confirmed.subset(names1)
    origin = confirmed.when_case_exceed(200)
    
    data = subset.get_day_indexed(origin).subset(start=0, end=40)
    
    # intervals of 6 days every day 
    # the mindays keyword assure that their is always 6 points per sample (6 full days)
    intervals = data.intervals(window=6, step=1, mindays=6)
    result = data.fit_intervals(intervals)
    # first argument datekey can be 'start', 'end', 'center' define which date to plot
    axes = result.plot(datekey='end', style={'marker':'+'})
    axes.set(ylim=(1.5,20), xlabel="Days since 200 cases", ylabel="Doubling Time T in days", 
    title="From confirmed case window=%d days, step=1 days"%(6)) 
    save('confirmed_T', axes.figure)

if 7 in plt_list:
    plt.figure()
    subset = death.subset(names1)
    origin = death.when_case_exceed(20)
    data = subset.get_day_indexed(origin).subset(start=0, end=40)
    
    intervals = data.intervals(window=6, step=1, mindays=6)
    result = data.fit_intervals(intervals)
    # first argument datekey can be 'start', 'end', 'center' define which date to plot
    axes = result.plot(datekey='end', style={'marker':'+'})
    axes.set(ylim=(1,20), xlabel="Days since 20 cases", ylabel="Doubling Time T in days", 
    title="From death case window=%d days, step=1 days"%(6))  
    save('death_T', axes.figure)

if 8 in plt_list:
    plt.figure()
    numerator = death.subset(names2, start="2020-03-01")
    denominator = confirmed.subset( start="2020-03-01") # denominator only requiere to have the same number of dates
    axes = numerator.plot_proportion(denominator, scale=100)
    axes.set(ylabel="Death / Confirmed [%]", xlabel="Date")
    save('death_ratio', axes.figure)

if 9 in plt_list:
    plt.figure()
    origins = death.when_case_exceed(20)
    subset = death.subset(names1)
    # index by days and keep day from -10 to 50 
    subset = subset.get_day_indexed(origins).subset(start=-10, end=60) # .cases is necessary to extract only data 
    # np.asarray (or .to_numpy) is necessary to avoid that pandas is aligning data
    sd = subset.cases.iloc[:,1:] - np.asarray(subset.cases.iloc[:,0:-1])  # to_numpy 
    sd2 = sd.copy() # for smoothed data 
    
    ws = 7 # window size
    # window type
    wt = "triang" # None "bartlett" "blackman" "triang" "boxcar" "hamming" "parzen" "nuttall" "barthann"
    
    rolling = sd.cases.rolling(ws, win_type=wt, center=True, axis=1)
    w = np.ones((ws,)) if wt is None else rolling._get_window(win_type=wt)
    
    # put back the header part for legend 
    sd = subset.header.join(sd)
    
    axes = sd.plot(linestyle="None", marker="+")
    rolling.mean().plot(label=None,linestyle="solid").set(ylabel="Daily Death", xlabel="Day since 20 cases", title="Convolution of a %r  %d days window"%(wt,ws))
    _ = plt.axes((0.16, 0.7, 0.1, 0.1), title="convol shape").plot(w, 'k-')
    save('death_daily', axes.figure)
