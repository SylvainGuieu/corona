from coronapy import load_data, set_default_styles
import  matplotlib 
from matplotlib.pylab import plt
import os 
import sys 

matplotlib.use('Agg')

if len(sys.argv) > 1:
    root = sys.argv[1]
else:
    root = "./img"

matplotlib.rcParams['figure.figsize'] = 12, 6
styles = {"France":{"color":"blue"}, 
          "Italy":{"color":"green"}, 
          "Spain":{"color":"red"}, 
          "Germany":{"color":"black"} }
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

names1 = ['France', 'Italy', 'Spain', 'Hubei']
names2 = ['France', 'Italy', 'Spain']

plt.figure()
subset = confirmed.subset(names1, start="2020-02-20")
axes = subset.plot()
axes.set(yscale="log", ylabel='Confirmed cases', xlabel="Date")
save('confirmed', axes.figure)


plt.figure()
origins = confirmed.when_case_exceed(200) # the dates from when at least 100 case detected
subset = confirmed.subset(names1)
axes = subset.get_day_indexed(day_zero=origins).subset(start=-10, end=50).plot()
axes.set(yscale="log", ylabel="Confirmed case", xlabel="Days since at least 200 cases")
save('confirmed_days_200', axes.figure)


plt.figure()
origins = death.when_case_exceed(20)
data = death.subset(names1).get_day_indexed(origins)
data = data.subset(start=0, ndays=50) # 30 days max 
axes = data.plot()
axes.set(xlabel='Days since at least 20 Cases', ylabel='Number of Death', yscale="log")
save('death_days_20', axes.figure)


plt.figure()
subset = death.subset(names2, start="2020-02-20")
fit_result = subset.subset(start=-10).fit('2') # fit the last 10 days

axes = subset.plot()
fit_result.plot_model(("2020-03-15", "2020-04-01"), axes=axes)
axes.set(yscale='log', xlabel="Date", ylabel="Death", title = "Death cases = A $2^{t/T}$")
axes.grid( which='both')
save('death_fit', axes.figure)

plt.figure()
subset = death.subset(names1)
s, e, c = 6,13, 20
origin = death.when_case_exceed(c)
subset = subset.get_day_indexed(origin).subset(start=0, end=30)

fit_result = subset.subset(start=s, end=e).fit('2') # fit the last 10 days
axes = subset.plot()
fit_result.plot_model( (s, 25), axes=axes)
axes.set(yscale='log', xlabel="Days since %s cases"%c, ylabel="Death", title = "Death = A $2^{t/T}$ Fit between day %s and day %s since %s cases"%(s,e,c))
axes.grid( which='both')
[axes.axvline(x, color='k', linestyle=":") for x in (s,e)]
save('death_fit_days', axes.figure)



plt.figure()
subset = confirmed.subset(names1)
origin = confirmed.when_case_exceed(200)

data = subset.get_day_indexed(origin).subset(start=0, end=30)

# intervals of 6 days every day 
# the mindays keyword assure that their is always 6 points per sample (6 full days)
intervals = data.intervals(window=6, step=1, mindays=6)
result = data.fit_intervals(intervals)
# first argument datekey can be 'start', 'end', 'center' define which date to plot
axes = result.plot(datekey='end', style={'marker':'+'})
axes.set(ylim=(1.5,8), xlabel="Days since 200 cases", ylabel="Doubling Time T in days", 
           title="From confirmed case window=%d days, step=1 days"%(6)) 
save('confirmed_T', axes.figure)

plt.figure()
subset = death.subset(names1)
origin = death.when_case_exceed(20)
data = subset.get_day_indexed(origin).subset(start=0, end=30)

intervals = data.intervals(window=6, step=1, mindays=6)
result = data.fit_intervals(intervals)
# first argument datekey can be 'start', 'end', 'center' define which date to plot
axes = result.plot(datekey='end', style={'marker':'+'})
axes.set(ylim=(1,8), xlabel="Days since 20 cases", ylabel="Doubling Time T in days", 
           title="From death case window=%d days, step=1 days"%(6))  
save('death_T', axes.figure)


plt.figure()
numerator = death.subset(names2, start="2020-03-01")
denominator = confirmed.subset( start="2020-03-01") # denominator only requiere to have the same number of dates
axes = numerator.plot_proportion(denominator)
axes.set(ylabel="Death / Confirmed [%]", xlabel="Date")
save('death_ratio', axes.figure)

