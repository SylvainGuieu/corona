from coronapy import load_data, set_default_styles
import  matplotlib 
from matplotlib.pylab import plt
import numpy as np 
import os 
import sys 

matplotlib.use('Agg')
plt_list = list(range(20))
plt_list = [2]

if len(sys.argv) > 1:
    root = sys.argv[1]
else:
    root = "./tmp"

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
names1 = ['France', 'Italy', 'Spain', 'Hubei', 'US', 'Korea, South']

mdates = range(2, 40)

if 1 in plt_list:
    plt.figure()
    subset = death.subset(names1)
    origin = death.when_case_exceed(20)
    subset = subset.get_day_indexed(origin).subset(start=0)
    perday = subset.cases.iloc[:,1:] - np.asarray(subset.cases.iloc[:,0:-1])
    perday = perday.rolling(4, center=True, axis=1).mean()
    total  = subset.cases.iloc[:,1:]
    
    axes = plt.gca()
    for mdate in mdates:
        axes.clear()
        pd1 = perday.subset(end=mdate)
        ttl = total.subset(end=mdate)
        for name, r in pd1.iterrows():
            axes.plot(ttl.loc[name], r, label=name, **styles.get(name,{}))
            axes.plot(ttl.loc[name].iloc[-1], r.iloc[-1], 'k*')        
        axes.legend(loc='upper left')
        axes.set(yscale="log", xscale="log", 
            xlabel="Totala Death", ylabel="Daily Death", 
            xlim=(20,30000), ylim=(20,1000)
            )
        save('daily_total_day_%03d'%mdate)    

    #convert -delay 25 tmp/daily_total_day*.png img/daily_total_day.gif

if 2 in plt_list:
    plt.figure()
    subset = death.subset(names1)
    
    perday = subset.cases.iloc[:,1:] - np.asarray(subset.cases.iloc[:,0:-1])
    perday = perday.rolling(4, center=True, axis=1).mean()
    total  = subset.cases.iloc[:,1:]
    mdates = range(2,len(total.iloc[0])+6) # add 6 iddentical graph for pause
    axes = plt.gca()
    for mdate in mdates:
        axes.clear()
        pd1 = perday.subset(end=mdate)
        ttl = total.subset(end=mdate)
        for name, r in pd1.iterrows():
            
            axes.plot(ttl.loc[name], r, label=name, **styles.get(name,{}))
            axes.plot(ttl.loc[name].iloc[-1], r.iloc[-1], 'k*')        
        axes.legend(loc='upper left')
        axes.set(yscale="log", xscale="log", 
            xlabel="Totala Death", ylabel="Daily Death", 
            xlim=(1,30000), ylim=(1,1000), 
            title=max(r.index)
        )
        save('daily_total_date_%03d'%mdate)    
    #convert -delay 25 tmp/*date_*.png img/daily_total_date.gif
            
