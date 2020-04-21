""" need ffmpeg installed """
from coronapy import load_data, set_default_styles
import  matplotlib 
from matplotlib.pylab import plt
import matplotlib.animation as manimation

import numpy as np 
import os 
import sys 
import subprocess


matplotlib.use('Agg')
plt_list = list(range(20))
#plt_list = [3]

if len(sys.argv) > 1:
    root = sys.argv[1]
else:
    root = "./img"
dpi = 100
matplotlib.rcParams['figure.figsize'] = 1920/dpi, 1080/dpi
matplotlib.rcParams['figure.dpi'] = dpi
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
names1 = ['France', 'Italy', 'Spain', 'Hubei', 'US', 'Korea, South', 'Japan', 'United Kingdom']

mdates = range(2, 46)


def anim_daily_total(data, file_name, kind="cases", minlim=1):
    fig = plt.figure()    
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Corona virus temp evolution', artist='Sylvain Guieu',
                    comment='https://github.com/SylvainGuieu/corona')
    writer = FFMpegWriter(fps=6, metadata=metadata)
    
    subset = data
    
    perday = subset.cases.iloc[:,1:] - np.asarray(subset.cases.iloc[:,0:-1])
    perday = perday.rolling(4, center=True, axis=1).mean()
    total  = subset.cases.iloc[:,1:]
    mdates = range(2,len(total.iloc[0])+1) # add 1 iddentical graph for pause
    axes = plt.gca()
    xlm = np.nanmax(np.asarray(total), axis=None)*2
    ylm = np.nanmax(np.asarray(perday), axis=None)*2
    print(xlm)
    with writer.saving(fig, os.path.join(root, file_name)+".mp4", len(mdates)):
        for mdate in mdates:
            axes.clear()
            pd1 = perday.subset(end=mdate)
            ttl = total.subset(end=mdate)
            for name, r in pd1.iterrows():                
                axes.plot(ttl.loc[name], r, label=name, **styles.get(name,{}))
                axes.plot(ttl.loc[name].iloc[-1], r.iloc[-1], 'k*')        
            axes.legend(loc='upper left')
            axes.set(yscale="log", xscale="log", 
                xlabel="Total %s"%kind, ylabel="Daily %s"%kind, 
                xlim=(minlim,xlm), ylim=(minlim,ylm), 
                title=r.index[-1]
            )
            writer.grab_frame()
            print(".", end="")
    print('')
    subprocess.run(f"ffmpeg -y -i {root}/{file_name}.mp4 -vf \"fps=10,scale=800:-1:flags=lanczos\" -loop 999 {root}/{file_name}.gif", shell=True)





if 1 in plt_list:
    anim_daily_total(death.subset(names1), "daily_total_date", "Death")

if 2 in plt_list:
    anim_daily_total(confirmed.subset(names1), "daily_total_date_confirmed", kind="Confirmed", minlim=10)

if 3 in plt_list:
    #fig = plt.figure()    
    #axes = plt.gca()
    fig, axs = plt.subplots(1,2)
    axes = axs[0]
    axes2 = axs[1]
    
    file_name = "death_over_total"
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Corona virus temp evolution', artist='Sylvain Guieu',
                    comment='https://github.com/SylvainGuieu/corona')
    writer = FFMpegWriter(fps=2, metadata=metadata)
    
    # from coronapy import _default_styles
    subset = death.subset([ 'Italy', 'Spain', 'France', 'Hubei', 'US', 'Japan','Korea, South'])
    
    #origin = subset.when_case_exceed(5)s    
    origin = death.when_case_exceed(5)
    #subset = subset.get_day_indexed(origin).subset(start=0)
    perday = subset.cases.iloc[:,1:] - np.asarray(subset.cases.iloc[:,0:-1])
    perday = perday.rolling(7, center=True, axis=1).mean()
    total = subset.cases.iloc[:,1:]
        
    nfit = 10
    nfitcond = 2
    
    cont = True
    i=1
    
    days = {}
    xzeros = {}
    
    with writer.saving(fig, os.path.join(root, file_name)+".mp4", len(total.iloc[0,:])):
        while cont:
            cont = False 
            i += 1
            axes.clear()
            axes2.clear()
            
            for name, r in perday.iterrows():                
                t = total.loc[name]
                
                test = (~np.isnan(r)) & (~np.isnan(t))
                t = t[test]
                r = r[test]
                cont = cont or i<len(t)
                                
                t = t[:i]
                r = r[:i]
                
                
                
                t_f = t[-nfit:]
                r_f = r[-nfit:]
                
                test = (t_f > 100) 
                t_f = t_f[test]#[-nfit:]
                r_f = r_f[test]#[-nfit:]
                                
                test_plot = (t>100) 
                
                axes.plot(t[test_plot], r[test_plot]/t[test_plot], label=name, **styles.get(name, {}), marker="+")
                #if True:#len(t_f)>=nfit:
                if len(t_f)>=nfitcond:
                    try:
                        pol = np.polyfit(t_f, r_f/t_f, 1)
                        p = np.poly1d(pol)  
                    except (np.linalg.LinAlgError, ValueError) as er:
                        print('Failed', name, er)
                    else:
                        #print(name, p)
                        xzero = -pol[1]/pol[0]
                        x1 = t_f[0]
                        print(name, [x1, xzero])
                        axes.plot( [x1, xzero], p([x1,xzero]),  **styles.get(name, {}), linestyle="dashed")
                        axes.plot( xzero, 0,  **styles.get(name, {}), linestyle="None", marker="*")
                        days.setdefault(name, []).append( (t_f.index[-1] - origin.loc[name,'date']).days )
                        xzeros.setdefault(name, []).append(xzero)
                        axes2.plot( days[name], xzeros[name], **styles.get(name, {}))
                        
            #axvline(x1, color="k", linestyle="dotted")
        
            axes.legend()        
            axes.set(ylim=(-0.01,0.3), xlim=(0, 25000), yscale="linear", xscale="linear", xlabel="Totala Death", ylabel="Daily Death / Total", 
            title="Fit of max %s last points"%nfit
            )
            axes2.set(xlim=(0,55), ylim=(0,30000), xlabel="Day since 5 cases", ylabel="Projection of Number of Death")
            writer.grab_frame()
    
    subprocess.run(f"ffmpeg -y -i {root}/{file_name}.mp4 -vf \"fps=10,scale=800:-1:flags=lanczos\" -loop 999 {root}/{file_name}.gif", shell=True)

    
    


