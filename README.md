
Tools to load and annylise the Corona virus daily report data from the GitHub of Johns Hopkins University Center for Systems Science and Engineering : [https://github.com/CSSEGISandData/COVID-19](https://github.com/CSSEGISandData/COVID-19)

The best way to have an overview is to check the [notebook](./corona.ipynb)

## install 

Using pip

```
>>> pip install coronapy 
```

```
> git clone https://github.com/SylvainGuieu/corona.git
> cd corona 
> python setup.py install
```

Or simply copy `corona.py` in python directory (matplotlib and pandas modules needed)

## Usage 

Best to see the [notebook](./corona.ipynb) for examples. 

## Some Key plots
Some plot created by the coronapy_key_plot.py script in this repository

Last Updates : 2020-03-24

![](./img/confirmed.png)

![](./img/confirmed_days_200.png)

![](./img/death_days_20.png)

![](./img/death_fit.png)

![](./img/death_fit_days.png)

![](./img/confirmed_T.png)

![](./img/death_T.png)

![](./img/death_ratio.png)
