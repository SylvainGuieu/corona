import corona as crn

confirmed, death, recoverd = crn.load_data()

sample = dict(countries=["France", "Italy", "Germany", "Spain"], date_min="2020-03-01")
confirmed_ue, death_ue = (crn.get_subdata(data, **sample) for data in [confirmed, death])
print(confirmed_ue)

crn.plot_data(confirmed_ue , ylabel="Confirmed")
crn.plt.gcf().set_size_inches( (10,7) )
crn.plt.gcf().savefig("./img/confirmed.png")
crn.plot_data(confirmed_ue , ylabel="Confirmed", log=True)
crn.plt.gcf().set_size_inches( (10,7) )
crn.plt.gcf().savefig("./img/confirmed_log.png")
crn.plot_proportion(death_ue, confirmed_ue, ylabel="Death/Confirmed %")
crn.plt.gcf().set_size_inches( (10,7) )
crn.plt.gcf().savefig("./img/death_ratio.png")
#crn.plt.show()