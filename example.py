#!/usr/bin/env python
import mkipp
import matplotlib.pyplot as plt
fig = plt.figure()
axis = fig.add_subplot(111)    
plots = mkipp.kipp_plot(axis, logs_dir = "../LOGS", profile_numbers = range(1,30), \
        identifiers = ["eps_nuc","h1"], contour_cmaps = [plt.get_cmap("Blues"),plt.get_cmap("Oranges")], \
        scales = ["log", "linear"], levels = [[],[0.0,0.2,0.4,0.6,0.8,1.0]], alphas = [1.0,0.2], xaxis = "star_age")
#axis.set_ylim([-2,2])
bar=plt.colorbar(plots["eps_nuc"],pad=0.01)
bar.set_label('Nuclear Energy generation,  Log (erg/g/s)')
bar=plt.colorbar(plots["h1"],pad=0.01)
bar.set_label('h1')
plt.show()
