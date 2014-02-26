#!/usr/bin/env python
"""
Kippenhahn plotter based on SimpleMesaKippy.py and Kippy.py
Includes various mixing zones as hatched areas
History: SimpleMesaKippy.py, Author: Alfred Gautschy /23.VIII.2012
         Kippy.py, Author: Pablo Marchant /13.V.2013
         MKippi.py, Author: Matteo Cantiello /V.2013 (Added LOSSES. Color/Line Style inspired by convplot.pro IDL routine of A.Heger) 
                                             /VI.2013 (Added RADIUS, TIME and NORMALIZE Options. OMEGA Allows to plot w-contours)   
                                             /VII.2013 (Tentatively added Rotational diff. coefficients,conv. velocities and Equipartition B-field)
Requirements: mesa.py. Also needs history.data and profiles.data containing 
              History (star_age,model_number, mixing_regions, mix_relr_regions)
              Profile (star_mass,photosphere_r,q,radius,eps_nuc,non_nuc_neu,logRho,
              conv_vel_div_csound,omega,am_log_D_ST,am_log_D_ES,am_log_D_DSI,log_conv_vel,initial_mass,initial_z,version_number)
"""
import numpy as np
import mesa as ms
from math import log10, pi

#matplotlib specifics
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
#--------------------------------------------------
#...Allow for using TeX mode in matplotlib Figures
#--------------------------------------------------
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

###################################################
##########SUPPORT CLASSES AND FUNCTIONS############
###################################################

# Class used to store mixing zones
class Zone:
    def __init__(self,min_x,max_x,min_y,max_y,mix_type):
        self.border = [[max_x,min_y],[min_x,min_y],[min_x,max_y],[max_x,max_y]]
        self.mix_type = mix_type
        self.checked = True
    def check_region(self,min_x,max_x,min_y,max_y,mix_type):
        last_max_y = self.border[-1][1]
        last_min_y = self.border[0][1]
        if mix_type!=self.mix_type:
            return False
        if (max_y >= last_max_y and last_max_y > min_y) or (last_max_y >= max_y and max_y > last_min_y):
            self.border.append([min_x,max_y])
            self.border.append([max_x,max_y])
            self.border.insert(0,[min_x,min_y])
            self.border.insert(0,[max_x,min_y])
            self.checked = True
            return True
        return False

# Create contour levels array (TBD: Need to be improved)
def get_float_levels(min,max,n_levels):
    max = round(max,2)
    min = round(min,2)
    delta = round((max-min)/n_levels,2)
    levels = np.arange(min,max,delta)
    return levels;
    
def get_levels(min,max,n_levels):
    max = int(max)+1
    min = int(round(min,2))-1
    #levels = range(min+2,max+1)
    levels = range(0,max+1)
    return levels;

def default_extractor(identifier, scale, prof):
    if scale == "linear":
        return prof.get(identifier)
    elif scale == "log":
        return np.log10(prof.get(identifier)+1e-99)
    else:
        print "Unrecognized scale: " + scale

###################################################
########END SUPPORT CLASSES AND FUNCTIONS##########
###################################################

def kipp_plot(
   axis, #matplotlib axis where data will be plotted
   ######## MESA DATA
   #Directory with profile and history data
   logs_dir = 'LOGS',
   #List of profile numbers to be plotted. TODO: automate if not specified
   profile_numbers = [],
   #List of tuples containing logs_dir and profile number. If used logs_dir and
   #profile_numbers is ignored. Use if data is spread accross different logs dirs.
   profile_names = [],
   #List of paths to history.data files to use instead of default one. Use in case data
   #is spread among many history.data files, or you have a single one with non-default
   #location or naming.
   history_names = [],

   ######## CONTOURS TO PLOT
   #Strings used as identifiers of data to plot. If not using any custom extractors these
   #will be used to call get() on the profile.
   identifiers = [],
   #Functions that extract from a profile data to plot. Use when data requires more than
   #a simple get(). The default one is mkipp.default_extractor.
   extractors = [],
   #Scales to use to plot each identifier, can be "linear" or "log". If linear, the data
   #is read as is, if using log, log10 is applied to it.
   scales = [],
   #matplotlib colormaps to use for each plot
   contour_cmaps = [],
   #List of lists containing the levels for each plotted quantity. If empty, levels will
   #be automatic. If one of the entries is an empty list, levels will be determined automatically
   #for that field.
   levels = [],
   #Scales for the levels. As data can come already in log from the profiles, it needs not match
   #the values given in "scales"
   levels_scales = [],
   #List with the number of levels to use per identifier when using automatic level definition.
   #If not specified it defaults to 8
   levels_num = [],

   ######## PLOT OPTIONS (Note Python is case sensitive: True/False)
   xaxis = "model_number",    # Either "model_number" or "star_age"
   xaxis_log_time = False,   # Xaxis is Log (t_end - t) # Need to be implemented
   yaxis = "mass",          # Either "mass" or "radius"
   yaxis_normalize = False, #Normalize yaxis at each point using total mass/total radius
   show_conv = True,       # Visualize Convective Regions
   show_therm = False,     # Visualize Thermohaline Regions
   show_semi = True,       # Visualize Semiconvective Regions
   show_over = True,       # Visualize Overshoot Regions
   show_omega = False,      # Visualize Contours of Constant Omega (in microHz)
   show_rot = True,        # Visualize Rotationally Mixed Regions
   ######## PLOT PARAMETERS
   numy = 300,      #Resolution in yaxis (Standard value: 300. Increase for higher res)
                            #Y direction is divided in this amount of points, and data is interpolated
                            #in between. TODO: actually interpolate
   mass_tolerance = 0.001,       # To discard tiny convective regions in mass and/or radius
   radius_tolerance = 0.00001,
):

   #Fill profile names
   if len(profile_names) == 0:
       profile_names = [(logs_dir, number) for number in profile_numbers]
   #Fill up extractors if not provided
   if len(extractors) == 0:
       extractors = [default_extractor]*len(identifiers)
   #Fill up scales if not provided
   if len(scales) == 0:
       scales = ["log"]*len(identifiers)

   # Initialize interpolation grids
   Z_data_array = np.zeros((len(identifiers),numy,len(profile_names)))

   # XY coordinates for data
   X_data_array = np.zeros((numy,len(profile_names)))
   Y_data_array = np.zeros((numy,len(profile_names)))

   # Extract data from profiles
   max_x_coord = -1
   for i,profile_name in enumerate(profile_names):
       try:
           prof=ms.mesa_profile(profile_name[0], profile_name[1], num_type='profile_num')
       except IOError as e:
           print "Couldn't open profile number " + str(profile_name[1]) + " in folder " + profile_name[0]
       x_coord = prof.header_attr.get(xaxis)
       if x_coord < max_x_coord:
           print "Profiles are not ordered in X coordinate!!!"
       max_x_coord = max(max_x_coord,x_coord)

       #fill up positions
       if yaxis_normalize:
           max_y = 1.0
       elif yaxis == "mass":
           star_mass = prof.header_attr.get('star_mass')
           max_y = star_mass
       elif yaxis == "radius":
           star_radius = prof.header_attr.get('photosphere_r')
           max_y = star_radius
       for j in range(numy):
           X_data_array[j,i] = x_coord
           Y_data_array[j,i] = max_y * j / (numy-1)
       
       #read and interpolate data
       if yaxis == "mass":
           y_data = prof.get('q') * star_mass
           y_interp = np.array([star_mass * j / (numy-1) for j in range(numy)])
       elif yaxis == "radius":
           y_data = prof.get('radius')
           y_interp = np.array([star_radius * j / (numy-1) for j in range(numy)])
       for k in range(len(identifiers)):
           z_data = extractors[k](identifiers[k], scales[k], prof)
           #reverse y_data and z_data for np.interp
           y_data = y_data[::-1]
           z_data = z_data[::-1]
           interp_z_data = np.interp(y_interp, y_data, z_data)
           print y_data
           for j in range(numy):
               Z_data_array[k,j,i] = interp_z_data[j]

   #make the color plots
   fig = plt.figure()
   ax = fig.add_subplot(111)    
   levels_neu = get_levels(np.min(Z_data_array),np.max(Z_data_array),8)
   neuplot = ax.contourf(X_data_array,Y_data_array,Z_data_array[0,:,:],cmap=plt.get_cmap("Purples"),levels=levels_neu)
   bar=plt.colorbar(neuplot,pad=0.01)
   bar.set_label('Nuclear Energy generation,  Log (erg/g/s)')
   plt.show()
                
   ## Get mixing zones and mass borders from history.data
   #mix_data = []
   #history=ms.history_data(logs_dir)
   #xcoords = np.zeros((0))
   #xaxis = history.get(xaxis)
   #if NORMALIZE: # Lame!
   #   if RADIUS:
   #      radii = m2.get('photosphere_r')/m2.get('photosphere_r')
   #   else:
   #      masses = m2.get('star_mass')/m2.get('star_mass')
   #else:
   #   if RADIUS:
   #      radii = m2.get('photosphere_r')
   #   else:
   #      masses = m2.get('star_mass')

   #mod_number=m2.get('model_number')
   #mod_age=m2.get('star_age')

   #ages = m2.get(XAXIS)
   #zones = []
   #open_zones = []
   #borderTop = []
   #borderBottom = []
   #mesa_mix_zones = 0
   #while True:
   #    try:
   #        mesa_mix_zones = mesa_mix_zones + 1
   #        if RADIUS:
   #             mix_data.append([m2.get('mix_type_'+str(mesa_mix_zones)),m2.get('mix_relr_top_'+str(mesa_mix_zones))])
   #        else:
   #             mix_data.append([m2.get('mix_type_'+str(mesa_mix_zones)),m2.get('mix_qtop_'+str(mesa_mix_zones))])
   #    except Exception, e:
   #        mesa_mix_zones = mesa_mix_zones - 1
   #        break

   #print "history.data has " + str(mesa_mix_zones) + " mix zones"

   ## IN Relative RADIUS COORDINATE
   #if RADIUS:
   #    borderTop.append([ages[0],radii[0]]) 
   #    borderBottom.append([ages[0],0])
   #   # borderTop.append([age,radii[i]])    
   #    for i in range(1,len(radii)):
   #     age = ages[i]
   #     age_prev = ages[i-1]
   #     borderTop.append([age,radii[i]])
   #     borderBottom.append([age,0])
   #     for j in range(0,mesa_mix_zones):
   #        mix_type = mix_data[j][0][i]
   #        if mix_type == 0 or mix_type == -1:
   #            continue 
   #        max_radius = mix_data[j][1][i]*radii[i]
   #        min_radius = 0
   #        if j>0:
   #            min_radius = mix_data[j-1][1][i]*radii[i]
   #        #ignore too small regions
   #        if max_radius-min_radius < RADIUS_TOLERANCE*radii[i]:
   #            continue
   #        exists = False
   #        for z in open_zones:
   #            if not z.checked:
   #                if z.check_region(age_prev,age,min_radius,max_radius,mix_type):
   #                    exists = True
   #                    break
   #        if not exists:
   #            z = Zone(age_prev,age,min_radius,max_radius,mix_type)
   #            open_zones.append(z)
   #     for z in open_zones:
   #        if z.checked == False:
   #            zones.append(z)
   #            open_zones.remove(z)
   #        else:
   #            z.checked = False
   #     if age > age_max:
   #        break
   ## IN MASS COORDINATE
   #else:
   #    borderTop.append([ages[0],masses[0]]) 
   #    borderBottom.append([ages[0],0])
   #   # borderTop.append([age,masses[i]])         
   #    for i in range(1,len(masses)):
   #     age = ages[i]
   #     age_prev = ages[i-1]
   #     borderTop.append([age,masses[i]])
   #     borderBottom.append([age,0])
   #     for j in range(0,mesa_mix_zones):
   #        mix_type = mix_data[j][0][i]
   #        if mix_type == 0 or mix_type == -1:
   #            continue 
   #        max_mass = mix_data[j][1][i]*masses[i]
   #        min_mass = 0
   #        if j>0:
   #            min_mass = mix_data[j-1][1][i]*masses[i]
   #        #ignore too small regions
   #        if max_mass-min_mass < MASS_TOLERANCE*masses[i]:
   #            continue
   #        exists = False
   #        for z in open_zones:
   #            if not z.checked:
   #                if z.check_region(age_prev,age,min_mass,max_mass,mix_type):
   #                    exists = True
   #                    break
   #        if not exists:
   #            z = Zone(age_prev,age,min_mass,max_mass,mix_type)
   #            open_zones.append(z)
   #     for z in open_zones:
   #        if z.checked == False:
   #            zones.append(z)
   #            open_zones.remove(z)
   #        else:
   #            z.checked = False
   #     if age > age_max:
   #        break

   #for z in open_zones:
   #    zones.append(z)
   #borderBottom.reverse()
   #borderTop.extend(borderBottom)

   ##Determine levels for OMEGA
   #omax = round(np.max(ome))
   #omin = round(np.min(ome))
   #delta=round((omax-omin)/OME_LEVELS,2)
   #if delta > 0.0:
   #   levels_ome = np.arange(omin,omax,delta)
   #else:
   #   levels_ome=[0]

   # MAKE PLOT

   #fig = plt.figure()
   #if HORIZONTAL:
   #    ax = fig.add_subplot(111,position=[0.1,0.08,0.85,0.78])   
   #else:
   #    ax = fig.add_subplot(111)    
   #if LOSSES:
   #    levels_neu = get_levels(min_neu,max_neu,NEU_LEVELS)
   #    neuplot = ax.contourf(Xeps,Yeps,neu,cmap=plt.get_cmap("Purples"),levels=levels_neu)
   #if GAINS:   
   #    levels_eps = get_levels(min_eps,max_eps,EPS_LEVELS)  
   #    print min_eps, max_eps, levels_eps
   #    epsplot = ax.contourf(Xeps,Yeps,eps,cmap=plt.get_cmap("Blues"),levels=levels_eps)
   #if CVEL: 
   #    #min_con = -3 
   #    levels_con = get_float_levels(min_con,max_con,CON_LEVELS)  
   #    conplot = ax.contourf(Xeps,Yeps,con,cmap=plt.get_cmap("Oranges"),levels=levels_con) 
   #elif BFI:
   #    #levels_bfi = get_levels(min_bfi,max_bfi,BFI_LEVELS) 
   #    levels_bfi=(3,3.5,4,4.5,5,5.5,6)
   #    bfiplot = ax.contourf(Xeps,Yeps,bfi,cmap=plt.get_cmap("Purples"),levels=levels_bfi)
   #if ST:
   #    min_mst = 0    
   #    levels_mst = get_float_levels(min_mst,max_mst,MST_LEVELS)  
   #    stplot = ax.contourf(Xeps,Yeps,mst,cmap=plt.get_cmap("Greens"),levels=levels_mst)
   #if ES:
   #    min_mes = 0
   #    levels_mes = get_float_levels(min_mes,max_mes,MES_LEVELS)    
   #    esplot = ax.contourf(Xeps,Yeps,mes,cmap=plt.get_cmap("Reds"),levels=levels_mes)
   #if SI:  
   #    #min_msi = 3 
   #    levels_msi = get_levels(min_msi,max_msi,MSI_LEVELS) 
   #    siplot = ax.contourf(Xeps,Yeps,msi,cmap=plt.get_cmap("Oranges"),levels=levels_msi)                    
   ## create a legend for the contour set
   ##hatches=['//', '||']
   ##colors=['Chartreuse','red']
   ##names = ['Convection','Semiconvection']
   ##plt.legend(names, handleheight=2)
   #for z in zones:
   #    color = ""
   #    #Convective mixing
   #    if z.mix_type == 1 and show_conv:
   #        color = "Chartreuse"
   #        hatch = "//"
   #        line  = 1
   #    #Overshooting 
   #    elif z.mix_type == 2 and show_over:
   #        color = "pink"
   #        hatch = "--"
   #        line  = 1
   #    #Semiconvective mixing
   #    elif z.mix_type == 3 and show_semi:
   #        color = "red"
   #        hatch = "\\\\"
   #        line  = 1
   #    #Thermohaline mixing
   #    elif z.mix_type == 4 and show_therm:
   #        color = "Gold" #Salmon
   #        hatch = "||"
   #        line  = 1
   #    #Rotational mixing
   #    elif z.mix_type == 5 and show_rot:
   #        color = "brown"
   #        hatch = "--"
   #        line  = 1
   #    #Anonymous mixing
   #    else: 
   #        color = "white"
   #        hatch = " "
   #        line = 0
   #    ax.add_patch(Polygon(z.border,closed=True, fill=False, hatch=hatch, edgecolor=color, linewidth=0))
   #    ax.add_patch(Polygon(z.border,closed=True, fill=False, edgecolor=color, linewidth=line))
   #ax.add_patch(Polygon(borderTop,closed=True, fill=False, edgecolor='black'))    # Add Black Line at Stellar Surface
   #if GAINS:
   #    if LOSSES:    
   #      bar=plt.colorbar(epsplot,pad=0.01)
   #      bar.set_label('Nuclear Energy generation,  Log (erg/g/s)')
   #      bar2=plt.colorbar(neuplot,pad=0.05)
   #      bar2.set_label('Neutrino Losses,  Log (erg/g/s)')
   #    elif HORIZONTAL:   
   #      cbaxes = fig.add_axes([0.1, 0.91, 0.85,0.04]) 
   #      bar=plt.colorbar(epsplot,orientation='horizontal',cax = cbaxes)   
   #      bar.set_label('Nuclear Energy generation,  Log (erg/g/s)',labelpad=-47)
   #    else:
   #      bar=plt.colorbar(epsplot,pad=0.05) 
   #      bar.set_label('Nueclear Energy generation,  Log (erg/g/s)')  
   #if BFI:
   #    bar3=plt.colorbar(bfiplot,pad=0.05)
   #    bar3.set_label('Equipartition B-Field,  Log B (G)') 
   #if ST:
   #    bar4=plt.colorbar(stplot,pad=0.05)
   #    bar4.set_label('Spruit-Tayler Dynamo,  Log D (cm$^2$/s)') 
   #if SI:
   #    bar5=plt.colorbar(siplot,pad=0.05)
   #    bar5.set_label('Dynamical Shear,  Log D (cm$^2$/s)')
   #if ES:
   #    bar6=plt.colorbar(esplot,pad=0.05)
   #    bar6.set_label('Eddington-Sweet,  Log D (cm$^2$/s)')                   
   ##if LOSSES:
   ##    bar2=plt.colorbar(neuplot,pad=0.05)
   ##    bar2.set_label('Neutrino Losses,  Log (erg/g/s)')
   #if  OMEGA:
   #    levels_ome = get_float_levels(min_ome,max_ome,OME_LEVELS)
   #    # NEXT LINE OVERRIDE PREVIOUS CALCULATION OF OMEGA LEVELS. Improve
   #    levels_ome=[-7.0,-6.5,-6.0,-5.5,-5.2,-5.0,-4.5,-4.00]
   #    omegaplot = ax.contour(Xeps,Yeps,ome,levels=levels_ome,linestyles='dashed') 
   #    ax.clabel(omegaplot, inline=1, fontsize=12,thickness=2,colors='black')   
   #    
   ##TBD: A PLOT LEGEND WOULD BE USEFUL!

   #if TIME:
   #   mod_n=mod_age
   #else:
   #   mod_n=mod_number
   #if RQ:
   ## To add coordinates of q
   #     rq1=m2.get('r_q_01')
   #     rq2=m2.get('r_q_02')
   #     rq3=m2.get('r_q_03')
   #     rq4=m2.get('r_q_04')
   #     rq5=m2.get('r_q_05')
   #     rq6=m2.get('r_q_06')
   #     rq7=m2.get('r_q_07')
   #     rq8=m2.get('r_q_08')
   #     rq9=m2.get('r_q_09')
   #     plt.plot(mod_n,rq1,'b')
   #     plt.plot(mod_n,rq2,'b')
   #     plt.plot(mod_n,rq3,'b')
   #     plt.plot(mod_n,rq4,'b')
   #     plt.plot(mod_n,rq5,'r',linewidth=2)
   #     plt.plot(mod_n,rq6,'b')
   #     plt.plot(mod_n,rq7,'b')
   #     plt.plot(mod_n,rq8,'b')
   #     plt.plot(mod_n,rq9,'b') 

   #if HEB:
   #    plt.plot(mod_n,he,'b:') # He core boundary Mass

   #if COB:
   ##    plt.plot(mod_n,he,'r:') # CO core boundary Mass
   #    plt.plot(mod_n,cmass,'b:') # CO core boundary Mass
   #    plt.plot(mod_n,omass,'g:') # CO core boundary Mass
   ## labeling the plot

   #if len(TITLE) > 0:      # USER Defined Title
   #    ax.set_title(TITLE)
   #elif HORIZONTAL:                   # Automatically Generated Title
   #    mm=m2.header_attr.get('initial_mass')
   #    zz=m2.header_attr.get('initial_z')
   #    nn=m2.header_attr.get('version_number')
   #    TITLE = '%2.1f$M_\odot$, Z=%1.4f,   MESA V. %4i' % (mm, zz, nn)
   #    #ax.set_title(TITLE)
   #else:
   #    mm=m2.header_attr.get('initial_mass')
   #    zz=m2.header_attr.get('initial_z')
   #    nn=m2.header_attr.get('version_number')
   #    print str(mm)+" "+str(zz)+" "+str(nn)
   #    TITLE = '%2.1f$M_\odot$, Z=%1.4f,   MESA V. %4i' % (mm, zz, nn)
   #    ax.set_title(TITLE)

   #if TIME:
   #    ax.set_xlabel(r'$t$')
   #else:
   #    ax.set_xlabel(r'$Model Number$')

   #if RADIUS:
   #   if NORMALIZE:
   #      ax.set_ylabel(r'$r/R$')
   #   else:   
   #      ax.set_ylabel(r'$r/R_\odot$')
   #else:
   #   if NORMALIZE:
   #      ax.set_ylabel(r'$m/M$')
   #   else:   
   #      ax.set_ylabel(r'$M/M_\odot$')
   #if YRANGE:
   #   plt.ylim(YLIM)
   #if XRANGE:
   #   plt.xlim(XLIM)         
   #if SAVEFILE:
   #    #save plot
   #    plt.savefig(FILENAME)
   #else:
   #    plt.show()
