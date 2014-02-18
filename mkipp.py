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

# Create Interpolation Matrix
def matrix(mat,numz,values,q,min_mat,max_mat,i,max_eps,min_eps):
    k = 0
    for j in range(numz-1,-1,-1):
        for l in range(k,len(values)):
            if (q[l]<=float(j)/float(numz-1)):
                k=l
                break
        mat[j,i] = log10(abs(values[k])+1e-99)
        if(mat[j,i]>max_eps):
            max_mat = mat[j,i]
        if(mat[j,i]<min_eps):
            min_mat = mat[j,i]     
    return mat, min_mat, max_mat;


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

###################################################
########END SUPPORT CLASSES AND FUNCTIONS##########
###################################################

def kipp_plot(
   ######## MESA DATA     
   LOGS_DIR = 'LOGS',
   FIRST_PROFILE = 1,     # First profile#  to read
   LAST_PROFILE = 500,    # Last profile# to read (N.B: Be careful that you haven't hit the max number of saved profiles (set 'max_num_profile_models = 100..'))
   PROFILE_STEP = 5,      # Stepping sequence
   ######## STRINGS
   FILENAME = "Kippenhahn.eps",
   TITLE ='',
   ######## PLOT OPTIONS (Note Python is case sensitive: True/False)
   TIME = False,      # Xaxis is Time. If False: Model Number
   #LOGTIME = True   # Xaxis is Log (t_end - t) # Need to be implemented
   SAVEFILE = True,   # Save pdf version of the plot with name FILENAME
   GAINS = True,      # Visualize Energy Generation (Nuclear - Neutrino Losses). Improve: Right now at least one of GAINS and LOSS need to be True
   LOSSES = False,    # Visualize Neutrino Losses
   CONV = True,       # Visualize Convective Regions
   THERM = False,     # Visualize Thermohaline Regions
   SEMI = True,       # Visualize Semiconvective Regions
   OVER = True,       # Visualize Overshoot Regions
   OMEGA = False,      # Visualize Contours of Constant Omega (in microHz)
   RADIUS = False,     # Radius as Y coordinate
   NORMALIZE = False,  # Normalized to total Mass/Radius coordinate
   XRANGE = False,    # Control X plot range (uses XLIM)
   YRANGE = False,     # Control Y plot range (uses YLIM)
   ## Experimental    # All these require error message improvements (e.g. if some of the required data is not present in the profiles)
   BFI  = False,       # Visualize Equipartition Magnetic Fields (NB: Only one between BFI or CVEL allowed. Improve with error message) 
   CVEL = False,      # Visualize Convective Velocities         (NB: Only one between BFI or CVEL allowed. Improve with error message) 
   ST   = False,       # " Diffusion coefficient for Spruit-Tayler B-fields 
   SI   = False,      # " Diffusion coefficient for Dynamical Shear Instability
   ES   = False,      # " Diffusion coefficient for Eddington-Sweet Circulation
   ROT  = False,      # Visualize Rotationally Mixed Regions
   HEB   = False,       # Show He-core Boundary (Be careful Mass/Radius!)
   COB   = False,      # Show Co-core Boundary
   RQ    = False,      # Show radial coordinates corresponding to q=0.2,0.4,0.6,0.8
   ######## PLOT PARAMETERS
   XLIM = [0,1000], # IF XRANGE = True uses X-axis limits 
   YLIM = [0.0,250],       # IF YRANGE = True uses Y-axis limits 
   HORIZONTAL = False,  # To plot the colorbar (energy generation) horizontally on top of the plot
   EPS_LEVELS = 8,   # Contour Levels for Energy Generation (Nuclear - Neutrino Losses)
   NEU_LEVELS = 8,   # Contour Levels for Neutrino Losses
   CON_LEVELS = 8,   # Contour Levels for Log(Convective Velocity)
   BFI_LEVELS = 8,   # Contour Levels for Log(B_Equipartition) 
   OME_LEVELS = 6,   # Contour Levels for Log(Angular Velocity)
   MST_LEVELS = 6,   # Contour Levels for Log(ST,ES,DSI)
   MES_LEVELS = 6,   # " 
   MSI_LEVELS = 6,   # "
   numz = 300,       # Interpolation Parameter (Standard value: 300. Increase for higher res)
   MASS_TOLERANCE = 0.001,       # To discard tiny convective regions in mass and/or radius
   RADIUS_TOLERANCE = 0.00001,

   # Initial Guesses for maximum values of interpolated grids 
   max_eps = 0,
   max_neu = 0,
   max_con = 0,
   max_bfi = 0,
   max_mst = 0,
   max_mes = 0,
   max_msi = 0,
   max_ome = 0,

   min_eps = 3,
   min_neu = 0,
   min_con = 0,
   min_bfi = 0,
   min_mst = 0,
   min_mes = 0,
   min_msi = 0,
   min_ome = 0,
):

   if TIME:
      XAXIS = 'star_age'
   else:
      XAXIS = 'model_number'           
      
   # Models to be plotted
   models = np.arange(FIRST_PROFILE,LAST_PROFILE,PROFILE_STEP) # how to step through profile-file sequence

   # Initialize interpolation grids
   eps = np.zeros((numz,len(models)))
   neu = np.zeros((numz,len(models)))
   ome = np.zeros((numz,len(models)))
   con = np.zeros((numz,len(models)))
   bfi = np.zeros((numz,len(models)))
   mst = np.zeros((numz,len(models)))
   msi = np.zeros((numz,len(models)))
   mes = np.zeros((numz,len(models)))

   Xeps = np.zeros((numz,len(models)))
   Yeps = np.zeros((numz,len(models)))
   i = -1
   # Get data for epsilon from profiles
   for mod_no in models:
       i=i+1
       try:
           prof=ms.mesa_profile(LOGS_DIR,mod_no,num_type='profile_num')
       except IOError as e:
           print "Couldn't open model number "+str(mod_no)
       age = prof.header_attr.get(XAXIS)
       age_max = age
       if NORMALIZE:   # Sets normalization of interpolated quantities (Matrix calculated below)
          #LAME! Improve next (set to single valued array = 1.0, with dimension n)
          if RADIUS:
             radius = prof.header_attr.get('photosphere_r')/prof.header_attr.get('photosphere_r')
          else:
             mass = prof.header_attr.get('star_mass')/prof.header_attr.get('star_mass')
       else:
          mass = prof.header_attr.get('star_mass')
          #radius = prof.header_attr.get('photosphere_r')
       #fill up positions
       for j in range(numz):
           Xeps[j,i] = age
           if RADIUS:
             radius = prof.header_attr.get('photosphere_r') 
             Yeps[j,i] = radius*j/(numz-1)
           else:   
             Yeps[j,i] = mass*j/(numz-1)
       q = prof.get('q')
       if RADIUS:
           r=  prof.get('radius')/prof.header_attr.get('photosphere_r')   #LAME!
           q = r
           
   #Get values of profile quantities    
       epsnuc    = prof.get('eps_nuc')    
       epsneu    = prof.get('non_nuc_neu')
       density   = prof.get('logRho')
       vcon_div_cs  = prof.get('conv_vel_div_csound')     
   #    omega     = prof.get('omega')/(2.0*pi)
   #    DST       = 10**(prof.get('am_log_D_ST'))
   #    DES       = 10**(prof.get('am_log_D_ES'))
   #    DSI       = 10**(prof.get('am_log_D_DSI'))
       log_conv_vel = prof.get('log_conv_vel')   
       conv_vel  = 10**(log_conv_vel)
       b_field    = ((10**density)**0.5)*((2.0*3.14)**0.5)*conv_vel
       
   # Create Nuclear Energy Generation Matrix
       eps, min_eps, max_eps = matrix(eps,numz,epsnuc,q,min_eps,max_eps,i,max_eps,min_eps)
       neu, min_neu, max_neu = matrix(neu,numz,epsneu,q,min_neu,max_neu,i,max_eps,min_eps)

   # Create Other Matrix 
       con, min_con, max_con = matrix(con,numz,conv_vel,q,min_con,max_con,i,max_eps,min_eps)
       bfi, min_bfi, max_bfi = matrix(bfi,numz,b_field,q,min_bfi,max_bfi,i,max_eps,min_eps)
   #    ome, min_ome, max_ome = matrix(ome,numz,omega,q,min_ome,max_ome,i,max_eps,min_eps)
   #    mst, min_mst, max_mst = matrix(mst,numz,DST,q,min_mst,max_mst,i,max_eps,min_eps)
   #    msi, min_msi, max_msi = matrix(msi,numz,DSI,q,min_mst,max_msi,i,max_eps,min_eps)
   #    mes, min_mes, max_mes = matrix(mes,numz,DES,q,min_mes,max_mes,i,max_eps,min_eps)
                
   # Get mixing zones and mass borders from history.data

   mix_data = []
   m2=ms.history_data(LOGS_DIR)
   xaxis = m2.get(XAXIS)
   if NORMALIZE: # Lame!
      if RADIUS:
         radii = m2.get('photosphere_r')/m2.get('photosphere_r')
      else:
         masses = m2.get('star_mass')/m2.get('star_mass')
   else:
      if RADIUS:
         radii = m2.get('photosphere_r')
      else:
         masses = m2.get('star_mass')

   # TO ADD BOUNDARY MASS/RADIUS LINES  # TBD: Some of the var names have changed. Need to update

   #h1 = m2.get('h1_boundary_mass')
   #he4 = m2.get('he4_boundary_mass')
   #c12 = m2.get('c12_boundary_mass')
   #r1 = m2.get('h1_boundary_radius')
   #r4 = m2.get('he4_boundary_radius')
   #r12 = m2.get('c12_boundary_radius')
   # TO ADD BOUNDARY MASS/RADIUS LINES

   mod_number=m2.get('model_number')
   mod_age=m2.get('star_age')

   if (HEB or COB or RQ) :
      #h1 = m2.get('h1_boundary_mass')
      he = m2.get('he_core_mass')
      cmass = m2.get('c_core_mass')
      omass = m2.get('o_core_mass')
      #r1 = m2.get('h1_boundary_radius')
      #r4 = m2.get('he4_boundary_radius')
      #r12 = m2.get('c12_boundary_radius')
      


   ages = m2.get(XAXIS)
   zones = []
   open_zones = []
   borderTop = []
   borderBottom = []
   mesa_mix_zones = 0
   while True:
       try:
           mesa_mix_zones = mesa_mix_zones + 1
           if RADIUS:
                mix_data.append([m2.get('mix_type_'+str(mesa_mix_zones)),m2.get('mix_relr_top_'+str(mesa_mix_zones))])
           else:
                mix_data.append([m2.get('mix_type_'+str(mesa_mix_zones)),m2.get('mix_qtop_'+str(mesa_mix_zones))])
       except Exception, e:
           mesa_mix_zones = mesa_mix_zones - 1
           break

   print "history.data has " + str(mesa_mix_zones) + " mix zones"

   # IN Relative RADIUS COORDINATE
   if RADIUS:
       borderTop.append([ages[0],radii[0]]) 
       borderBottom.append([ages[0],0])
      # borderTop.append([age,radii[i]])    
       for i in range(1,len(radii)):
        age = ages[i]
        age_prev = ages[i-1]
        borderTop.append([age,radii[i]])
        borderBottom.append([age,0])
        for j in range(0,mesa_mix_zones):
           mix_type = mix_data[j][0][i]
           if mix_type == 0 or mix_type == -1:
               continue 
           max_radius = mix_data[j][1][i]*radii[i]
           min_radius = 0
           if j>0:
               min_radius = mix_data[j-1][1][i]*radii[i]
           #ignore too small regions
           if max_radius-min_radius < RADIUS_TOLERANCE*radii[i]:
               continue
           exists = False
           for z in open_zones:
               if not z.checked:
                   if z.check_region(age_prev,age,min_radius,max_radius,mix_type):
                       exists = True
                       break
           if not exists:
               z = Zone(age_prev,age,min_radius,max_radius,mix_type)
               open_zones.append(z)
        for z in open_zones:
           if z.checked == False:
               zones.append(z)
               open_zones.remove(z)
           else:
               z.checked = False
        if age > age_max:
           break
   # IN MASS COORDINATE
   else:
       borderTop.append([ages[0],masses[0]]) 
       borderBottom.append([ages[0],0])
      # borderTop.append([age,masses[i]])         
       for i in range(1,len(masses)):
        age = ages[i]
        age_prev = ages[i-1]
        borderTop.append([age,masses[i]])
        borderBottom.append([age,0])
        for j in range(0,mesa_mix_zones):
           mix_type = mix_data[j][0][i]
           if mix_type == 0 or mix_type == -1:
               continue 
           max_mass = mix_data[j][1][i]*masses[i]
           min_mass = 0
           if j>0:
               min_mass = mix_data[j-1][1][i]*masses[i]
           #ignore too small regions
           if max_mass-min_mass < MASS_TOLERANCE*masses[i]:
               continue
           exists = False
           for z in open_zones:
               if not z.checked:
                   if z.check_region(age_prev,age,min_mass,max_mass,mix_type):
                       exists = True
                       break
           if not exists:
               z = Zone(age_prev,age,min_mass,max_mass,mix_type)
               open_zones.append(z)
        for z in open_zones:
           if z.checked == False:
               zones.append(z)
               open_zones.remove(z)
           else:
               z.checked = False
        if age > age_max:
           break

   for z in open_zones:
       zones.append(z)
   borderBottom.reverse()
   borderTop.extend(borderBottom)

   ##Determine levels for OMEGA
   #omax = round(np.max(ome))
   #omin = round(np.min(ome))
   #delta=round((omax-omin)/OME_LEVELS,2)
   #if delta > 0.0:
   #   levels_ome = np.arange(omin,omax,delta)
   #else:
   #   levels_ome=[0]

   # MAKE PLOT

   fig = plt.figure()
   if HORIZONTAL:
       ax = fig.add_subplot(111,position=[0.1,0.08,0.85,0.78])   
   else:
       ax = fig.add_subplot(111)    
   if LOSSES:
       levels_neu = get_levels(min_neu,max_neu,NEU_LEVELS)
       neuplot = ax.contourf(Xeps,Yeps,neu,cmap=plt.get_cmap("Purples"),levels=levels_neu)
   if GAINS:   
       levels_eps = get_levels(min_eps,max_eps,EPS_LEVELS)  
       print min_eps, max_eps, levels_eps
       epsplot = ax.contourf(Xeps,Yeps,eps,cmap=plt.get_cmap("Blues"),levels=levels_eps)
   if CVEL: 
       #min_con = -3 
       levels_con = get_float_levels(min_con,max_con,CON_LEVELS)  
       conplot = ax.contourf(Xeps,Yeps,con,cmap=plt.get_cmap("Oranges"),levels=levels_con) 
   elif BFI:
       #levels_bfi = get_levels(min_bfi,max_bfi,BFI_LEVELS) 
       levels_bfi=(3,3.5,4,4.5,5,5.5,6)
       bfiplot = ax.contourf(Xeps,Yeps,bfi,cmap=plt.get_cmap("Purples"),levels=levels_bfi)
   if ST:
       min_mst = 0    
       levels_mst = get_float_levels(min_mst,max_mst,MST_LEVELS)  
       stplot = ax.contourf(Xeps,Yeps,mst,cmap=plt.get_cmap("Greens"),levels=levels_mst)
   if ES:
       min_mes = 0
       levels_mes = get_float_levels(min_mes,max_mes,MES_LEVELS)    
       esplot = ax.contourf(Xeps,Yeps,mes,cmap=plt.get_cmap("Reds"),levels=levels_mes)
   if SI:  
       #min_msi = 3 
       levels_msi = get_levels(min_msi,max_msi,MSI_LEVELS) 
       siplot = ax.contourf(Xeps,Yeps,msi,cmap=plt.get_cmap("Oranges"),levels=levels_msi)                    
   # create a legend for the contour set
   #hatches=['//', '||']
   #colors=['Chartreuse','red']
   #names = ['Convection','Semiconvection']
   #plt.legend(names, handleheight=2)
   for z in zones:
       color = ""
       #Convective mixing
       if z.mix_type == 1 and CONV:
           color = "Chartreuse"
           hatch = "//"
           line  = 1
       #Overshooting 
       elif z.mix_type == 2 and OVER:
           color = "pink"
           hatch = "--"
           line  = 1
       #Semiconvective mixing
       elif z.mix_type == 3 and SEMI:
           color = "red"
           hatch = "\\\\"
           line  = 1
       #Thermohaline mixing
       elif z.mix_type == 4 and THERM:
           color = "Gold" #Salmon
           hatch = "||"
           line  = 1
       #Rotational mixing
       elif z.mix_type == 5 and ROT:
           color = "brown"
           hatch = "--"
           line  = 1
       #Anonymous mixing
       else: 
           color = "white"
           hatch = " "
           line = 0
       ax.add_patch(Polygon(z.border,closed=True, fill=False, hatch=hatch, edgecolor=color, linewidth=0))
       ax.add_patch(Polygon(z.border,closed=True, fill=False, edgecolor=color, linewidth=line))
   ax.add_patch(Polygon(borderTop,closed=True, fill=False, edgecolor='black'))    # Add Black Line at Stellar Surface
   if GAINS:
       if LOSSES:    
         bar=plt.colorbar(epsplot,pad=0.01)
         bar.set_label('Nuclear Energy generation,  Log (erg/g/s)')
         bar2=plt.colorbar(neuplot,pad=0.05)
         bar2.set_label('Neutrino Losses,  Log (erg/g/s)')
       elif HORIZONTAL:   
         cbaxes = fig.add_axes([0.1, 0.91, 0.85,0.04]) 
         bar=plt.colorbar(epsplot,orientation='horizontal',cax = cbaxes)   
         bar.set_label('Nuclear Energy generation,  Log (erg/g/s)',labelpad=-47)
       else:
         bar=plt.colorbar(epsplot,pad=0.05) 
         bar.set_label('Nueclear Energy generation,  Log (erg/g/s)')  
   if BFI:
       bar3=plt.colorbar(bfiplot,pad=0.05)
       bar3.set_label('Equipartition B-Field,  Log B (G)') 
   if ST:
       bar4=plt.colorbar(stplot,pad=0.05)
       bar4.set_label('Spruit-Tayler Dynamo,  Log D (cm$^2$/s)') 
   if SI:
       bar5=plt.colorbar(siplot,pad=0.05)
       bar5.set_label('Dynamical Shear,  Log D (cm$^2$/s)')
   if ES:
       bar6=plt.colorbar(esplot,pad=0.05)
       bar6.set_label('Eddington-Sweet,  Log D (cm$^2$/s)')                   
   #if LOSSES:
   #    bar2=plt.colorbar(neuplot,pad=0.05)
   #    bar2.set_label('Neutrino Losses,  Log (erg/g/s)')
   if  OMEGA:
       levels_ome = get_float_levels(min_ome,max_ome,OME_LEVELS)
       # NEXT LINE OVERRIDE PREVIOUS CALCULATION OF OMEGA LEVELS. Improve
       levels_ome=[-7.0,-6.5,-6.0,-5.5,-5.2,-5.0,-4.5,-4.00]
       omegaplot = ax.contour(Xeps,Yeps,ome,levels=levels_ome,linestyles='dashed') 
       ax.clabel(omegaplot, inline=1, fontsize=12,thickness=2,colors='black')   
       
   #TBD: A PLOT LEGEND WOULD BE USEFUL!

   if TIME:
      mod_n=mod_age
   else:
      mod_n=mod_number
   if RQ:
   # To add coordinates of q
        rq1=m2.get('r_q_01')
        rq2=m2.get('r_q_02')
        rq3=m2.get('r_q_03')
        rq4=m2.get('r_q_04')
        rq5=m2.get('r_q_05')
        rq6=m2.get('r_q_06')
        rq7=m2.get('r_q_07')
        rq8=m2.get('r_q_08')
        rq9=m2.get('r_q_09')
        plt.plot(mod_n,rq1,'b')
        plt.plot(mod_n,rq2,'b')
        plt.plot(mod_n,rq3,'b')
        plt.plot(mod_n,rq4,'b')
        plt.plot(mod_n,rq5,'r',linewidth=2)
        plt.plot(mod_n,rq6,'b')
        plt.plot(mod_n,rq7,'b')
        plt.plot(mod_n,rq8,'b')
        plt.plot(mod_n,rq9,'b') 

   if HEB:
       plt.plot(mod_n,he,'b:') # He core boundary Mass

   if COB:
   #    plt.plot(mod_n,he,'r:') # CO core boundary Mass
       plt.plot(mod_n,cmass,'b:') # CO core boundary Mass
       plt.plot(mod_n,omass,'g:') # CO core boundary Mass
   # labeling the plot

   if len(TITLE) > 0:      # USER Defined Title
       ax.set_title(TITLE)
   elif HORIZONTAL:                   # Automatically Generated Title
       mm=m2.header_attr.get('initial_mass')
       zz=m2.header_attr.get('initial_z')
       nn=m2.header_attr.get('version_number')
       TITLE = '%2.1f$M_\odot$, Z=%1.4f,   MESA V. %4i' % (mm, zz, nn)
       #ax.set_title(TITLE)
   else:
       mm=m2.header_attr.get('initial_mass')
       zz=m2.header_attr.get('initial_z')
       nn=m2.header_attr.get('version_number')
       print str(mm)+" "+str(zz)+" "+str(nn)
       TITLE = '%2.1f$M_\odot$, Z=%1.4f,   MESA V. %4i' % (mm, zz, nn)
       ax.set_title(TITLE)

   if TIME:
       ax.set_xlabel(r'$t$')
   else:
       ax.set_xlabel(r'$Model Number$')

   if RADIUS:
      if NORMALIZE:
         ax.set_ylabel(r'$r/R$')
      else:   
         ax.set_ylabel(r'$r/R_\odot$')
   else:
      if NORMALIZE:
         ax.set_ylabel(r'$m/M$')
      else:   
         ax.set_ylabel(r'$M/M_\odot$')
   if YRANGE:
      plt.ylim(YLIM)
   if XRANGE:
      plt.xlim(XLIM)         
   #ax.legend(loc=3,label=[r"$y = \alpha^2$",r"$y = \alpha^2$"])
   if SAVEFILE:
       #save plot
       plt.savefig(FILENAME)
   else:
       plt.show()
