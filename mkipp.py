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
        self.items = [[min_x, max_x, min_y, max_y]]
        self.mix_type = mix_type
        self.checked = True
    #Check if region is a continuation of the current zone
    def check_region(self,min_x,max_x,min_y,max_y,mix_type):
        last_max_y = self.items[-1][3]
        last_min_y = self.items[-1][2]
        if mix_type!=self.mix_type:
            return False
        if (max_y >= last_max_y and last_max_y > min_y) or (last_max_y >= max_y and max_y > last_min_y):
            self.items.append([min_x, max_x, min_y, max_y])
            self.checked = True
            return True
        return False
    #Creates polygon used to plot hatch
    def get_zone_vertices(self):
        vertices = []
        #Add lower border
        for i in range(len(self.items)):
            vertices.extend(([self.items[i][0], self.items[i][2]], \
                    [self.items[i][1], self.items[i][2]]))
        #Add upper border
        for i in range(1,len(self.items)+1):
            vertices.extend(([self.items[-i][1], self.items[-i][3]], \
                    [self.items[-i][0], self.items[-i][3]]))
        return vertices


# Create contour levels array (TBD: Need to be improved)
def get_levels_linear(min,max,n_levels):
    max = round(max,2)
    min = round(min,2)
    delta = round((max-min)/n_levels,2)
    levels = np.arange(min,max,delta)
    return levels;
    
def get_levels_log(min,max,n_levels):
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
   #Transparencies for contours, useful in case they might overlap. Defaults to 1.0
   alphas = [],
   #List of lists containing the levels for each plotted quantity. If empty, levels will
   #be automatic. If one of the entries is an empty list, levels will be determined automatically
   #for that field.
   levels = [],
   #Scales for the levels. As data can come already in log from the profiles, it needs not match
   #the values given in "scales"
   levels_scale = [],
   #List with the number of levels to use per identifier when using automatic level definition.
   #If not specified it defaults to 8
   levels_num = [],

   ######## PLOT OPTIONS (Note Python is case sensitive: True/False)
   # Either "model_number" or "star_age"
   xaxis = "model_number",
   # Xaxis is Log (t_end - t) # TODO: Needs to be implemented
   xaxis_log_time = False,
   # Either "mass" or "radius"
   yaxis = "mass",
   #Normalize yaxis at each point using total mass/total radius
   yaxis_normalize = False,
   # Visualize Convective Regions
   show_conv = True,
   # Visualize Thermohaline Regions
   show_therm = True,
   # Visualize Semiconvective Regions
   show_semi = True,
   # Visualize Overshoot Regions
   show_over = True,
   # Visualize Contours of Constant Omega (in microHz)
   show_omega = False,
   # Visualize Rotationally Mixed Regions
   show_rot = False,
   ######## PLOT PARAMETERS
   #Resolution in yaxis (Standard value: 300. Increase for higher res)
   #Y direction is divided in this amount of points, and data is interpolated
   #in between.
   numy = 300,
   # To discard tiny convective regions in mass and/or radius. Value represents
   # fraction of total mass or radius
   mass_tolerance = 0.001,
   radius_tolerance = 0.001,
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
           max_y = star_mass = star_radius = 1.0
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
       #reverse y_data and z_data for np.interp
       y_data = y_data[::-1]
       for k in range(len(identifiers)):
           z_data = extractors[k](identifiers[k], scales[k], prof)
           z_data = z_data[::-1]
           interp_z_data = np.interp(y_interp, y_data, z_data)
           for j in range(numy):
               Z_data_array[k,j,i] = interp_z_data[j]

   #Fill defaults for levels and alpha
   if len(levels) == 0:
       levels = [[] for k in range(len(identifiers))]
   if len(levels_scale) == 0:
       levels_scale = ["log"]*len(identifiers)
   if len(levels_num) == 0:
       levels_num = [8]*len(identifiers)
   if len(alphas) == 0:
       alphas = [1.0]*len(identifiers)
   #Get levels that are undefined and plot
   plots = {}
   for k in range(len(identifiers)):
       if len(levels[k]) == 0:
           if levels_scale[k] == "log":
               levels[k] = get_levels_log(np.min(Z_data_array[k,:,:]), np.max(Z_data_array[k,:,:]), \
                       levels_num[k])
           elif levels_scale[k] == "linear":
               levels[k] = get_levels_linear(np.min(Z_data_array[k,:,:]), np.max(Z_data_array[k,:,:]), \
                       levels_num[k])
           else:
               print "unkown level scale " + levels_scale + " for identifier " + identifiers[k]

       #make plot
       plots[identifiers[k]] = axis.contourf(X_data_array, Y_data_array, Z_data_array[k,:,:], \
               cmap=contour_cmaps[k], levels=levels[k], alpha=alphas[k])
                
   # Get mixing zones and mass borders from history.data files
   mix_data = []
   histories = []
   if len(history_names) == 0:
       history_names = [logs_dir + "/" + "history.data"]
   for history_name in history_names:
       print history_name
       histories.append(ms.history_data(".", slname = history_name))
   x_coords = []
   for history in histories:
       x_coords.extend(history.get(xaxis))
   y_coords = []
   if yaxis_normalize:
       y_coords = [1.0]*len(x_coords)
   elif yaxis == "radius":
       for history in histories:
           y_coords.extend(history.get('photosphere_r'))
   else:
       for history in histories:
           y_coords.extend(history.get('star_mass'))

   zones = []
   open_zones = []
   borderTop = []
   borderBottom = []
   mesa_mix_zones = 0
   while True:
       try:
           mesa_mix_zones = mesa_mix_zones + 1
           mix_type = []
           mix_top = []
           for history in histories:
               mix_type.extend(history.get('mix_type_'+str(mesa_mix_zones)))
               if yaxis == "radius":
                    mix_top.extend(history.get('mix_relr_top_'+str(mesa_mix_zones)))
               else:
                    mix_top.extend(history.get('mix_qtop_'+str(mesa_mix_zones)))
           mix_data.append([mix_type, mix_top])
       except Exception, e:
           #reached all mix zones included
           mesa_mix_zones = mesa_mix_zones - 1
           break

   print "history.data has " + str(mesa_mix_zones) + " mix zones"

   if yaxis == "radius":
       tolerance = radius_tolerance
   else:
       tolerance = mass_tolerance

   for i in range(1,len(x_coords)):
       current_x = x_coords[i]
       if current_x > max_x_coord:
           break
       previous_x = x_coords[i-1]
       for j in range(0,mesa_mix_zones):
           mix_type = mix_data[j][0][i]
           if mix_type == 0 or mix_type == -1:
               continue 
           max_y_coord = mix_data[j][1][i]*y_coords[i]
           min_y_coord = 0
           if j > 0:
               min_y_coord = mix_data[j-1][1][i]*y_coords[i]
           #ignore too small regions
           if max_y_coord - min_y_coord < tolerance*y_coords[i]:
               continue
           exists = False
           for z in open_zones:
               if not z.checked:
                   if z.check_region(previous_x,current_x,min_y_coord,max_y_coord,mix_type):
                       exists = True
                       break
           if not exists:
               z = Zone(previous_x,current_x,min_y_coord,max_y_coord,mix_type)
               open_zones.append(z)
       #separate zones which didn't continue here
       for z in open_zones:
           if z.checked == False:
               zones.append(z)
               open_zones.remove(z)
           else:
               z.checked = False

   for z in open_zones:
       zones.append(z)

   for z in zones:
       color = ""
       #Convective mixing
       if z.mix_type == 1 and show_conv:
           color = "Chartreuse"
           hatch = "//"
           line  = 1
       #Overshooting 
       elif z.mix_type == 2 and show_over:
           color = "pink"
           hatch = "--"
           line  = 1
       #Semiconvective mixing
       elif z.mix_type == 3 and show_semi:
           color = "red"
           hatch = "\\\\"
           line  = 1
       #Thermohaline mixing
       elif z.mix_type == 4 and show_therm:
           color = "Gold" #Salmon
           hatch = "||"
           line  = 1
       #Rotational mixing
       elif z.mix_type == 5 and show_rot:
           color = "brown"
           hatch = "--"
           line  = 1
       #Anonymous mixing
       else: 
           color = "white"
           hatch = " "
           line = 0
       axis.add_patch(Polygon(z.get_zone_vertices(),closed=True, fill=False, hatch=hatch, edgecolor=color, linewidth=0))
       axis.add_patch(Polygon(z.get_zone_vertices(),closed=True, fill=False, edgecolor=color, linewidth=line))

   #limit x_coords to data of contours and add line at stellar surface
   for i, x_coord in enumerate(x_coords):
       if x_coord > max_x_coord:
           break
   axis.plot(x_coords[:i], y_coords[:i], "k-")

   return plots
