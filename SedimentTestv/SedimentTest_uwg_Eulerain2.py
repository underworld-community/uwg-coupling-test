#!/usr/bin/env python
# coding: utf-8

# In[1]:


# https://github.com/underworldcode/underworld2/blob/main/docs/examples/08_Uplift_TractionBCs.ipynb

import underworld as uw
from underworld import function as fn
import math 
import numpy as np
import os

from underworld import UWGeodynamics as GEO

from _badlands_evalfelev import *

u = GEO.UnitRegistry
ndim = GEO.non_dimensionalise
dimen = GEO.dimensionalise

comm = uw.mpi.comm
rank = uw.mpi.rank
size = uw.mpi.size

# GEO.rcParams["initial.nonlinear.tolerance"] = 1e-3
# GEO.rcParams['initial.nonlinear.max.iterations'] = 100
# GEO.rcParams["nonlinear.tolerance"] = 1e-3
# GEO.rcParams['nonlinear.max.iterations'] = 100
GEO.rcParams["popcontrol.particles.per.cell.2D"] = 16
GEO.rcParams["swarm.particles.per.cell.2D"] = 16
GEO.rcParams["surface.pressure.normalization"] = True
GEO.rcParams["pressure.smoothing"] = True
GEO.rcParams["popcontrol.split.threshold"] = 0.1


# In[2]:


half_rate = 1.0 * u.centimeter / u.year
model_length = 100. * u.kilometer
gravity = 9.81 * u.meter / u.second**2
bodyforce = 3300 * u.kilogram / u.metre**3 *gravity 

KL = model_length
Kt = KL / half_rate
KM = bodyforce * KL**2 * Kt**2

GEO.scaling_coefficients["[length]"] = KL
GEO.scaling_coefficients["[time]"] = Kt
GEO.scaling_coefficients["[mass]"]= KM

xmin, xmax = ndim(-100 * u.kilometer), ndim(100 * u.kilometer)
ymin, ymax = ndim(-24 * u.kilometer), ndim(8 * u.kilometer)
yint = 0.

#dy = ndim(100 * u.kilometer)/25
#dx = dy
#xRes,yRes = int((xmax-xmin)/dx),int((ymax-ymin)/dy)
#yResa,yResb =int((ymax-yint)/dy),int((yint-ymin)/dy)

xRes,yRes = 400,64

#tRatio =  int(sys.argv[1])
#tRatio = 100
#save_every = 1
use_fssa = False
use_coupling = True

if use_fssa:
    save_every = 1
    op_case_name = "withFSSA0.5_"
else:
    op_case_name = "noFSSA_"
    save_every = 1

if use_coupling :
    op_case_name += 'cm_'
else:
    op_case_name += 'tm_'

outputPath = "op_SedimentTest_" + op_case_name + "yres{:n}_uwg_Eulerain_evalfelev".format(yRes)

Model = GEO.Model(elementRes=(xRes, yRes),
                  minCoord=(xmin,ymin),
                  maxCoord=(xmax, ymax),
                  gravity=(0.0, -gravity))
Model.outputDir= outputPath
#Model.minStrainRate = 1e-18 / u.second


# In[3]:


def surfe_npfn(xx,x1,x2,h1,h2,wavel,A):
    yy = np.zeros_like(xx)
    yy = A*np.sin(-2*np.pi*xx/wavel) 
    yy[xx<x1] = h1
    yy[xx>x2] = h2
    return yy


def surfe_uwfn(x1,x2,h1,h2,wavel,A):
    surf_fn = fn.branching.conditional([( Model.x <= x1, h1),
                              ( Model.x >= x2, h2),
                              (True, A*fn.math.sin(-2*math.pi*Model.x/wavel))])  
    return surf_fn 


# In[4]:


npoints = 101
xx = np.linspace(xmin,xmax,npoints)

x1,x2 = ndim(-25*u.kilometer),ndim(25*u.kilometer)
h1,h2 = ndim(2.5*u.kilometer),ndim(-2.5*u.kilometer)
wavel = (x2-x1)*2
amplitude = ndim(2.5*u.kilometer)


yy = surfe_npfn(xx,x1,x2,h1,h2,wavel,amplitude)
surf_fn = surfe_uwfn(x1,x2,h1,h2,wavel,amplitude)

#plt.plot(xx,yy)


# In[5]:


materialAShape = Model.y > surf_fn
materialMShape = Model.y <= surf_fn

materialA = Model.add_material(name="Air", shape=materialAShape)
materialM = Model.add_material(name="Mantle", shape=materialMShape)
sediment = Model.add_material(name="sediment")

npoints = xRes*2+1
coords = np.zeros((npoints,2))
coords[:,0] = np.linspace(xmin,xmax,npoints)
coords[:,1] = surfe_npfn(coords[:,0],x1,x2,h1,h2,wavel,amplitude)
Model.add_passive_tracers('surf',vertices=coords)

Model.maxViscosity = 1e22 * u.pascal * u.second
Model.minViscosity = 1e18 * u.pascal * u.second
materialA.viscosity = 1e18 * u.pascal * u.second
materialM.viscosity = 1e20 * u.pascal * u.second
sediment.viscosity = 1e23 * u.pascal * u.second

# bulk_visc = 1e11 * u.pascal * u.second
# materialA.compressibility = 1/bulk_visc
# materialM.compressibility = 0.
# sediment.compressibility = 0.

materialA.density = 0.
materialM.density = 3300 * u.kilogram / u.metre**3
sediment.density = 2300 * u.kilogram / u.metre**3 

# density   =  3300 * u.kilogram / u.meter**3
# Lx        = ndim( 100e3 * u.meter)
# Ly        = ndim( 60e3 * u.meter)

# # traction perturbation parameters
# xp        = ndim( 50e3 * u.meter)
# width     = ndim( 3e3  * u.meter)

# # compute lithostatic load a prior
# lithostaticPressure = 0.6*Ly*ndim(density)*ndim(gravity)

# tractionField = Model.mesh.add_variable(nodeDofCount=2 )
# for ii in Model.bottom_wall:
#     coord = Model.mesh.data[ii]
#     tractionField.data[ii] = [0.0,lithostaticPressure*(1.+0.2*np.exp((-1/width*(coord[0]-xp)**2)))]

# bottomWall = Model.bottom_wall

# npoints = 1
# coords = np.zeros((npoints,2))
# coords[:,0] = 0.5*Lx
# coords[:,1] = 0.
# Model.add_passive_tracers('peak',vertices=coords)


# In[6]:


Model.set_velocityBCs(left=[0.,None],right=[0,None],bottom=[None,0.], top=[None, None])
#Model.set_stressBCs(bottom=[None, tractionField])


# In[7]:


import underworld.visualisation as vis
cm_per_year = dimen(1,u.centimeter/u.year)
fig1 = vis.Figure(title="Uplift map - scaled viz", figsize=(700,400), quality=2, rulers=True)
fig1.append( vis.objects.Points(Model.swarm, Model.materialField, fn_size=2.,colourBar = False  ) )
fig1.append( vis.objects.VectorArrows(Model.mesh, cm_per_year.magnitude*0.1*Model.velocityField) )
fig1.show()


# In[8]:


import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(12,5))
coords = Model.surf_tracers.data
ax.plot(coords[:,0],coords[:,1])
ax.set_xlim([-1,1])
ax.set_ylim([-0.03,0.03])


# In[9]:


# # record output
# outfile = open(Model.outputDir+'buildMount.txt', 'w+')
# string = "steps, timestep, vrms, peak height"
# if uw.mpi.rank==0:
#     print(string)
#     outfile.write( string+"\n")

# fn_y = fn.coord()[1]
# fn_y_minmax = fn.view.min_max(fn_y)

# fn_y_minmax.reset()
# fn_y_minmax.evaluate(Model.peak_tracers)
# h1 = fn_y_minmax.max_global()


# def print_info():
#     h1 = fn_y_minmax.max_global()
#     h0 = h1

#     # update peak heigh
#     fn_y_minmax.reset()
#     fn_y_minmax.evaluate(Model.peak_tracers)
#     h1 = fn_y_minmax.max_global()

#     diffH = h1-h0
#     string = "{}, {:.3e}, {:.3e}, {:.3e}".format(Model.step,
#                                      dimen(Model.dt.value, u.kiloyear),
#                                      dimen(Model.stokes_SLE.velocity_rms(), u.cm/u.year),
#                                      dimen(diffH, u.metre) )
#     if uw.mpi.rank == 0:
#         print(string)
#         outfile.write(string+"\n")

# Model.post_solve_functions["print_info"] = print_info


# In[10]:


Model.init_model()

if use_coupling:
    Model.surfaceProcesses = Badlands_evalfelev(airIndex=[materialA.index],sedimentIndex=sediment.index,XML="badlands.xml", resolution=0.5 * u.kilometre, checkpoint_interval=250*u.year,aspectRatio2d=0.25,surfElevation=surf_fn)

Model.solver.set_inner_method("mumps")

if use_fssa:
    Model._fssa_factor = 0.5


# In[11]:


Model.run_for(25000*u.year, checkpoint_interval=500*u.year,dt=250*u.year)


# In[ ]:


# #Model.run_for(nstep =3 ,checkpoint_interval=1)
# Model.run_for(1e6*u.year, checkpoint_interval=5e4*u.year,dt=5e4*u.year)


# In[ ]:


#Model.run_for(nstep =40,checkpoint_interval=1)


# In[ ]:


fig, ax = plt.subplots(1, 1, figsize=(12,5))
coords = Model.surf_tracers.data
ax.plot(coords[:,0],coords[:,1])
ax.set_xlim([-1,1])
ax.set_ylim([-0.03,0.03])


# In[ ]:





# In[ ]:




