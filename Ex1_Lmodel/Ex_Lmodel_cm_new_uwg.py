import underworld as uw
from underworld import function as fn
import math 
import numpy as np
import os

from underworld import UWGeodynamics as GEO
u = GEO.UnitRegistry
ndim = GEO.non_dimensionalise
dimen = GEO.dimensionalise

GEO.rcParams["popcontrol.particles.per.cell.2D"] = 16
GEO.rcParams["swarm.particles.per.cell.2D"] = 16
GEO.rcParams["surface.pressure.normalization"] = True
GEO.rcParams["pressure.smoothing"] = True
GEO.rcParams["popcontrol.split.threshold"] = 0.1

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

xmin, xmax = ndim(-64 * u.kilometer), ndim(64 * u.kilometer)
ymin, ymax = ndim(-32 * u.kilometer), ndim(32 * u.kilometer)
yint = 0.

dy = ndim(1.0 * u.kilometer)
dx = ndim(1.0 * u.kilometer)
xRes,yRes = int(np.around((xmax-xmin)/dx)),int(np.around((ymax-ymin)/dy))
yResa,yResb =int(np.around((ymax-yint)/dy)),int(np.around((yint-ymin)/dy))


use_coupling = True

if use_coupling :
    op_case_name = 'cm_'
else:
    op_case_name = 'tm_'

outputPath = "op_cascade_" + op_case_name + "yres{:n}_uwg_Eulerain".format(yRes)

Model = GEO.Model(elementRes=(xRes, yRes),
                  minCoord=(xmin,ymin),
                  maxCoord=(xmax, ymax),
                  gravity=(0.0, -gravity))
Model.outputDir= outputPath

def surfe_npfn(xx,x1,x2,h1,h2,wavel,A):
    yy = np.zeros_like(xx)
    #yy = h1
    yy[xx>=x2] = h2
    yy[xx<=x2] = h1
    return yy

def surfe_uwfn(x1,x2,h1,h2,wavel,A):
    surf_fn = fn.branching.conditional([( Model.x >= x2, h2),
                              (True, h1)])  
    return surf_fn 
    
npoints = xRes*2+1
xx = np.linspace(xmin,xmax,npoints)

x1,x2 = ndim(0*u.kilometer),ndim(0*u.kilometer)
h1,h2 = ndim(20.0*u.kilometer),ndim(-20.0*u.kilometer)
wavel = (x2-x1)*2
amplitude = h1

yy = surfe_npfn(xx,x1,x2,h1,h2,wavel,amplitude)
surf_fn = surfe_uwfn(x1,x2,h1,h2,wavel,amplitude)
surf_fn = surfe_uwfn(x1,x2,h1,h2,wavel,amplitude)


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

materialA.viscosity = 1e18 * u.pascal * u.second
materialM.viscosity = 1e21 * u.pascal * u.second
sediment.viscosity = 1e23 * u.pascal * u.second

materialA.density = 0.
materialM.density = 3300 * u.kilogram / u.metre**3
sediment.density = 2300 * u.kilogram / u.metre**3 

Model.set_velocityBCs(left=[0.,None],right=[0,None],bottom=[None,0.], top=[None, None])

Model.init_model()

dt_set =  500*u.year
max_time = 80000*u.year
checkpoint_interval = 500*u.year

if use_coupling:
    Model.surfaceProcesses = GEO.surfaceProcesses.Badlands(airIndex=[materialA.index],sedimentIndex=sediment.index,XML="badlands.xml", resolution=0.5 * u.kilometre, checkpoint_interval=dt_set ,aspectRatio2d=0.25,surfElevation=surf_fn)

Model.solver.set_inner_method("mumps")

#Model.run_for(nstep=2)
Model.run_for(max_time, checkpoint_interval=checkpoint_interval,dt=dt_set)