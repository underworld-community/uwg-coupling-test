#!/usr/bin/env python
# coding: utf-8

import underworld as uw
import math
from underworld import function as fn
import numpy as np
import os

from underworld import UWGeodynamics as GEO
u = GEO.UnitRegistry
ndim = GEO.non_dimensionalise
dimen = GEO.dimensionalise

comm = uw.mpi.comm
rank = uw.mpi.rank
size = uw.mpi.size

GEO.rcParams["initial.nonlinear.tolerance"] = 1e-3
GEO.rcParams['initial.nonlinear.max.iterations'] = 100
GEO.rcParams["nonlinear.tolerance"] = 1e-3
GEO.rcParams['nonlinear.max.iterations'] = 100
GEO.rcParams["popcontrol.particles.per.cell.2D"] = 16
GEO.rcParams["swarm.particles.per.cell.2D"] = 16
GEO.rcParams["surface.pressure.normalization"] = True
GEO.rcParams["pressure.smoothing"] = True
GEO.rcParams["popcontrol.split.threshold"] = 0.1

half_rate = 1.0 * u.centimeter / u.year
model_length = 500. * u.kilometer
gravity = 9.81 * u.meter / u.second**2
bodyforce = 3300 * u.kilogram / u.metre**3 *gravity 

KL = model_length
Kt = KL / half_rate
KM = bodyforce * KL**2 * Kt**2

GEO.scaling_coefficients["[length]"] = KL
GEO.scaling_coefficients["[time]"] = Kt
GEO.scaling_coefficients["[mass]"]= KM


xmin, xmax = ndim(-250 * u.kilometer), ndim(250 * u.kilometer)
ymin, ymax = ndim(-500 * u.kilometer), ndim(100 * u.kilometer)
yint = 0.

dy = ndim(500 * u.kilometer)/50
dx = dy
xRes,yRes = int((xmax-xmin)/dx),int((ymax-ymin)/dy)
yResa,yResb =int((ymax-yint)/dy),int((yint-ymin)/dy)

#tRatio =  int(sys.argv[1])
tRatio = 100
save_every = 1
use_fssa = False
if use_fssa:
    outputPath = "op_Ex_2DTopoRelax_FreeSurf_Eulerian_withFSSA0.5_yres{:n}_tRatio{:n}_Swarm_uwg".format(yRes,tRatio)
    save_every = 1
else:
    outputPath = "op_Ex_2DTopoRelax_FreeSurf_Eulerian_noFSSA_yres{:n}_tRatio{:n}_Swarm_uwg".format(yRes,tRatio)
    save_every = 5


Model = GEO.Model(elementRes=(xRes, yRes),
                  minCoord=(xmin,ymin),
                  maxCoord=(xmax, ymax),
                  gravity=(0.0, -gravity),
                  periodic=(True, False))
Model.outputDir= outputPath
Model.minStrainRate = 1e-18 / u.second

wRatio = 1
D = np.abs(ymin)
Lambda = D/wRatio
k = 2.0 * np.pi / Lambda
mu0 = ndim(1e21  * u.pascal * u.second)
g = ndim(gravity)
rho0 = ndim(3300* u.kilogram / u.metre**3)
drho = rho0-0.
w_m = ndim(10*u.kilometer)

tau0 = 2*k*mu0/drho/g
tau = (D*k+np.sinh(D*k)*np.cosh(D*k))/(np.sinh(D*k)**2)*tau0

def perturbation(x):
    return w_m * np.cos(2.*np.pi*(x)/Lambda)

fn_coord = fn.input()
deform_fn = w_m * fn.math.cos(2.*np.pi*fn_coord[0]/Lambda)

# def get_analytical(x0,load_time):
#     #tau = (D*k+np.sinh(D*k)*np.cosh(D*k))/(np.sinh(D*k)**2)*tau0
#     #A = -F0/k/tau0
#     #B = -F0/k/tau
#     #C = F0/tau
#     #E = F0/tau/np.tanh(D*k)
#     #phi = np.sin(k*x)*np.exp(-tmax/tau)*(A*np.sinh(k*z)+B*np.cosh(k*z)+C*z*np.sinh(k*z)+E*z*np.cosh(k*z))
#     w = w_m*np.exp(-load_time/tau)
#     return w

max_time =  tau*4
dt_set = tau/tRatio


materialAShape = fn_coord[1] > deform_fn #GEO.shapes.Layer(top=Model.top, bottom=Model.bottom)
materialMShape = fn_coord[1] <= deform_fn

materialA = Model.add_material(name="Air", shape=materialAShape)
materialM = Model.add_material(name="Mantle", shape=materialMShape)


npoints = xRes*2+1
coords = np.zeros((npoints,2))
coords[:,0] = np.linspace(xmin,xmax,npoints)
coords[:,1] = perturbation(coords[:,0])

Model.add_passive_tracers('surf',vertices=coords)


# if uw.mpi.rank == 0:
#     from underworld import visualisation as vis
#     fig_res = (500,500)

#     Fig = vis.Figure(resolution=fig_res,rulers=False,margin = 20,rulerticks=7,quality=2,clipmap=False) 
#     Fig.Points(Model.swarm, Model.materialField,fn_size=2.0,discrete=True,colourBar=True,colours='blue orange')
#     Fig.Mesh(Model.mesh)
#     Fig.show()
#     Fig.save("Modelsetup.png")


# Model.maxViscosity = 1e21 * u.pascal * u.second
# Model.minViscosity = 1e18 * u.pascal * u.second

materialA.viscosity = 1e18 * u.pascal * u.second
materialM.viscosity = 1e21 * u.pascal * u.second

materialA.density = 0.
materialM.density = 3300 * u.kilogram / u.metre**3


Model.set_velocityBCs(left=[0.,None],right=[0,None],bottom=[0.,0.], top=[None, 0.])

Model.init_model()

#Model.surfaceProcesses = GEO.surfaceProcesses.Badlands(airIndex=[air.index],sedimentIndex=sediment.index,XML="badlands.xml", resolution=1. * u.kilometre, checkpoint_interval=0.1 * u.megayears,aspectRatio2d=0.25)


Model.solver.set_inner_method("mumps")
Model.solver.set_penalty(1e3)


if use_fssa:
    Model._fssa_factor = 0.5


max_time = dimen(tau*4,u.kiloyear)
dt_set = dimen(tau/tRatio,u.kiloyear)
checkpoint_interval = dt_set*save_every 

Model.run_for(max_time, checkpoint_interval=checkpoint_interval,dt=dt_set)

