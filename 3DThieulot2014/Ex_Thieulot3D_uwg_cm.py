#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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

# solver parameters
GEO.rcParams["initial.nonlinear.tolerance"] = 1e-2
GEO.rcParams['initial.nonlinear.max.iterations'] = 100
GEO.rcParams["nonlinear.tolerance"] = 1e-2
GEO.rcParams['nonlinear.max.iterations'] = 50
GEO.rcParams["popcontrol.particles.per.cell.3D"] = 40
GEO.rcParams["swarm.particles.per.cell.3D"] = 40
GEO.rcParams["surface.pressure.normalization"] = True
GEO.rcParams["pressure.smoothing"] = True
GEO.rcParams["popcontrol.split.threshold"] = 0.10


# scaling 3: vel
half_rate = (0.5 * u.centimeter / u.year).to(u.meter / u.second)
model_length = 192e3 * u.meter
model_width = 64e3 * u.meter
refViscosity = (1e24 * u.pascal * u.second).to_base_units()
surfaceTemp = ((0.+273.15) * u.degK).to_base_units()
baseModelTemp = ((550.+273.15)  * u.degK).to_base_units()
bodyforce = (2800 * u.kilogram / u.metre**3 * 9.81 * u.meter / u.second**2)

KL_meters = model_length
KT_seconds = KL_meters / half_rate
KM_kilograms = bodyforce * KL_meters**2 * KT_seconds**2
Kt_degrees = (baseModelTemp - surfaceTemp)
K_substance = 1. * u.mole

GEO.scaling_coefficients["[length]"] = KL_meters 
GEO.scaling_coefficients["[time]"] = KT_seconds
GEO.scaling_coefficients["[mass]"]= KM_kilograms 
GEO.scaling_coefficients["[temperature]"] = Kt_degrees
GEO.scaling_coefficients["[substance]"] = K_substance

gravity = ndim(9.81 * u.meter / u.second**2)
R = ndim(8.3144621 * u.joule / u.mole / u.degK)

#xRes, yRes,zRes = 95,43,13  # paper for 192km, 64km, 28km
#xRes, yRes,zRes = 96,32,20  # hpc
xRes, yRes,zRes= 48,16,10 # local

xmin, xmax = ndim(0. * u.kilometer), ndim(192. * u.kilometer)
ymin, ymax = ndim(0. * u.kilometer), ndim(64. * u.kilometer)
zmin, zmax = ndim(-28 * u.kilometer), ndim(12 * u.kilometer)
zint = 0.

dy = (xmax-xmin)/xRes
dx = (ymax-ymin)/yRes
dz = (zmax-zmin)/zRes

outputPath = "op_Ex_Thieulot2D_uwg_zres{:n}_cm".format(zRes)
Model = GEO.Model(elementRes=(xRes,yRes,zRes),
                  minCoord=(xmin,ymin,zmin),
                  maxCoord=(xmax,ymax,zmax),
                  gravity=(0.0,0.0, -gravity))
Model.outputDir= outputPath 

air        = Model.add_material(name="Air", shape=GEO.shapes.Layer3D(top=Model.top, bottom=0.0 * u.kilometer))
uppercrust = Model.add_material(name="UppperCrust", shape=GEO.shapes.Layer3D(top=0.0 * u.kilometer, bottom=-14.0 * u.kilometer))
lowercrust = Model.add_material(name="LowerCrust", shape=GEO.shapes.Layer3D(top=-14.0 * u.kilometer, bottom=-28.0 * u.kilometer))
sediment   = Model.add_material(name="Sediment")
# npoints = 101
# coords = np.ndarray((npoints, 2))
# coords[:, 0] = np.linspace(GEO.nd(Model.minCoord[0]), GEO.nd(Model.maxCoord[0]), npoints)
# coords[:, 1] = GEO.nd(uppercrust.top)
# surf_tracers = Model.add_passive_tracers(name="Surface", vertices=coords)

# In[12]

# if size == 1:
#     import underworld.visualisation as vis
#     figsize = (800,400)
#     camera = ['rotate x 30']
#     Fig = vis.Figure(resolution=figsize,rulers=False,margin = 80,axis=False)
#     Fig.Points(Model.swarm, Model.materialField,fn_size=2.0,discrete=True,colourBar=True) #colours='blue orange')
#     #Fig.Mesh(Model.mesh)
#     lv = Fig.window() 
#     lv.rotate('z',45)
#     lv.rotate('x',-60)
#     lv.redisplay()
# In[13]:


air.density =  1. * u.kilogram / u.metre**3 
# uppercrust.density =  2800. * u.kilogram / u.metre**3 
# lowercrust.density =  2800. * u.kilogram / u.metre**3 
# sediment.density =  2800. * u.kilogram / u.metre**3 
# air.thermalExpansivity = 0.0
# uppercrust.thermalExpansivity = 2.5e-5 / u.kelvin
# lowercrust.thermalExpansivity = 2.5e-5 / u.kelvin
# sediment.thermalExpansivity = 2.5e-5 / u.kelvin 

uppercrust.density = GEO.LinearDensity(2800. * u.kilogram / u.metre**3,thermalExpansivity= 2.5e-5 * u.kelvin**-1,reference_temperature= 273.15*u.kelvin)
lowercrust.density = GEO.LinearDensity(2800. * u.kilogram / u.metre**3,thermalExpansivity= 2.5e-5 * u.kelvin**-1,reference_temperature= 273.15*u.kelvin)
sediment.density = 2300. * u.kilogram / u.metre**3

Model.diffusivity      = 1.0e-6 * u.metre**2 / u.second 
air.diffusivity        = 1.0e-6 * u.metre**2 / u.second
uppercrust.diffusivity = 1.0e-6 * u.metre**2 / u.second
lowercrust.diffusivity = 1.0e-6 * u.metre**2 / u.second
sediment.diffusivity   = 1.0e-6 * u.metre**2 / u.second

Model.capacity    = 803.57 * u.joule / (u.kelvin * u.kilogram)  
air.capacity = 100. * u.joule / (u.kelvin * u.kilogram)
uppercrust.capacity  = 803.57 * u.joule / (u.kelvin * u.kilogram) 
lowercrust.capacity  = 803.57 * u.joule / (u.kelvin * u.kilogram) 
sediment.capacity    = 803.57 * u.joule / (u.kelvin * u.kilogram)  

# uppercrust.capacity = k/uppercrust.diffusivity/uppercrust.density
# lowercrust.capacity = k/lowercrust.diffusivity/lowercrust.density
# sediment.capacity = k/sediment.diffusivity/sediment.density     

air.radiogenicHeatProd = 0.0
uppercrust.radiogenicHeatProd = 0.9 * u.microwatt / u.meter**3
lowercrust.radiogenicHeatProd = 0.9 * u.microwatt / u.meter**3
sediment.radiogenicHeatProd   = 0.6 * u.microwatt / u.meter**3

Model.maxViscosity = 1e25 * u.pascal * u.second
Model.minViscosity = 1e19 * u.pascal * u.second

air.viscosity = 1e19 * u.pascal * u.second

viscosity = GEO.ViscousCreep(name="Powerlaw",preExponentialFactor=1.10e-28/u.pascal ** 4.0 /u.second,
                              stressExponent=4.0,
                              activationVolume=0.,
                              activationEnergy=223 * u.kilojoules,
                              f=1.0)
# rh = GEO.ViscousCreepRegistry()
# viscosity = 1 * rh.Wet_Quartz_Dislocation_Gleason_and_Tullis_1995
uppercrust.viscosity = viscosity
lowercrust.viscosity = viscosity
sediment.viscosity   = viscosity


# Plastic rheology
# Huismans, R. S. and C. Beaumont (2007), Roles of lithospheric strain softening and heterogeneity in determining the geometry of rifts and continental margins, in G. D. Karner, G. Manatschal, and L. M. Pinheiro, Imaging, Mapping and Modelling Continental Lithosphere Exten- sion and Breakup, Geol. Soc. Spec. Publ., 282, 111–138, doi:10.1144/SP282.6.
# pl = GEO.PlasticityRegistry()
# pl.Huismans_et_al_2011_Crust

frictionCoefficient1,frictionCoefficient2 = math.tan(np.deg2rad(15)),math.tan(np.deg2rad(2))
#frictionCoefficient1,frictionCoefficient2 = 0.017,0.123
plasticity = GEO.DruckerPrager(name="Huismans2007",cohesion=20.0 * u.megapascal,
                               cohesionAfterSoftening=10 * u.megapascal,
                               frictionCoefficient=frictionCoefficient1,
                               frictionAfterSoftening=frictionCoefficient2,
                               epsilon1=0.25,
                               epsilon2=1.25)
uppercrust.plasticity  = plasticity
lowercrust.plasticity  = plasticity
sediment.plasticity    = plasticity

Model.set_temperatureBCs(top=surfaceTemp , 
                         bottom=baseModelTemp, 
                         materials=[(air, 273.15 * u.degK)])

Model.set_velocityBCs(left=[half_rate, None, None],
                       right=[-half_rate, None,None],
                      front= [None,0.,None],
                      back = [None,0.,None], 
                       bottom = [None,None,0.])
#                        bottom=GEO.LecodeIsostasy(reference_mat=mantle, average=True))

Model.init_model(temperature='steady-state',pressure='lithostatic')

# Model.plasticStrain.data[...] = 0.
# xx = Model.swarm.particleCoordinates.data[:,0]
# yy = Model.swarm.particleCoordinates.data[:,-1]
# Lx = xmax-xmin
# Lz = 0.-zmin
# z1_index = np.where(Model.swarm.particleCoordinates.data[:,-1]>0.)
# z2_index = np.where(Model.swarm.particleCoordinates.data[:,-1]<=0.)
# xx = Model.swarm.particleCoordinates.data[z2_index,0]
# zz = Model.swarm.particleCoordinates.data[z2_index,-1]
# z1 = 0.
# z2 = (1 - np.cos(2.0 * np.pi * xx / Lx))**4 * (1-np.cos(2.0*np.pi * zz / Lz)) 
# Model.plasticStrain.data[z1_index,0] = z1
# Model.plasticStrain.data[z2_index,0] = z2


# fn_minmax = fn.view.min_max(Model.plasticStrain)
# fn_minmax.evaluate(Model.swarm)

# Model.plasticStrain.data[...] = Model.plasticStrain.data[...]/ fn_minmax.max_global() * 1.75
# Model.plasticStrain.data[...] *= np.random.rand(*Model.plasticStrain.data.shape[:])

Model.plasticStrain.data[:,0] = 0.
Model.plasticStrain.data[:,0] = np.random.rand(Model.plasticStrain.data.size)
Model.plasticStrain.data[:,0] *= (1.0 - np.cos(2.0*np.pi*Model.swarm.particleCoordinates.data[:,0]/(GEO.nd(Model.maxCoord[0] - Model.minCoord[0]))))**4
Model.plasticStrain.data[:,0] *= (1.0 - np.cos(2.0*np.pi*Model.swarm.particleCoordinates.data[:,2]/ GEO.nd(0.-Model.bottom)))
Model.plasticStrain.data[:,0] /= 10 # It looks like there is a factor 10 missing in the paper
Model.plasticStrain.data[Model.swarm.particleCoordinates.data[:,2] > GEO.nd(0.)] = 0.


# import underworld.visualisation as vis
# Fig = vis.Figure(figsize=(1200,400))
# Fig.Surface(Model.mesh, Model.projPlasticStrain,colourBar=False)
# Fig.show()

Model.solver.set_inner_method("mumps")
Model.solver.set_penalty(1.0e6)
# GEO.rcParams["CFL"] = 0.2/2

Model.surfaceProcesses = GEO.surfaceProcesses.Badlands(airIndex=[air.index],sedimentIndex=sediment.index,XML="badlands.xml", resolution=2. * u.kilometre, checkpoint_interval=10*u.kiloyear,aspectRatio2d=0.25)

#Model.checkpoint(0) 

#Model.run_for(0.10 * u.megayear, checkpoint_interval=0.05 * u.megayear,dt=2500*u.kiloyear) #,dt=dt)
Model.run_for(5.01 * u.megayear, checkpoint_interval=0.1 * u.megayear,dt=10*u.kiloyear)


# In[ ]:




