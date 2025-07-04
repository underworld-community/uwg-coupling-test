#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
from underworld import UWGeodynamics as GEO
import numpy as np
from underworld import visualisation as vis

GEO.rcParams["initial.nonlinear.tolerance"] = 1e-3
GEO.rcParams['initial.nonlinear.max.iterations'] = 100
GEO.rcParams["nonlinear.tolerance"] = 1e-3
GEO.rcParams['nonlinear.max.iterations'] = 100
GEO.rcParams["popcontrol.particles.per.cell.2D"] = 16
GEO.rcParams["swarm.particles.per.cell.2D"] = 16
GEO.rcParams["surface.pressure.normalization"] = True
GEO.rcParams["pressure.smoothing"] = True
GEO.rcParams["popcontrol.split.threshold"] = 0.1

u = GEO.UnitRegistry
ndim = GEO.non_dimensionalise
dimen = GEO.dimensionalise

velocity = 1 * u.centimeter / u.year
model_length = 100. * u.kilometer
bodyforce = 2700. * u.kilogram / u.metre**3 * 9.81 * u.meter / u.second**2

KL = model_length
Kt = KL / velocity
KM = bodyforce * KL**2 * Kt**2

GEO.scaling_coefficients["[length]"] = KL
GEO.scaling_coefficients["[time]"] = Kt
GEO.scaling_coefficients["[mass]"]= KM


xmin, xmax = ndim(0. * u.kilometer), ndim(128 * u.kilometer)
ymin, ymax = ndim(-7 * u.kilometer), ndim(9 * u.kilometer)
yint = 0.

dy = ndim(1.0 * u.kilometer/4)
dx = ndim(1.0 * u.kilometer)
xRes,yRes = int(np.around((xmax-xmin)/dx)),int(np.around((ymax-ymin)/dy))
yResa,yResb =int(np.around((ymax-yint)/dy)),int(np.around((yint-ymin)/dy))


Model = GEO.Model(elementRes=(xRes, yRes),
                  minCoord=(xmin,ymin),
                  maxCoord=(xmax, ymax),
                  gravity=(0.0, -9.81 * u.meter / u.second**2))

outputPath = "op_Wedge_FreeSurfEulerian_yres{:n}_uwg_cm".format(yRes)
Model.outputDir = outputPath

air_shape = GEO.shapes.Layer(top=Model.top, bottom=0.)
fricLayerShape = GEO.shapes.Layer(top=Model.bottom + ndim(1. * u.kilometer), bottom=Model.bottom + ndim(0.5 * u.kilometer))
rigidBaseShape = GEO.shapes.Layer(top=Model.bottom + ndim(0.5 * u.kilometer), bottom=Model.bottom)

air = Model.add_material(name="Air", shape=air_shape)
frictionalBasal = Model.add_material(name="Frictional", shape=fricLayerShape)
rigidBase = Model.add_material(name="Frictional", shape=rigidBaseShape)
sediment = Model.add_material(name="Sediment")

top_pile = 0.
bottom_pile = -ndim(6.0 * u.kilometer) 

NLayers = 12
layer_thickness = (top_pile - bottom_pile) / NLayers

plastic_pile = []

layer_above = air_shape

for index in range(NLayers):
    shape = GEO.shapes.Layer(top=layer_above.bottom, bottom=layer_above.bottom - layer_thickness)
    material = Model.add_material(name="Plastic {0}".format(index), shape=shape)
    plastic_pile.append(material)
    layer_above = shape

Model.density = 2700 * u.kilogram / u.metre**3
Model.viscosity = 1e23 * u.pascal * u.second
Model.maxViscosity = 1e23 * u.pascal * u.second
Model.minViscosity = 5e19 * u.pascal * u.second

air.viscosity = 1e19 * u.pascal * u.second
air.minViscosity = 1e19 * u.pascal * u.second
air.density = 1. * u.kilogram / u.metre**3

# Note that this is not necessary as this does not differ from the
# Model property.
for material in plastic_pile:
    material.density = 2700 * u.kilogram / u.metre**3
    material.viscosity = 1e23 * u.pascal * u.second


frictionalBasal.viscosity = 1e23 * u.pascal * u.second
rigidBase.viscosity = 1e23 * u.pascal * u.second

plastic_Law = GEO.DruckerPrager(
        cohesion=20. * u.megapascal,
        cohesionAfterSoftening=4. * u.megapascal,
        frictionCoefficient=np.tan(np.radians(25.0)),
        frictionAfterSoftening=np.tan(np.radians(20.0)),
        epsilon1=0.01,
        epsilon2=0.06
    )

for material in plastic_pile:
    material.plasticity = plastic_Law

sediment.density = 2700 * u.kilogram / u.metre**3
sediment.viscosity = 1e23 * u.pascal * u.second
sediment.plasticity = plastic_Law

frictionalBasal.plasticity = GEO.DruckerPrager(
    cohesion=0.1 * u.megapascal,
    frictionCoefficient=np.tan(np.radians(12.0)),
    frictionAfterSoftening=np.tan(np.radians(6.0)),
    epsilon1=0.01,
    epsilon2=0.06
)

import underworld.function as fn

tapeL=frictionalBasal
flthick=GEO.nd(tapeL.top-tapeL.bottom)

conditions = [(Model.y <= GEO.nd(rigidBase.top), GEO.nd(-velocity)),
              (Model.y < GEO.nd(tapeL.top),
               GEO.nd(-velocity)*(flthick-(Model.y-GEO.nd(tapeL.bottom)))/flthick),
              (True, GEO.nd(0. * u.centimeter / u.year))]

fn_condition = fn.branching.conditional(conditions)

Model.set_velocityBCs(left=[fn_condition, 0.],
                      right=[-velocity, None],
                      top=[None, None],
                      bottom=[-velocity, 0.])

 
Model.solver.set_inner_method("mumps")
#Model.solver.set_penalty(1e6)

Model.init_model(pressure="lithostatic")

dt_set =  5.0*u.kiloyear
max_time = 1e3*u.kiloyear
checkpoint_interval = 1e2*u.kiloyear

Model.surfaceProcesses = GEO.surfaceProcesses.Badlands(airIndex=[air.index],sedimentIndex=sediment.index,XML="badlands.xml", resolution=0.5 * u.kilometre, checkpoint_interval=dt_set,aspectRatio2d=0.25,surfElevation=0.)

Model.run_for(max_time, checkpoint_interval=checkpoint_interval,dt=dt_set)
