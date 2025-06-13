#!/usr/bin/env python
# coding: utf-8

# In[1]:


from underworld import UWGeodynamics as GEO
from underworld import visualisation as vis


# In[2]:


u = GEO.UnitRegistry


# In[3]:


# build reference units
KL_meters   = 100e3 * u.meter
K_viscosity = 1e16 * u.pascal * u.second
K_density   = 3.3e3 * u.kilogram / (u.meter) ** 3

# compute dependent scaling units
KM_kilograms = K_density * KL_meters**3
KT_seconds   = KM_kilograms / (KL_meters * K_viscosity)
K_substance  = 1.0 * u.mole
Kt_degrees   = 1.0 * u.kelvin

# Characteristic values of the system
GEO.scaling_coefficients["[length]"] = KL_meters.to_base_units()
GEO.scaling_coefficients["[temperature]"] = Kt_degrees.to_base_units()
GEO.scaling_coefficients["[time]"] = KT_seconds.to_base_units()
GEO.scaling_coefficients["[mass]"] = KM_kilograms.to_base_units()


# In[4]:


Model = GEO.Model(
    elementRes=(100, 60),
    minCoord=(0 * u.km, 0 * u.km),
    maxCoord=(100 * u.km, 60 * u.km),
)


# In[5]:


Model.outputDir = "1_12_Uplift_TractionBCs"


# In[6]:


layer_h = 0.6 * Model.maxCoord[1]


# In[7]:


# lithostaticPressure = GEO.nd(-Model.gravity[1] * K_density * layer_h)
lithostaticPressure = -Model.gravity[1] * K_density * layer_h


# In[8]:


air = Model.add_material(
    name="air", shape=GEO.shapes.Layer2D(top=Model.top, bottom=0.0)
)
background = Model.add_material(
    name="background", shape=GEO.shapes.Layer2D(top=layer_h, bottom=Model.bottom)
)


# In[9]:


# if GEO.size == 1:
#     Fig = vis.Figure(figsize=(1200,400))
#     Fig.Points(Model.swarm, Model.materialField, fn_size=2.0)
#     Fig.save("Figure_1.png")
#     Fig.show()


# In[10]:


air.density        = 0.0 * u.kilogram / u.metre**3
background.density = 3300.0 * u.kilogram / u.metre**3

air.viscosity        = 1e22 * u.pascal * u.second
background.viscosity = 1e22 * u.pascal * u.second

air.compressibility        = 1.0 / (1e11 * u.Pa * u.sec)
background.compressibility = 0.0


# In[11]:


# if GEO.size == 1:
#     Fig = vis.Figure(figsize=(1200,400))
#     Fig.Points(Model.swarm, Model.materialField, fn_size=2)
#     # Fig.Mesh(Model.mesh)
#     Fig.VectorArrows(Model.mesh, Model.velocityField)
#     Fig.show()


# In[12]:


# traction perturbation parameters
import numpy as np

xp = GEO.nd(50e3 * u.meter)
width = GEO.nd(3e3 * u.meter)
for ii in Model.bottom_wall:
    coord = Model.mesh.data[ii]
    P = lithostaticPressure * (1.0 + 0.2 * np.exp((-1 / width * (coord[0] - xp) ** 2)))
    Model.tractionField.data[ii] = [
        0.0,
        GEO.nd(P),
    ]  # important to non dimensionalise this


# In[13]:


# # visualise the bottom stress condition
# if GEO.size == 1:
#     GEO.uw.utils.matplotlib_inline()
#     import matplotlib.pyplot as pyplot
#     import matplotlib.pylab as pylab
#     pyplot.ion()
#     pylab.rcParams[ 'figure.figsize'] = 12, 6
#     pyplot.title('Prescribed traction component normal to base wall')
#     km_scaling  = GEO.dimensionalise(1,u.kilometer)
#     MPa_scaling = GEO.dimensionalise(1,u.MPa)
#     pyplot.xlabel('X coordinate - (x{}km)'.format(km_scaling.magnitude))
#     pyplot.ylabel('Normal basal traction MPa - (x{:.3e}MPa)'.format(MPa_scaling.magnitude))

#     xcoord = Model.mesh.data[Model.bottom_wall.data][:,0]          # x coordinate
#     stress = Model.tractionField.data[Model.bottom_wall.data][:,1] # 2nd component of the traction along the bottom wall

#     pyplot.plot( xcoord, stress, 'o', color = 'black', label='numerical')
#     pyplot.show()


# In[14]:


Model.set_velocityBCs(left=[0.0, None], right=[0.0, None], top=[None, 0.0])
Model.set_stressBCs(bottom=[0.0, Model.tractionField[1]])


# In[15]:


Model.run_for(nstep=10)


# In[16]:


# if GEO.size==1:
#     Fig = vis.Figure(figsize=(1200,400))
#     Fig.Points(Model.swarm, Model.materialField, fn_size=2.0)
#     # Fig.Mesh(Model.mesh)
#     Fig.VectorArrows(Model.mesh, Model.velocityField)
#     Fig.show()


# In[17]:


# TODO, better test for model
if not np.isclose(Model.stokes_SLE.velocity_rms(), 1.3497493646857656e-18):
    raise RuntimeError("Velocity is not as expected")


# In[ ]:
