from __future__ import print_function,  absolute_import
import abc
import underworld as uw
import underworld.function as fn
import numpy as np
import sys
import math
from scipy.ndimage.filters import gaussian_filter
from scipy.interpolate import griddata, interp1d
from underworld.scaling import non_dimensionalise as nd
from underworld.scaling import dimensionalise
from underworld.scaling import units as u
from tempfile import gettempdir

comm = uw.mpi.comm
rank = uw.mpi.rank
size = uw.mpi.size

ABC = abc.ABCMeta('ABC', (object,), {})
_tempdir = gettempdir()


class SurfaceProcesses(ABC):

    def __init__(self, Model=None):

        self.Model = Model

    @property
    def Model(self):
        return self._Model

    @Model.setter
    def Model(self, value):
        self._Model = value
        if value:
            self._init_model()

    @abc.abstractmethod
    def _init_model(self):
        pass

    @abc.abstractmethod
    def solve(self, dt):
        pass


class Badlands(SurfaceProcesses):
    """ A wrapper class for Badlands """

    def __init__(self, airIndex,
                 sedimentIndex, XML, resolution, checkpoint_interval,
                 surfElevation=0., verbose=True, Model=None, outputDir="outbdls",
                 restartFolder=None, restartStep=None, timeField=None,
                 minCoord=None, maxCoord=None, aspectRatio2d=1.):
        """
        Creates a SurfaceProcesses object to encapsulate a Badlands model.
        The arguments for this function will override, or change the behaviour, of the
        badlands .xml file that usually parametises a Badlands model.
        Note Badlands is only made available on processor rank 0 in parallel. Information is sent
        to proc 0, computed via Badlands, and then broadcast to all procs for Underworld processing.

        Paramemters
        -----------

            XML : str
                The xml file that parametrises the Badlands model to be coupled with UWGeo.
                Note aguments defined here will override the values of the xml file.
            resolution : int array
                The resolution of the Badlands DEM.
            checkpoint_interval : 
                Overwrites the tDisplay value in Badlands. 
            surfElevation : underworld.function, defaults to 0
                Sets the initial Z coordinate of Badland's dem mesh to a given value.
            verbose :
            Model : UWGeo.Model, optional
                The UWGeodynamics Model object
            outputDir : str, default 'outbdls'
                The output directory of the Badlands execution.
            restartStep : int, optional 
                If defined, it will activate Badlands to restart,
                Using this value as the step number to restart from the `restartFoler` dir defined above. 
            restartFolder : str
                The restart folder when restart enabled.
            timeField : Underworld Swarm variable fields
                The time field used to measure sedimentation timing of underworld particles. (Check if actually used not used in coupling)
            minCoord : float array, optional
                The min coordinates of the Badland's dem, if `None` use the UWGeodynamics model
            max Coord : float array, optional
                The max coordinates of the Badland's dem, if `None` use the UWGeodynamics model
            aspectRatio2D : float
                Always used in 2D to set the seondary coordinate min/max as a mulitple of the first component.
        """
        try:
            import badlands

        except ImportError:
            raise ImportError("""badlands import as failed. Please check your
                              installation, PYTHONPATH and PATH environment
                              variables""")

        self.verbose = verbose
        self.outputDir = outputDir
        self.restartStep = restartStep
        self.restartFolder = restartFolder

        self.airIndex = airIndex
        self.sedimentIndex = sedimentIndex
        self.resolution = nd(resolution)
        self.surfElevation = fn.Function.convert(nd(surfElevation))
        self.checkpoint_interval = nd(checkpoint_interval)
        self.timeField = timeField
        self.XML = XML
        self.time_years = 0.
        self.minCoord = minCoord
        self.maxCoord = maxCoord
        self.aspectRatio2d = aspectRatio2d
        self.Model = Model

    def _init_model(self):

        if self.minCoord:
            self.minCoord = tuple([nd(val) for val in self.minCoord])
        else:
            self.minCoord = self.Model.mesh.minCoord

        if self.maxCoord:
            self.maxCoord = tuple([nd(val) for val in self.maxCoord])
        else:
            self.maxCoord = self.Model.mesh.maxCoord

        if self.Model.mesh.dim == 2:
            self.minCoord = (self.minCoord[0], self.aspectRatio2d*self.minCoord[0])
            self.maxCoord = (self.maxCoord[0], self.aspectRatio2d*self.maxCoord[0])

        if rank == 0:
            from badlands.model import Model as BadlandsModel
            self.badlands_model = BadlandsModel()
            self.badlands_model.load_xml(self.XML)

            if self.restartStep:
                # this will kick off internal restart code in Badlands that overwrites the _demfile
                # made below. See Badlands documentation for details.
                self.badlands_model.input.restart = True
                self.badlands_model.input.rstep = self.restartStep
                self.badlands_model.input.rfolder = self.restartFolder
                self.badlands_model.input.outDir = self.restartFolder
                self.badlands_model.outputStep = self.restartStep

                # Parse xmf for the last timestep time
                import xml.etree.ElementTree as etree
                xmf = (self.restartFolder +
                       "/xmf/tin.time" +
                       str(self.restartStep) + ".xmf")
                tree = etree.parse(xmf)
                root = tree.getroot()
                self.time_years = float(root[0][0][0].attrib["Value"])

            # Create Initial DEM
            self._demfile = _tempdir + "/dem.csv"
            self.dem = self._generate_dem()
            np.savetxt(self._demfile, self.dem)

            # Build Mesh
            self.badlands_model._build_mesh(self._demfile, verbose=False)

            self.badlands_model.input.outDir = self.outputDir
            self.badlands_model.input.disp3d = True  # enable 3D displacements
            self.badlands_model.input.region = 0  # TODO: check what this does
            self.badlands_model.input.tStart = self.time_years
            self.badlands_model.tNow = self.time_years

            # Override the checkpoint/display interval in the Badlands xml file.
            self.badlands_model.input.tDisplay = (
                dimensionalise(self.checkpoint_interval, u.years).magnitude)

            # Set Badlands minimal distance between nodes before regridding
            self.badlands_model.force.merge3d = (
                self.badlands_model.input.Afactor *
                self.badlands_model.recGrid.resEdges * 0.5)

            # Bodge Badlands to perform an initial checkpoint
            # FIXME: we need to run the model for at least one
            # iteration before this is generated.
            # It would be nice if this wasn't the case.
            self.badlands_model.force.next_display = 0

        comm.Barrier()

        self._disp_inserted = False

        # Transfer the initial DEM state to Underworld
        self._update_material_types()
        comm.Barrier()

    def _generate_dem(self):
        """
        Generate a badlands DEM. This can be used as the initial Badlands state.

        """

        # Calculate number of nodes from required resolution.
        nx = np.int32((self.maxCoord[0] - self.minCoord[0]) / self.resolution)
        ny = np.int32((self.maxCoord[1] - self.minCoord[1]) / self.resolution)
        nx += 1
        ny += 1

        x = np.linspace(self.minCoord[0], self.maxCoord[0], nx)
        y = np.linspace(self.minCoord[1], self.maxCoord[1], ny)

        coordsX, coordsY = np.meshgrid(x, y)

        dem = np.zeros((nx * ny, 3))
        dem[:, 0] = coordsX.flatten()
        dem[:, 1] = coordsY.flatten()

        coordsZ = self.surfElevation.evaluate(dem[:, :2])

        dem[:, 2] = coordsZ.flatten()
        return dimensionalise(dem, u.meter).magnitude

    def solve(self, dt, sigma=0):
        """
        Execute Badlands a badlands solve in the Underworld coupling.
            1. Collect Badland's recGrid and broadcast to all procs.
            2. Inerpolate Underworld velocity field on recGrid
            3. Calculate overall displacement in meters by muliplying velocity (m/yr) with input dt (yr).
            4. Using the displacement run badlands from currect time `t` to time `t+dt`.
            5. Save final stratigraphic field from badlands.
            6. Update Underworld particles depending on Badland tin. Another interpolation. 
        """

        dt_years = np.round(dimensionalise(dt, u.years).magnitude,6)  # fix pint scaling issue 

        if rank == 0 and self.verbose:
            purple = "\033[0;35m"
            endcol = "\033[00m"
            print(purple + f"Processing surface with Badlands {dt_years}:\n\t from {self.time_years} -> {self.time_years+dt_years}" + endcol)
            sys.stdout.flush()

        fact = dimensionalise(1.0, u.meter).magnitude
        if self.Model.mesh.dim == 2:
            known_xy = None
            known_z = None
            xs = None
            ys = None

            if rank == 0:
                known_xy = self.badlands_model.recGrid.tinMesh['vertices'] / fact 
                known_z = self.badlands_model.elevation / fact
                xs = self.badlands_model.recGrid.regX / fact
                ys = self.badlands_model.recGrid.regY / fact
            
            known_xy = comm.bcast(known_xy, root=0)
            known_z = comm.bcast(known_z, root=0)
            xs = comm.bcast(xs, root=0)
            ys = comm.bcast(ys, root=0)

            comm.Barrier()

            grid_x, grid_y = np.meshgrid(xs, ys)
            interpolate_z = griddata(known_xy,
                                     known_z,
                                     (grid_x, grid_y),
                                     method='nearest').T
            interpolate_z = interpolate_z.mean(axis=1)
            nd_coords = np.column_stack((xs, interpolate_z))

        if self.Model.mesh.dim == 3:
            known_xy = None
            known_z = None
            rect_y= None
            rect_x = None
            if rank == 0:
                known_xy = self.badlands_model.recGrid.tinMesh['vertices'] / fact
                known_z = self.badlands_model.elevation / fact
                rect_x = self.badlands_model.recGrid.rectX / fact
                rect_y  = self.badlands_model.recGrid.rectY / fact

            comm.Barrier()

            known_xy = comm.bcast(known_xy, root=0)
            known_z = comm.bcast(known_z, root=0)
            rect_x = comm.bcast(rect_x, root=0)
            rect_y = comm.bcast(rect_y, root=0)

            #comm.Barrier()
            interpolate_z = griddata(points=known_xy,
                                     values=known_z,
                                     xi=(rect_x, rect_y),
                                     method='nearest')
            nd_coords = np.column_stack((rect_x,rect_y, interpolate_z))

        np_surface = comm.bcast(nd_coords, root=0)
        comm.Barrier()
        tracer_velocity = self.Model.velocityField.evaluate_global(nd_coords)

        if rank == 0:
            tracer_disp = dimensionalise(tracer_velocity * dt, u.meter).magnitude

            self._inject_badlands_displacement(self.time_years,
                                               dt_years,        # in years
                                               tracer_disp,     # displacement in m/y 
                                               sigma)           # controls gaussian filter smoothing

            # get badlands
            bdm = self.badlands_model

            # force badlands checkpoint to align with UW
            #bdm.force.tDisplay = dt_years # this or the following
            run_until = self.time_years + dt_years
            bdm.force.next_display = run_until

            # Run the Badlands model to the same time point
            bdm.run_to_time(run_until)

            # # perform final checkpoint to syc 
            # # time_uw = time_bdm
            # # These save are criticle for checkpoint/restart
            # # so save them somewhere else
            # from badlands import checkPoints
            # checkPoints.write_checkpoints(
            #     bdm.input,
            #     bdm.recGrid,
            #     bdm.lGIDs,
            #     bdm.inIDs,
            #     bdm.tNow,
            #     bdm.FVmesh,
            #     bdm.force,
            #     bdm.flow,
            #     bdm.rain,
            #     bdm.elevation,
            #     bdm.fillH,
            #     bdm.cumdiff,
            #     bdm.cumhill,
            #     bdm.cumfail,
            #     bdm.wavediff,
            #     bdm.outputStep,
            #     bdm.prop,
            #     bdm.mapero,
            #     bdm.cumflex,
            # )

            # # force a final stratigraphy step
            # _ = bdm.strata.buildStrata(
            #     bdm.elevation,
            #     bdm.cumdiff,
            #     bdm.force.sealevel,
            #     bdm.recGrid.boundsPt,
            #     1,
            #     bdm.outputStep,
            # )

        self.time_years += dt_years

        # TODO: Improve the performance of this function
        self._update_material_types()
        comm.Barrier()

        if rank == 0 and self.verbose:
            purple = "\033[0;35m"
            endcol = "\033[00m"
            print(purple + "Processing surface with Badlands...Done" + endcol)
            sys.stdout.flush()

        return

    def _determine_particle_state_2D(self):

        known_xy = None
        known_z = None
        xs = None
        ys = None
        fact = dimensionalise(1.0, u.meter).magnitude
        if rank == 0:
            # points that we have known elevation for
            known_xy = self.badlands_model.recGrid.tinMesh['vertices'] / fact
            # elevation for those points
            known_z = self.badlands_model.elevation / fact
            xs = self.badlands_model.recGrid.regX / fact
            ys = self.badlands_model.recGrid.regY / fact

        known_xy = comm.bcast(known_xy, root=0)
        known_z = comm.bcast(known_z, root=0)
        xs = comm.bcast(xs, root=0)
        ys = comm.bcast(ys, root=0)

        comm.Barrier()

        grid_x, grid_y = np.meshgrid(xs, ys)
        interpolate_z = griddata(known_xy,
                                 known_z,
                                 (grid_x, grid_y),
                                 method='nearest').T
        interpolate_z = interpolate_z.mean(axis=1)

        f = interp1d(xs, interpolate_z)

        uw_surface = self.Model.swarm.particleCoordinates.data
        bdl_surface = f(uw_surface[:, 0])

        flags = uw_surface[:, 1] < bdl_surface

        return flags

    def _determine_particle_state(self):
        # Given Badlands' mesh, determine if each particle in 'volume' is above
        # (False) or below (True) it.

        # To do this, for each X/Y pair in 'volume', we interpolate its Z value
        # relative to the mesh in blModel. Then, if the interpolated Z is
        # greater than the supplied Z (i.e. Badlands mesh is above particle
        # elevation) it's sediment (True). Else, it's air (False).

        # TODO: we only support air/sediment layers right now; erodibility
        # layers are not implemented

        known_xy = None
        known_z = None
        fact = dimensionalise(1.0, u.meter).magnitude
        if rank == 0:
            # points that we have known elevation for
            known_xy = self.badlands_model.recGrid.tinMesh['vertices'] / fact
            known_z = self.badlands_model.elevation / fact

        known_xy = comm.bcast(known_xy, root=0)
        known_z = comm.bcast(known_z, root=0)

        comm.Barrier()

        volume = self.Model.swarm.particleCoordinates.data

        interpolate_xy = volume[:, [0, 1]]

        # NOTE: we're using nearest neighbour interpolation. This should be
        # sufficient as Badlands will normally run at a much higher resolution
        # than Underworld. 'linear' interpolation is much, much slower.
        interpolate_z = griddata(points=known_xy,
                                 values=known_z,
                                 xi=interpolate_xy,
                                 method='nearest')

        # True for sediment, False for air
        flags = volume[:, 2] < interpolate_z

        return flags

    def _update_material_types(self):

        # What do the materials (in air/sediment terms) look like now?
        if self.Model.mesh.dim == 3:
            under_bd_surface = self._determine_particle_state()
        if self.Model.mesh.dim == 2:
            under_bd_surface = self._determine_particle_state_2D()

        # If any materials changed state, update the Underworld material types
        mi = self.Model.materialField.data

        # convert air to sediment
        for air_material in self.airIndex:
            # if material air, and we're below surface, make it sediment
            sedimented_mask = np.logical_and(np.in1d(mi, air_material), under_bd_surface)
            mi[sedimented_mask] = self.sedimentIndex

        # convert sediment to air
        for air_material in self.airIndex:
            # if material is not air, and above surface, make it air
            eroded_mask = np.logical_and(~np.in1d(mi, air_material), ~under_bd_surface)
            mi[eroded_mask] = self.airIndex[0]

    def _inject_badlands_displacement(self, time, dt, disp, sigma):
        """
        Takes a plane of tracer points and their DISPLACEMENTS in 3D over time
        period dt applies a gaussian filter on it. Injects it into Badlands as 3D
        tectonic movement.
        """

        # The Badlands 3D interpolation map is the displacement of each DEM
        # node at the end of the time period relative to its starting position.
        # If you start a new displacement file, it is treated as starting at
        # the DEM starting points (and interpolated onto the TIN as it was at
        # that tNow).

        # kludge; don't keep adding new entries
        if self._disp_inserted:
            self.badlands_model.force.T_disp[0, 0] = time
            self.badlands_model.force.T_disp[0, 1] = (time + dt)
        else:
            self.badlands_model.force.T_disp = np.vstack(([time, time + dt], self.badlands_model.force.T_disp))
            self._disp_inserted = True

        # Extent the velocity field in the third dimension
        if self.Model.mesh.dim == 2:
            dispX = np.tile(disp[:, 0], self.badlands_model.recGrid.rny)
            dispY = np.zeros((self.badlands_model.recGrid.rnx * self.badlands_model.recGrid.rny,))
            dispZ = np.tile(disp[:, 1], self.badlands_model.recGrid.rny)

            disp = np.zeros((self.badlands_model.recGrid.rnx * self.badlands_model.recGrid.rny,3))
            disp[:, 0] = dispX
            disp[:, 1] = dispY
            disp[:, 2] = dispZ

        # Gaussian smoothing
        if sigma > 0:
            dispX = np.copy(disp[:, 0]).reshape(self.badlands_model.recGrid.rnx, self.badlands_model.recGrid.rny)
            dispY = np.copy(disp[:, 1]).reshape(self.badlands_model.recGrid.rnx, self.badlands_model.recGrid.rny)
            dispZ = np.copy(disp[:, 2]).reshape(self.badlands_model.recGrid.rnx, self.badlands_model.recGrid.rny)
            smoothX = gaussian_filter(dispX, sigma)
            smoothY = gaussian_filter(dispY, sigma)
            smoothZ = gaussian_filter(dispZ, sigma)
            disp[:, 0] = smoothX.flatten()
            disp[:, 1] = smoothY.flatten()
            disp[:, 2] = smoothZ.flatten()

        self.badlands_model.force.injected_disps = disp


class ErosionThreshold(SurfaceProcesses):

    def __init__(self, air=None, threshold=None, surfaceTracers=None,
                 Model=None, **kwargs):

        super(ErosionThreshold, self).__init__(Model=Model)

        self.Model = Model
        self.threshold = threshold
        self.air = air
        self.surfaceTracers = surfaceTracers
        self.Model = Model

    def _init_model(self):

        materialField = self.Model.materialField

        materialMap = {}
        for material in self.air:
            materialMap[material.index] = 1.0

        isAirMaterial = fn.branching.map(fn_key=materialField,
                                         mapping=materialMap,
                                         fn_default=0.0)

        belowthreshold = [(((isAirMaterial < 0.5) & (fn.input()[1] > nd(self.threshold))), self.air[0].index),
                          (True, materialField)]

        self._fn = fn.branching.conditional(belowthreshold)

    def solve(self, dt):

        if not self.Model:
            raise ValueError("Model is not defined")

        self.Model.materialField.data[:] = self._fn.evaluate(self.Model.swarm)
        if self.surfaceTracers:
            if self.surfaceTracers.swarm.particleCoordinates.data.size > 0:
                coords = self.surfaceTracers.swarm.particleCoordinates
                coords.data[coords.data[:, -1] > nd(self.threshold), -1] = nd(self.threshold)
        return


class SedimentationThreshold(SurfaceProcesses):

    def __init__(self, air=None, sediment=None,
                 threshold=None, timeField=None, Model=None,
                 surfaceTracers=None, **kwargs):

        super(SedimentationThreshold, self).__init__(Model=Model)

        self.timeField = timeField
        self.air = air
        self.sediment = sediment
        self.threshold = threshold
        self.surfaceTracers = surfaceTracers
        self.Model = Model

    def _init_model(self):

        materialField = self.Model.materialField

        materialMap = {}
        for material in self.air:
            materialMap[material.index] = 1.0

        isAirMaterial = fn.branching.map(fn_key=materialField,
                                         mapping=materialMap,
                                         fn_default=0.0)

        belowthreshold = [(((isAirMaterial > 0.5) & (fn.input()[1] < nd(self.threshold))), 0.),
                          (True, 1.)]

        self._change_material = fn.branching.conditional(belowthreshold)

        conditions = [(self._change_material < 0.5, self.sediment[0].index),
                      (True, materialField)]

        self._fn = fn.branching.conditional(conditions)

    def solve(self, dt):

        if not self.Model:
            raise ValueError("Model is not defined")

        if self.timeField:
            fn = self._change_material * self.timeField
            self.timeField.data[...] = fn.evaluate(self.Model.swarm)

        self.Model.materialField.data[:] = self._fn.evaluate(self.Model.swarm)

        if self.surfaceTracers:
            if self.surfaceTracers.swarm.particleCoordinates.data.size > 0:
                coords = self.surfaceTracers.swarm.particleCoordinates
                coords.data[coords.data[:, -1] < nd(self.threshold), -1] = nd(self.threshold)


class ErosionAndSedimentationThreshold(SedimentationThreshold, ErosionThreshold):

    def __init__(self, air=None, sediment=None,
                 threshold=None, timeField=None,
                 surfaceTracers=None, Model=None, **kwargs):

        super(ErosionAndSedimentationThreshold, self).__init__(Model=Model)

        self.timeField = timeField
        self.air = air
        self.sediment = sediment
        self.threshold = threshold
        self.surfaceTracers = surfaceTracers
        self.Model = Model

    def _init_model(self):

        ErosionThreshold._init_model(self)
        SedimentationThreshold._init_model(self)

    def solve(self, dt):

        ErosionThreshold.solve(self, dt)
        SedimentationThreshold.solve(self, dt)


class diffusiveSurface_2D(SurfaceProcesses):
    """Linear diffusion surface
    """


    def __init__(self, airIndex, sedimentIndex, D, surfaceArray, updateSurfaceLB=0.*u.kilometer, updateSurfaceRB=0.*u.kilometer, Model=None, timeField=None):
        """
        Parameters
        ----------

            airIndex :
                air index
            sedimentIndex :
                sediment Index
            D :
                Diffusive rate, in unit length^2 / unit time
            surfaceArray:
                Numpy array that marks the initial surface
            updateSurfaceLB :
                Distance to update surface from left boundary, default is 0 km which results in a free slip boundary
            updateSurfaceRB :
                Distance to update surface from right boundary, default is 0 km which results in a free slip boundary

            '''updated'''

            ***All units are converted under the hood***


            ***
            usage:
            Model.surfaceProcesses = diffusiveSurface(
                airIndex=air.index,
                sedimentIndex=Sediment.index,
                D= 100.0*u.meter**2/u.year,
                surfaceArray = coords
            )
            ***

        """

        self.airIndex = airIndex
        self.sedimentIndex = sedimentIndex
        self.timeField = timeField
        
        self.D  = D.to(u.kilometer**2 / u.year)


        ### a conversion, will throw an error if units are neglected
        self.surfaceArray    = surfaceArray
        self.updateSurfaceLB = updateSurfaceLB.to(u.kilometer)
        self.updateSurfaceRB = updateSurfaceRB.to(u.kilometer)
    
        
        self.Model = Model

        self.originalZ = None
        self.min_dist = None
        self.nd_coords = None
        self.dx = None


        ''' function to create grid for surface '''


    def _init_model(self):
        '''  creates a PT output '''
        ### automatically non-dimensionalises the imput coords if they have a dim
        self.Model.add_passive_tracers(name="surface", vertices=nd(self.surfaceArray), advect=False)

        self.Model.surface_tracers.allow_parallel_nn = True

        self.nd_coords = nd(self.surfaceArray)

        ### get distance between 1st and 2nd x coord and y coords to determine min distance between grid points
        x = np.sort(np.unique(self.nd_coords[:,0]))

        self.min_dist = np.diff(x).min()

        # self.dx = np.diff(x).min()

        ### create copy of original surface
        self.originalZ  = self.nd_coords[:,1]

        #### add variables for tracking that aren't included in UWGeo
        self.Model.surface_tracers.z_coord = self.Model.surface_tracers.add_variable( dataType="double", count=1 )
        
        self.Model.surface_tracers.D = self.Model.surface_tracers.add_variable( dataType="double", count=1 )

        if self.Model.surface_tracers.data.size != 0:
            ### erosion is downward (negative)
            self.Model.surface_tracers.D.data[:,0] = abs( np.repeat( nd(self.D), self.Model.surface_tracers.data.shape[0] ) )

            self.Model.surface_tracers.z_coord.data[:,0] = self.Model.surface_tracers.data[:,1]

        comm.barrier()


        ### add fields to track

        ### track velocity field on tracers
        self.Model.surface_tracers.add_tracked_field(self.Model.velocityField,
                                       name="surface_vel",
                                       units=u.centimeter/u.year,
                                       dataType="float", count=self.Model.mesh.dim)

        self.Model.surface_tracers.add_tracked_field(self.Model.surface_tracers.particleCoordinates,
                                       name="coords",
                                       units=u.centimeter/u.year,
                                       dataType="float", count=self.Model.mesh.dim)

        ## track the surface coordinates (could change to only show the height)
        self.Model.surface_tracers.add_tracked_field(self.Model.surface_tracers.z_coord,
                                name="topo_height",
                                units=u.kilometer,
                                dataType="float", count=1)

        ### track the diffusive surface rate
        self.Model.surface_tracers.add_tracked_field(self.Model.surface_tracers.D,
                                name="Diffusive rate",
                                units=u.meter**2/u.year,
                                dataType="float", count=1)


        comm.barrier()

    def solve(self, dt):


        if self.Model.surface_tracers.data.shape[0] > 0:
            ### evaluate on all nodes and get the tracer velocity on root proc
            tracer_velocity_local = self.Model.velocityField.evaluate(self.Model.surface_tracers.data)
            x_local = nd(self.Model.x.evaluate(self.Model.surface_tracers.data))
            y_local = nd(self.Model.y.evaluate(self.Model.surface_tracers.data))

            x  = np.ascontiguousarray(x_local)
            y  = np.ascontiguousarray(y_local)
            vx = np.ascontiguousarray(tracer_velocity_local[:,0])
            vy = np.ascontiguousarray(tracer_velocity_local[:,1])
        else:
            x = np.array([None], dtype='float64')
            y = np.array([None], dtype='float64')
            vx = np.array([None], dtype='float64')
            vy = np.array([None], dtype='float64')

        comm.barrier()


        ### Collect local array sizes using the high-level mpi4py gather
        sendcounts = np.array(comm.gather(len(x), root=0))

        comm.barrier()

        if rank == 0:
        ### creates dummy data on all nodes to store the surface
            # surface_data = np.zeros((npoints,2))
            x_data = np.zeros((sum(sendcounts)), dtype='float64')
            y_data = np.zeros((sum(sendcounts)), dtype='float64')
            vx_data = np.zeros((sum(sendcounts)), dtype='float64')
            vy_data = np.zeros((sum(sendcounts)), dtype='float64')
        else:
            x_data  = None
            y_data  = None
            vx_data = None
            vy_data = None

        ### store the surface spline on each node
        f1 = None

        comm.Gatherv(sendbuf=x, recvbuf=(x_data, sendcounts), root=0)

        comm.Gatherv(sendbuf=y, recvbuf=(y_data, sendcounts), root=0)

        ### gather velocity values
        comm.Gatherv(sendbuf=vx, recvbuf=(vx_data, sendcounts), root=0)

        comm.Gatherv(sendbuf=vy, recvbuf=(vy_data, sendcounts), root=0)


        if rank == 0:

            nd_D = nd( self.D )

            surface_data = np.zeros((len(x_data), 4), dtype='float64')
            surface_data[:,0] = x_data
            surface_data[:,1] = y_data
            surface_data[:,2] = vx_data
            surface_data[:,3] = vy_data

            surface_data = surface_data[~np.isnan(surface_data[:,0])]
            surface_data = surface_data[np.argsort(surface_data[:,0])]

            # # Advect top surface
            x_new = (surface_data[:,0] + (surface_data[:,2]*dt))
            y_new = (surface_data[:,1] + (surface_data[:,3]*dt))

            ## Spline top surface
            f = interp1d(x_new, y_new, kind='cubic', fill_value='extrapolate')
            
            ''' interpolate new surface back onto original grid '''
            x_nd = self.nd_coords[:,0]
            z_nd = f(x_nd)

            ### time to diffuse surface based on Model dt
            total_time = dt

            '''Velocity surface process'''

            '''erosion dt for vel model'''
            surface_dt_diffusion = ( 0.2 * ( (self.min_dist**2) / nd_D ) )

            vel_for_surface = max(abs(vx_data.max()), abs(vy_data.max()))

            surface_dt_vel = (0.2 *  ( self.min_dist / vel_for_surface) )

            surface_dt = min(surface_dt_diffusion, surface_dt_vel)

            surf_time = min(surface_dt, total_time)

            nts = math.ceil(total_time/surf_time)
            
            surf_dt = (total_time / nts)

            print('SP total time:', dimensionalise(total_time, u.year), 'timestep:', dimensionalise(surf_dt, u.year), 'No. of its:', nts, flush=True)


            ### Basic Hillslope diffusion
            for i in range(nts):
                qs = -nd_D * np.diff(z_nd)/np.diff(x_nd)
                dzdt = -np.diff(qs)/np.diff(x_nd[:-1])


                z_nd[1:-1] += dzdt*surface_dt




            ''' creates no movement condition near boundary '''
            ''' important when imposing a velocity as particles are easily deformed near the imposed condition'''
            ''' This changes the height to the points original height '''
            resetArea_x = (self.nd_coords[:,0] < nd(self.updateSurfaceLB)) | (self.nd_coords[:,0] > (nd(self.Model.maxCoord[0]) - (nd(self.updateSurfaceRB))))


            z_nd[resetArea_x] = self.originalZ[resetArea_x]
        

            ### creates function for the new surface that has eroded, to be broadcast back to nodes
            f1 = interp1d(self.nd_coords[:,0], z_nd, fill_value='extrapolate', kind='cubic')

        comm.barrier()

        '''broadcast the new surface'''
        ### broadcast function for the surface
        f1 = comm.bcast(f1, root=0)



        ### update the z coord of the surface array
        self.nd_coords[:,1] = f1(self.nd_coords[:,0])

        comm.barrier()


        ### has to be done on all procs due to an internal comm barrier in deform swarm (?)
        with self.Model.surface_tracers.deform_swarm():
            self.Model.surface_tracers.data[:,1] = f1(self.Model.surface_tracers.data[:,0])

        comm.barrier()

        if self.Model.surface_tracers.data.size != 0:
            ### update the surface only on procs that have the tracers
            self.Model.surface_tracers.z_coord.data[:,0] = self.Model.surface_tracers.data[:,1]

        comm.barrier()

        ### update the time of the sediment and air material as sed & erosion occurs
        if self.timeField:
            ### Set newly deposited sediment time to 0 (to record deposition time)
            self.Model.timeField.data[(self.Model.swarm.data[:,1] < f1(self.Model.swarm.data[:,0])) & (self.Model.materialField.data[:,0] == self.airIndex)] = 0.
            ### reset air material time back to the model time
            self.Model.timeField.data[(self.Model.swarm.data[:,1] > f1(self.Model.swarm.data[:,0])) & (self.Model.materialField.data[:,0] != self.airIndex)] = self.Model.timeField.data.max()

        '''Erode surface/deposit sed based on the surface'''
        ### update the material on each node according to the spline function for the surface
        self.Model.materialField.data[(self.Model.swarm.data[:,1] > f1(self.Model.swarm.data[:,0])) & (self.Model.materialField.data[:,0] != self.airIndex)] = self.airIndex
        self.Model.materialField.data[(self.Model.swarm.data[:,1] < f1(self.Model.swarm.data[:,0])) & (self.Model.materialField.data[:,0] == self.airIndex)] = self.sedimentIndex



        return

class velocitySurface_2D(SurfaceProcesses):
    """velocity surface erosion
    """


    def __init__(self, airIndex, sedimentIndex, sedimentationRate, erosionRate, surfaceArray, updateSurfaceLB=0.*u.kilometer, updateSurfaceRB=0.*u.kilometer, surfaceElevation=0.*u.kilometer, Model=None, timeField=None):
        """
        Parameters
        ----------

            airIndex :
                air index
            sedimentIndex :
                sediment Index
            sedimentaitonRate :
                Rate at which to deposit sediment, in unit length / unit time
            erosionRate :
                Rate at which to erode surface, in unit length / unit time
            surfaceArray:
                Numpy array that contains the initial surface
            surfaceElevation :
                boundary between air and crust/material, unit length of y coord to be given, default is y = 0 km.
            updateSurfaceLB :
                Distance to update surface from left boundary, default is 0 km which results in a free slip boundary
            updateSurfaceRB :
                Distance to update surface from right boundary, default is 0 km which results in a free slip boundary


            ***All units are converted under the hood***

            ***
            usage:
            Model.surfaceProcesses = velocitySurface(
                airIndex=air.index,
                sedimentIndex=Sediment.index,
                sedimentationRate= 1.0*u.millimeter / u.year,
                erosionRate= 0.5*u.millimeter / u.year,
                surfaceElevation=0.0*u.kilometer,
                surfaceArray = coords
            )
            ***

        """

        self.airIndex = airIndex
        self.sedimentIndex = sedimentIndex
        self.timeField = timeField

        self.ve  = sedimentationRate.to(u.kilometer / u.year)
        self.vs  = erosionRate.to(u.kilometer / u.year)

        ### a conversion, will throw an error if units are neglected
        self.surfaceArray    = surfaceArray
        self.updateSurfaceLB = updateSurfaceLB.to(u.kilometer)
        self.updateSurfaceRB = updateSurfaceRB.to(u.kilometer)
    
        
        self.surfaceElevation = surfaceElevation.to(u.kilometer)
        self.Model = Model

        self.originalZ = None
        self.min_dist = None
        self.nd_coords = None

        self.tkey = self.__class__.__name__+"_surface"


    def _init_model(self):
        '''  creates a PT output '''
        ### automatically non-dimensionalises the imput coords if they have a dim
        ## TODO: Fix naming for internal passive tracer swarm
        self.Model.add_passive_tracers(name=self.tkey, vertices=nd(self.surfaceArray), advect=False)

        st = self.Model.passive_tracers.get(self.tkey)
        assert st != None, f"Error getting passive tracer {self.tkey}"
        st.allow_parallel_nn = True

        self.nd_coords = nd(self.surfaceArray)

        ### get distance between 1st and 2nd x coord and y coords to determine min distance between grid points
        x = np.sort(np.unique(self.nd_coords[:,0]))

        self.min_dist = np.diff(x).min()

        ### create copy of original surface
        self.originalZ  = self.nd_coords[:,1]

        comm.barrier()



        ### add fields to track

        ### track velocity field on tracers
#        st.add_tracked_field(self.Model.velocityField,
#                                       name="surface_vel",
#                                       units=u.centimeter/u.year,
#                                       dataType="float", count=self.Model.mesh.dim)
#
#        st.add_tracked_field(st.particleCoordinates,
#                                       name="coords",
#                                       units=u.centimeter/u.year,
#                                       dataType="float", count=self.Model.mesh.dim)

        comm.barrier()

    def solve(self, dt):

        st = self.Model.passive_tracers.get(self.tkey)
        assert st != None, f"Error getting passive tracer {self.tkey}"
        if st.data.shape[0] > 0:
            ### evaluate on all nodes and get the tracer velocity on root proc
            tracer_velocity_local = self.Model.velocityField.evaluate(st.data)
            x_local = nd(self.Model.x.evaluate(st.data))
            y_local = nd(self.Model.y.evaluate(st.data))

            x  = np.ascontiguousarray(x_local)
            y  = np.ascontiguousarray(y_local)
            vx = np.ascontiguousarray(tracer_velocity_local[:,0])
            vy = np.ascontiguousarray(tracer_velocity_local[:,1])
        else:
            x = np.array([None], dtype='float64')
            y = np.array([None], dtype='float64')
            vx = np.array([None], dtype='float64')
            vy = np.array([None], dtype='float64')

        comm.barrier()


        ### Collect local array sizes using the high-level mpi4py gather
        sendcounts = np.array(comm.gather(len(x), root=0))

        comm.barrier()

        if rank == 0:
        ### creates dummy data on all nodes to store the surface
            # surface_data = np.zeros((npoints,2))
            x_data = np.zeros((sum(sendcounts)), dtype='float64')
            y_data = np.zeros((sum(sendcounts)), dtype='float64')
            vx_data = np.zeros((sum(sendcounts)), dtype='float64')
            vy_data = np.zeros((sum(sendcounts)), dtype='float64')
        else:
            x_data  = None
            y_data  = None
            vx_data = None
            vy_data = None

        ### store the surface spline on each node
        f1 = None

        comm.Gatherv(sendbuf=x, recvbuf=(x_data, sendcounts), root=0)

        comm.Gatherv(sendbuf=y, recvbuf=(y_data, sendcounts), root=0)

        ### gather velocity values
        comm.Gatherv(sendbuf=vx, recvbuf=(vx_data, sendcounts), root=0)

        comm.Gatherv(sendbuf=vy, recvbuf=(vy_data, sendcounts), root=0)


        if rank == 0:

            nd_ve = -1. * abs( nd(self.ve) ) ### erode down(negative)
            nd_vs =  1. * abs( nd(self.vs) ) ### sed up    (positive)

            surface_data = np.zeros((len(x_data), 4), dtype='float64')
            surface_data[:,0] = x_data
            surface_data[:,1] = y_data
            surface_data[:,2] = vx_data
            surface_data[:,3] = vy_data

            surface_data = surface_data[~np.isnan(surface_data[:,0])]
            surface_data = surface_data[np.argsort(surface_data[:,0])]

            # # Advect top surface
            x_new = (surface_data[:,0] + (surface_data[:,2]*dt))
            y_new = (surface_data[:,1] + (surface_data[:,3]*dt))

            ## Spline top surface
            f = interp1d(x_new, y_new, kind='cubic', fill_value='extrapolate')
            
            ''' interpolate new surface back onto original grid '''
            z_nd = f(self.nd_coords[:,0])

            ### Ve and Vs for loop to preserve original values
            Ve_loop  = np.zeros_like(z_nd, dtype='float64')
            Vs_loop  = np.zeros_like(z_nd, dtype='float64')


            ### time to diffuse surface based on Model dt
            total_time = dt

            '''Velocity surface process'''

            '''erosion dt for vel model'''
            vel_for_surface = max(vx_data.max(), vy_data.max())
            Vel_for_surface = max(abs(nd_ve), abs(nd_ve), abs(vx_data.max()), abs(vy_data.max()))

            surface_dt_vel = (0.2 *  (self.min_dist / Vel_for_surface) )

            surf_time = min(surface_dt_vel, total_time)

            nts = math.ceil(total_time/surf_time)
            
            surf_dt = (total_time / nts)

            print('SP total time:', dimensionalise(total_time, u.year), 'timestep:', dimensionalise(surf_dt, u.year), 'No. of its:', nts, flush=True)


            ### Velocity erosion/sedimentation rates for the surface
            for i in range(nts):
                ''' determine if particle is above or below the original surface elevation '''
                ''' erosion function '''
                Ve_loop[:] = nd(0. * u.kilometer/u.year)
                Ve_loop[(z_nd > nd(self.surfaceElevation))] = nd_ve

                ''' sedimentation function '''
                Vs_loop[:] = nd(0. * u.kilometer/u.year)
                Vs_loop[(z_nd <= nd(self.surfaceElevation))] = nd_vs


                dzdt =  Vs_loop + Ve_loop

                z_nd += (dzdt[:]*surf_dt)


            ''' creates no movement condition near boundary '''
            ''' important when imposing a velocity as particles are easily deformed near the imposed condition'''
            ''' This changes the height to the points original height '''
            resetArea_x = (self.nd_coords[:,0] < nd(self.updateSurfaceLB)) | (self.nd_coords[:,0] > (nd(self.Model.maxCoord[0]) - (nd(self.updateSurfaceRB))))


            z_nd[resetArea_x] = self.originalZ[resetArea_x]
        

            ### creates function for the new surface that has eroded, to be broadcast back to nodes
            f1 = interp1d(self.nd_coords[:,0], z_nd, fill_value='extrapolate', kind='cubic')

        comm.barrier()

        '''broadcast the new surface'''
        ### broadcast function for the surface
        f1 = comm.bcast(f1, root=0)



        ### update the z coord of the surface array
        self.nd_coords[:,1] = f1(self.nd_coords[:,0])

        comm.barrier()


        ### has to be done on all procs due to an internal comm barrier in deform swarm (?)
        with st.deform_swarm():
            st.data[:,1] = f1(st.data[:,0])

        comm.barrier()

        ### update the time of the sediment and air material as sed & erosion occurs
        if self.timeField:
            ### Set newly deposited sediment time to 0 (to record deposition time)
            self.Model.timeField.data[(self.Model.swarm.data[:,1] < f1(self.Model.swarm.data[:,0])) & (self.Model.materialField.data[:,0] == self.airIndex)] = 0.
            ### reset air material time back to the model time
            self.Model.timeField.data[(self.Model.swarm.data[:,1] > f1(self.Model.swarm.data[:,0])) & (self.Model.materialField.data[:,0] != self.airIndex)] = self.Model.timeField.data.max()

        '''Erode surface/deposit sed based on the surface'''
        ### update the material on each node according to the spline function for the surface
        self.Model.materialField.data[(self.Model.swarm.data[:,1] > f1(self.Model.swarm.data[:,0])) & (self.Model.materialField.data[:,0] != self.airIndex)] = self.airIndex
        self.Model.materialField.data[(self.Model.swarm.data[:,1] < f1(self.Model.swarm.data[:,0])) & (self.Model.materialField.data[:,0] == self.airIndex)] = self.sedimentIndex



        return


class velocitySurface_3D(SurfaceProcesses):
    """velocity surface erosion
    """

    def __init__(self, airIndex, sedimentIndex,
                    vs_condition, ve_condition,
                    surfaceArray,
                    method='nearest',
                    surfaceElevation=0.*u.kilometer,
                    updateSurfaceLB=0.*u.kilometer, updateSurfaceRB=0.*u.kilometer,
                     updateSurfaceTB=0.*u.kilometer, updateSurfaceBB=0.*u.kilometer,
                    Model=None, timeField=None):
        """
        Parameters
        ----------

            airIndex :
                air index
            sedimentIndex :
                sediment Index
            
            ve_condition :
                Condition that contains the erosion rate based on the x and y coord
            vs_condition :
                Condition that contains the sedimentation rate based on the x and y coord
            
            surfaceArray :
                coords at which the surface will be construced (x, y, z)
            surfaceElevation :
                boundary between air and crust/material, unit length of y coord to be given

            method :
                Interpolation method for scipy griddata function to use for the surface, default is 'nearest', which is quicker at high res

            updateSurfaceLB :
                Distance to update surface from left boundary, default is 0 for free slip, unit length required
            updateSurfaceRB :
                Distance to update surface from right boundary, default is 0 for free slip, unit length required    
                
            updateSurfaceTB :
                Distance to update surface from top boundary, default is 0 for free slip, unit length required
            updateSurfaceBB :
                Distance to update surface from bottom boundary, default is 0 for free slip, unit length required
        

            ***All units are converted under the hood***

            ***
            usage:
            x = np.linspace(Model.minCoord[0], Model.maxCoord[0], 100)
            y = np.linspace(Model.minCoord[1], Model.maxCoord[1], 100)

            xi, yi = np.meshgrid(x, y)

            coords = np.zeros(shape=(xi.flatten().shape[0], 3))
            coords[:,0] = xi.flatten()
            coords[:,1] = yi.flatten()
            coords[:,2] = np.zeros_like(coords[:,0]) ### or any array with same shape as x and y coords with the initial height

            coords = coords * u.kilometer

            ve_conditions = fn.branching.conditional([((Model.x >= 0.5) & (Model.y >=0.5), GEO.nd(1 * u.millimeter/u.year)),
                                          (True, GEO.nd(0.5 * u.millimeter/u.year))])

            vs_conditions = fn.branching.conditional([(True, GEO.nd(0.05 * u.millimeter/u.year))])

            Model.surfaceProcesses = velocitySurface3D(airIndex=air.index,
                                                       sedimentIndex=Sediment.index,
                                                       surfaceArray = coords,                ### grid with surface points (x, y, z)
                                                       vs_condition = vs_conditions,         ### sedimentation rate at each grid point
                                                       ve_condition = ve_conditions,         ### erosion rate at each grid point
                                                       surfaceElevation=air.bottom)
            ***

        """

        self.airIndex = airIndex
        self.sedimentIndex = sedimentIndex
        self.timeField = timeField

        self.ve_condition = ve_condition
        self.vs_condition = vs_condition

        self.method = method

        ### a conversion, will throw an error if units are neglected
        self.surfaceArray = surfaceArray.to(u.kilometer)
        self.updateSurfaceLB = updateSurfaceLB.to(u.kilometer)
        self.updateSurfaceRB = updateSurfaceRB.to(u.kilometer)
        
        self.updateSurfaceTB = updateSurfaceTB.to(u.kilometer)
        self.updateSurfaceBB = updateSurfaceBB.to(u.kilometer)
        
        self.surfaceElevation = surfaceElevation.to(u.kilometer)
        self.Model = Model

        self.originalZ = None
        self.z_surf = None 
        self.min_dist = None
        self.nd_coords = None

        # we save the key of the passive tracer swarm, rather than the instance, because
        # upon restarts the instance can be replaced
        self.tkey = self.__class__.__name__+"_surface"


    def _init_model(self):
        ### automatically non-dimensionalises the imput coords if they have a dim
        self.Model.add_passive_tracers(name=self.tkey, vertices=self.surfaceArray, advect=False)

        st = self.Model.passive_tracers.get(self.tkey)
        assert st != None, f"Error getting passive tracer {self.tkey}"
        st.allow_parallel_nn = True

        self.nd_coords = nd(self.surfaceArray)

        ### get distance between 1st and 2nd x coord and y coords to determine min distance between grid points
        x = np.sort(np.unique(self.nd_coords[:,0]))
        y = np.sort(np.unique(self.nd_coords[:,1]))
        dx = np.diff(x).min()
        dy = np.diff(y).min()

        self.min_dist = min(dx, dy)

        ### create copy of original surface
        self.originalZ  = self.nd_coords[:,2]

        comm.barrier()




        ### add fields to track

        ### track velocity field on tracers
#        self.Model.surface_tracers.add_tracked_field(self.Model.velocityField,
#                                       name="surface_vel",
#                                       units=u.centimeter/u.year,
#                                       dataType="float", count=self.Model.mesh.dim)
#
#        self.Model.surface_tracers.add_tracked_field(self.Model.surface_tracers.particleCoordinates,
#                                       name="coords",
#                                       units=u.centimeter/u.year,
#                                       dataType="float", count=self.Model.mesh.dim)
#

    # def solve(self, dt):

    #     ### evaluate on all nodes and get the tracer velocity on root proc
    #     tracer_velocity = self.Model.velocityField.evaluate_global(self.nd_coords)

    #     ### utilises the evaluate_global to get values that are across multiple CPUs on root CPU
    #     ve = (self.ve_condition.evaluate_global(self.nd_coords))
    #     vs = (self.vs_condition.evaluate_global(self.nd_coords))


    #     comm.barrier()


    #     if rank == 0:

    #         ve = -1. * abs(ve) ### erode down(negative)
    #         vs =  1. * abs(vs) ### sed up    (positive)

    #         # # Advect top surface
    #         x_new = (self.nd_coords[:,0] + (tracer_velocity[:,0]*dt))
    #         y_new = (self.nd_coords[:,1] + (tracer_velocity[:,1]*dt))
    #         z_new = (self.nd_coords[:,2] + (tracer_velocity[:,2]*dt))


    #         ''' interpolate new surface back onto original grid '''
    #         #### griddata seems to be okay, rbf was causing issues with memory usage in parallel
    #         z_nd = griddata((x_new, y_new), z_new, (self.nd_coords[:,0], self.nd_coords[:,1]), method=self.method).ravel()


    #         ### Ve and Vs for loop to preserve original values
    #         Ve_loop  = np.zeros_like(z_nd, dtype='float64')
    #         Vs_loop  = np.zeros_like(z_nd, dtype='float64')


    #         ### time to diffuse surface based on Model dt
    #         total_time = dt

    #         '''Velocity surface process'''

    #         '''erosion dt for vel model'''
    #         Vel_for_surface = max(abs(vs).max(), abs(ve).max(), abs(tracer_velocity).max())

    #         surface_dt_vel = (0.2 *  (self.min_dist / Vel_for_surface) )

    #         surf_time = min(surface_dt_vel, total_time)

    #         nts = math.ceil(total_time/surf_time)
            
    #         surf_dt = (total_time / nts)

    #         print('SP total time:', dimensionalise(total_time, u.year), 'timestep:', dimensionalise(surf_dt, u.year), 'No. of its:', nts, flush=True)


    #         ### Velocity erosion/sedimentation rates for the surface
    #         for i in range(nts):
    #             ''' determine if particle is above or below the original surface elevation '''
    #             ''' erosion function '''
    #             Ve_loop[:] = nd(0. * u.kilometer/u.year)
    #             Ve_loop[(z_nd > nd(self.surfaceElevation))] = ve[:,0][(z_nd > nd(self.surfaceElevation))]

    #             ''' sedimentation function '''
    #             Vs_loop[:] = nd(0. * u.kilometer/u.year)
    #             Vs_loop[(z_nd <= nd(self.surfaceElevation))] = vs[:,0][(z_nd <= nd(self.surfaceElevation))]


    #             dzdt =  Vs_loop + Ve_loop

    #             z_nd += (dzdt[:]*surf_dt)


    #         ''' creates no movement condition near boundary '''
    #         ''' important when imposing a velocity as particles are easily deformed near the imposed condition'''
    #         ''' This changes the height to the points original height '''
    #         resetArea_x = (self.nd_coords[:,0] < nd(self.updateSurfaceLB)) | (self.nd_coords[:,0] > (nd(self.Model.maxCoord[0]) - (nd(self.updateSurfaceRB))))
            
    #         resetArea_y = (self.nd_coords[:,1] < nd(self.updateSurfaceBB)) | (self.nd_coords[:,1] > (nd(self.Model.maxCoord[1]) - (nd(self.updateSurfaceTB))))


    #         z_nd[resetArea_x | resetArea_y] = self.originalZ[resetArea_x | resetArea_y]
        

    #         self.z_new = z_nd


    #     comm.barrier()

    #     '''broadcast the new surface'''
    #     ### broadcast function for the surface
    #     self.z_new = comm.bcast(self.z_new, root=0)
        

    #     comm.barrier()

    #     ### update the z coord of the surface array
    #     self.nd_coords[:,2] = self.z_new

    #     comm.barrier()


    #     ### has to be done on all procs due to an internal comm barrier in deform swarm (?)
    #     with self.Model.surface_tracers.deform_swarm():
    #         self.Model.surface_tracers.data[:,2] = griddata((self.nd_coords[:,0], self.nd_coords[:,1]), self.z_new, (self.Model.surface_tracers.data[:,0], self.Model.surface_tracers.data[:,1]), method=self.method).ravel()

    #     comm.barrier()

    #     if self.Model.surface_tracers.data.size != 0:
    #         ### update the surface only on procs that have the tracers
    #         self.Model.surface_tracers.z_coord.data[:,0] = self.Model.surface_tracers.data[:,2]

    #     comm.barrier()


    #     ### cacluate surface for swarm particles
    #     z_new_surface = griddata((self.nd_coords[:,0], self.nd_coords[:,1]), self.z_new, (self.Model.swarm.data[:,0], self.Model.swarm.data[:,1]), method=self.method).ravel()

    #     comm.barrier()

    #     ### update the time of the sediment and air material as sed & erosion occurs
    #     if self.timeField:
    #         ### Set newly deposited sediment time to 0 (to record deposition time)
    #         self.Model.timeField.data[(self.Model.swarm.data[:,2] < z_new_surface) & (self.Model.materialField.data[:,0] == self.airIndex) ] = 0.
    #         ### reset air material time back to the model time
    #         self.Model.timeField.data[(self.Model.swarm.data[:,2] >= z_new_surface) & (self.Model.materialField.data[:,0] != self.airIndex) ] = self.Model.timeField.data.max()

    #     '''Erode surface/deposit sed based on the surface'''
    #     ### update the material on each node according to the spline function for the surface
    #     self.Model.materialField.data[(self.Model.swarm.data[:,2] >= z_new_surface) & (self.Model.materialField.data[:,0] != self.airIndex) ] = self.airIndex
    #     self.Model.materialField.data[(self.Model.swarm.data[:,2] < z_new_surface) & (self.Model.materialField.data[:,0] == self.airIndex) ] = self.sedimentIndex

    #     comm.barrier()



    #     return

    def solve(self, dt):
        st = self.Model.passive_tracers.get(self.tkey)
        assert st != None, f"Error getting passive tracer {self.tkey}"
        if st.data.shape[0] > 0:
            x = np.ascontiguousarray(st.data[:,0])
            y = np.ascontiguousarray(st.data[:,1])
            z = np.ascontiguousarray(st.data[:,2])

            ### evaluate to get the tracer velocity
            tracer_velocity = self.Model.velocityField.evaluate(st.data)
            vx = np.ascontiguousarray(tracer_velocity[:,0])
            vy = np.ascontiguousarray(tracer_velocity[:,1])
            vz = np.ascontiguousarray(tracer_velocity[:,2])
            
            ### evaluate to get the ve and vs values
            ve = np.ascontiguousarray(self.ve_condition.evaluate(st.data))
            vs = np.ascontiguousarray(self.vs_condition.evaluate(st.data))
        else:
            x =  np.array([None], dtype='float64')
            y =  np.array([None], dtype='float64')
            z =  np.array([None], dtype='float64')
            tracer_velocity =  np.array([None], dtype='float64')
            vx =  np.array([None], dtype='float64')
            vy =  np.array([None], dtype='float64')
            vz =  np.array([None], dtype='float64')
            ve =  np.array([None], dtype='float64')
            vs =  np.array([None], dtype='float64')



        comm.barrier()

        sendcounts = np.array(comm.gather(len(x), root=0))

        comm.barrier()

        if rank == 0:
        ### creates dummy data on all nodes to store the surface
            # surface_data = np.zeros((npoints,2))
            x_data = np.zeros((sum(sendcounts)), dtype='float64')
            y_data = np.zeros((sum(sendcounts)), dtype='float64')
            z_data = np.zeros((sum(sendcounts)), dtype='float64')

            vx_data = np.zeros((sum(sendcounts)), dtype='float64')
            vy_data = np.zeros((sum(sendcounts)), dtype='float64')
            vz_data = np.zeros((sum(sendcounts)), dtype='float64')

            ve_data = np.zeros((sum(sendcounts)), dtype='float64')
            vs_data = np.zeros((sum(sendcounts)), dtype='float64')

        else:
            x_data  = None
            y_data  = None
            z_data  = None
            vx_data = None
            vy_data = None
            vz_data = None
            ve_data = None
            vs_data = None

        comm.Gatherv(sendbuf=x, recvbuf=(x_data, sendcounts), root=0)
        comm.Gatherv(sendbuf=y, recvbuf=(y_data, sendcounts), root=0)
        comm.Gatherv(sendbuf=z, recvbuf=(z_data, sendcounts), root=0)

        ### gather velocity values
        comm.Gatherv(sendbuf=vx, recvbuf=(vx_data, sendcounts), root=0)
        comm.Gatherv(sendbuf=vy, recvbuf=(vy_data, sendcounts), root=0)
        comm.Gatherv(sendbuf=vz, recvbuf=(vz_data, sendcounts), root=0)

        ### Gather SP values
        comm.Gatherv(sendbuf=ve, recvbuf=(ve_data, sendcounts), root=0)
        comm.Gatherv(sendbuf=vs, recvbuf=(vs_data, sendcounts), root=0)



        if rank == 0:

            surface_data = np.zeros((len(x_data), 8), dtype='float64')
            surface_data[:,0] = x_data
            surface_data[:,1] = y_data
            surface_data[:,2] = z_data

            surface_data[:,3] = vx_data
            surface_data[:,4] = vy_data
            surface_data[:,5] = vz_data

            surface_data[:,6] = ve_data
            surface_data[:,7] = vs_data

            surface_data = surface_data[~np.isnan(surface_data[:,0])]
            # surface_data = surface_data[np.argsort(surface_data[:,0])]

            ve = -1. * abs(surface_data[:,6]) ### erode down(negative)
            vs =  1. * abs(surface_data[:,7]) ### sed up    (positive)

            # # Advect top surface
            x_new = (surface_data[:,0] + (surface_data[:,3]*dt))
            y_new = (surface_data[:,1] + (surface_data[:,4]*dt))
            z_new = (surface_data[:,2] + (surface_data[:,5]*dt))


            ''' interpolate new surface back onto original grid '''
            #### griddata seems to be okay, rbf was causing issues with memory usage in parallel
            # z_nd = griddata((x_new, y_new), z_new, (self.nd_coords[:,0], self.nd_coords[:,1]), method=self.method).ravel()
            z_nd = griddata((x_new, y_new), z_new, (surface_data[:,0], surface_data[:,1]), method=self.method).ravel()


            ### Ve and Vs for loop to preserve original values
            Ve_loop  = np.zeros_like(z_nd, dtype='float64')
            Vs_loop  = np.zeros_like(z_nd, dtype='float64')


            ### time to diffuse surface based on Model dt
            total_time = dt

            '''Velocity surface process'''

            '''erosion dt for vel model'''
            Vel_for_surface = surface_data[:,3:].max()

            surface_dt_vel = (0.2 *  (self.min_dist / Vel_for_surface) )

            surf_time = min(surface_dt_vel, total_time)

            nts = math.ceil(total_time/surf_time)
            
            surf_dt = (total_time / nts)

            print('SP total time:', dimensionalise(total_time, u.year), 'timestep:', dimensionalise(surf_dt, u.year), 'No. of its:', nts, flush=True)


            ### Velocity erosion/sedimentation rates for the surface
            for i in range(nts):
                ''' determine if particle is above or below the original surface elevation '''
                ''' erosion function '''
                Ve_loop[:] = nd(0. * u.kilometer/u.year)
                Ve_loop[(z_nd > nd(self.surfaceElevation))] = ve[(z_nd > nd(self.surfaceElevation))]

                ''' sedimentation function '''
                Vs_loop[:] = nd(0. * u.kilometer/u.year)
                Vs_loop[(z_nd <= nd(self.surfaceElevation))] = vs[(z_nd <= nd(self.surfaceElevation))]


                dzdt =  Vs_loop + Ve_loop

                z_nd += (dzdt[:]*surf_dt)


            ''' creates no movement condition near boundary '''
            ''' important when imposing a velocity as particles are easily deformed near the imposed condition'''
            ''' This changes the height to the points original height '''
            resetArea_x = (surface_data[:,0] < nd(self.updateSurfaceLB)) | (surface_data[:,0] > (nd(self.Model.maxCoord[0]) - (nd(self.updateSurfaceRB))))
            
            resetArea_y = (surface_data[:,1] < nd(self.updateSurfaceBB)) | (surface_data[:,1] > (nd(self.Model.maxCoord[1]) - (nd(self.updateSurfaceTB))))


            z_nd[resetArea_x | resetArea_y] = self.originalZ[resetArea_x | resetArea_y]
        

            self.z_surf = griddata((surface_data[:,0], surface_data[:,1]), z_nd, (self.nd_coords[:,0], self.nd_coords[:,1]), method=self.method).ravel()


        comm.barrier()

        '''broadcast the new surface'''
        ### broadcast function for the surface
        self.z_surf = comm.bcast(self.z_surf, root=0)
        

        comm.barrier()

        ### update the z coord of the surface array
        self.nd_coords[:,2] = self.z_surf

        comm.barrier()


        ### has to be done on all procs due to an internal comm barrier in deform swarm (?)
        with st.deform_swarm():
            st.data[:,2] = griddata((self.nd_coords[:,0], self.nd_coords[:,1]), self.z_surf, (st.data[:,0], st.data[:,1]), method=self.method).ravel()

        comm.barrier()

        if st.data.size != 0:
            ### update the surface only on procs that have the tracers
            st.z_coord.data[:,0] = st.data[:,2]

        comm.barrier()


        ### cacluate surface for swarm particles
        z_new_surface = griddata((self.nd_coords[:,0], self.nd_coords[:,1]), self.z_surf, (self.Model.swarm.data[:,0], self.Model.swarm.data[:,1]), method=self.method).ravel()

        comm.barrier()

        ### update the time of the sediment and air material as sed & erosion occurs
        if self.timeField:
            ### Set newly deposited sediment time to 0 (to record deposition time)
            self.Model.timeField.data[(self.Model.swarm.data[:,2] < z_new_surface) & (self.Model.materialField.data[:,0] == self.airIndex) ] = 0.
            ### reset air material time back to the model time
            self.Model.timeField.data[(self.Model.swarm.data[:,2] >= z_new_surface) & (self.Model.materialField.data[:,0] != self.airIndex) ] = self.Model.timeField.data.max()

        '''Erode surface/deposit sed based on the surface'''
        ### update the material on each node according to the spline function for the surface
        self.Model.materialField.data[(self.Model.swarm.data[:,2] >= z_new_surface) & (self.Model.materialField.data[:,0] != self.airIndex) ] = self.airIndex
        self.Model.materialField.data[(self.Model.swarm.data[:,2] < z_new_surface) & (self.Model.materialField.data[:,0] == self.airIndex) ] = self.sedimentIndex

        comm.barrier()

        return
    
