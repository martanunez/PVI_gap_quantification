import vtk
import math
import numpy as np
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
import os
import sys
from scipy import sparse
import scipy.sparse.linalg as linalg_sp
from scipy.sparse import vstack, hstack, coo_matrix, csc_matrix
import glob


import heapq
# from queue import Queue, heapq, deque  # for Python 2 and python 3. In needs pip install future


###     Input/Output    ###
def readvtk(filename):
    """Read VTK file"""
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()

def readvtp(filename):
    """Read VTP file"""
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()

def writevtk(surface, filename, type='ascii'):
    """Write binary or ascii VTK file"""
    writer = vtk.vtkPolyDataWriter()
    if vtk.vtkVersion.GetVTKMajorVersion() > 5:
        writer.SetInputData(surface)
    else:
        writer.SetInput(surface)
    writer.SetFileName(filename)
    if type == 'ascii':
        writer.SetFileTypeToASCII()
    elif type == 'binary':
        writer.SetFileTypeToBinary()
    writer.Write()

def writevtp(surface, filename):
    """Write VTP file"""
    writer = vtk.vtkXMLPolyDataWriter()
    if vtk.vtkVersion.GetVTKMajorVersion() > 5:
        writer.SetInputData(surface)
    else:
        writer.SetInput(surface)
    writer.SetFileName(filename)
#    writer.SetDataModeToBinary()
    writer.Write()

###     Math    ###
def euclideandistance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2 + (point1[2] - point2[2])**2)

def normvector(v):
    return math.sqrt(dot(v, v))

def angle(v1, v2):
    return math.acos(dot(v1, v2) / (normvector(v1) * normvector(v2)))

def acumvectors(point1, point2):
    return [point1[0] + point2[0], point1[1] + point2[1], point1[2] + point2[2]]

def subtractvectors(point1, point2):
    return [point1[0] - point2[0], point1[1] - point2[1], point1[2] - point2[2]]

def dividevector(point, n):
    nr = float(n)
    return [point[0]/nr, point[1]/nr, point[2]/nr]

def multiplyvector(point, n):
    nr = float(n)
    return [nr*point[0], nr*point[1], nr*point[2]]

def sumvectors(vect1, scalar, vect2):
    return [vect1[0] + scalar*vect2[0], vect1[1] + scalar*vect2[1], vect1[2] + scalar*vect2[2]]

def cross(v1, v2):
    return [v1[1]*v2[2] - v1[2]*v2[1], v1[2]*v2[0] - v1[0]*v2[2], v1[0]*v2[1] - v1[1]*v2[0]]

def dot(v1, v2):
    return sum((a*b) for a, b in zip(v1, v2))

def normalizevector(v):
    norm = normvector(v)
    return [v[0] / norm, v[1] / norm, v[2] / norm]

###     Mesh processing     ###

def cleanpolydata(polydata):
    cleaner = vtk.vtkCleanPolyData()
    if vtk.vtkVersion.GetVTKMajorVersion() > 5:
        cleaner.SetInputData(polydata)
    else:
        cleaner.SetInput(polydata)
    cleaner.Update()
    return cleaner.GetOutput()

def fillholes(polydata, size):
    """Fill mesh holes smaller than 'size' """
    filler = vtk.vtkFillHolesFilter()
    if vtk.vtkVersion.GetVTKMajorVersion() > 5:
        filler.SetInputData(polydata)
    else:
        filler.SetInput(polydata)
    filler.SetHoleSize(size)
    filler.Update()
    return filler.GetOutput()

def pointthreshold(polydata, arrayname, start=0, end=1, alloff=0):
    """ Clip polydata according to given thresholds in scalar array"""
    threshold = vtk.vtkThreshold()
    if vtk.vtkVersion.GetVTKMajorVersion() > 5:
        threshold.SetInputData(polydata)
    else:
        threshold.SetInput(polydata)
    threshold.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, arrayname)
    threshold.ThresholdBetween(start, end)
    if (alloff):
        threshold.AllScalarsOff()
    threshold.Update()
    surfer = vtk.vtkDataSetSurfaceFilter()
    if vtk.vtkVersion.GetVTKMajorVersion() > 5:
        surfer.SetInputData(threshold.GetOutput())
    else:
        surfer.SetInput(threshold.GetOutput())
    surfer.Update()
    return surfer.GetOutput()

def cellthreshold(polydata, arrayname, start=0, end=1):
    threshold = vtk.vtkThreshold()
    threshold.SetInputData(polydata)
    threshold.SetInputArrayToProcess(0,0,0,vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS,arrayname)
    threshold.ThresholdBetween(start,end)
    threshold.Update()

    surfer = vtk.vtkDataSetSurfaceFilter()
    surfer.SetInputConnection(threshold.GetOutputPort())
    surfer.Update()
    return surfer.GetOutput()

def roundpointarray(polydata, name):
    """Round values in point array"""
    # get original array
    array = polydata.GetPointData().GetArray(name)
    # round labels
    for i in range(polydata.GetNumberOfPoints()):
        value = array.GetValue(i)
        array.SetValue(i, round(value))
    return polydata

def planeclip(surface, point, normal, insideout=1):
    """Clip surface using plane given by point and normal"""
    clipplane = vtk.vtkPlane()
    clipplane.SetOrigin(point)
    clipplane.SetNormal(normal)
    clipper = vtk.vtkClipPolyData()
    if vtk.vtkVersion.GetVTKMajorVersion() > 5:
        clipper.SetInputData(surface)
    else:
        clipper.SetInput(surface)
    clipper.SetClipFunction(clipplane)

    if insideout == 1:
        # print 'insideout ON'
        clipper.InsideOutOn()
    else:
        # print 'insideout OFF'
        clipper.InsideOutOff()
    clipper.Update()
    return clipper.GetOutput()

def cutdataset(dataset, point, normal):
    """Similar to planeclip but using vtkCutter instead of vtkClipPolyData"""
    cutplane = vtk.vtkPlane()
    cutplane.SetOrigin(point)
    cutplane.SetNormal(normal)
    cutter = vtk.vtkCutter()
    if vtk.vtkVersion.GetVTKMajorVersion() > 5:
        cutter.SetInputData(dataset)
    else:
        cutter.SetInput(dataset)
    cutter.SetCutFunction(cutplane)
    cutter.Update()
    return cutter.GetOutput()

def pointset_centreofmass(polydata):
    centre = [0, 0, 0]
    for i in range(polydata.GetNumberOfPoints()):
        point = [polydata.GetPoints().GetPoint(i)[0],
          polydata.GetPoints().GetPoint(i)[1],
          polydata.GetPoints().GetPoint(i)[2]]
        centre = acumvectors(centre, point)
    return dividevector(centre, polydata.GetNumberOfPoints())

def seeds_to_csv(seedsfile, arrayname, labels, outfile):
    """Read seeds from VTP file, write coordinates in csv"""
    # f = open(outfile, 'wb')
    f = open(outfile, 'w')
    allseeds = readvtp(seedsfile)
    for l in labels:
        currentseeds = pointthreshold(allseeds, arrayname, l, l, 0)
        currentpoint = pointset_centreofmass(currentseeds)
        line = str(currentpoint[0]) + ',' + str(currentpoint[1]) + ',' + str(currentpoint[2]) + '\n'
        f.write(line)
    f.close()

def point2vertexglyph(point):
    """Create glyph from points to visualise them"""
    points = vtk.vtkPoints()
    points.InsertNextPoint(point[0], point[1], point[2])
    poly = vtk.vtkPolyData()
    poly.SetPoints(points)
    glyph = vtk.vtkVertexGlyphFilter()
    glyph.SetInputConnection(poly.GetProducerPort())
    glyph.Update()
    return glyph.GetOutput()

def generateglyph(polyIn, scalefactor=2):
    vertexGlyphFilter = vtk.vtkGlyph3D()
    sphereSource = vtk.vtkSphereSource()
    if vtk.vtkVersion.GetVTKMajorVersion() > 5:
        vertexGlyphFilter.SetSourceData(sphereSource.GetOutput())
        vertexGlyphFilter.SetInputData(polyIn)
    else:
        vertexGlyphFilter.SetSource(sphereSource.GetOutput())
        vertexGlyphFilter.SetInput(polyIn)
    vertexGlyphFilter.SetColorModeToColorByScalar()
    vertexGlyphFilter.SetSourceConnection(sphereSource.GetOutputPort())
    vertexGlyphFilter.ScalingOn()
    vertexGlyphFilter.SetScaleFactor(scalefactor)
    vertexGlyphFilter.Update()
    return vertexGlyphFilter.GetOutput()

def linesource(p1, p2):
    """Create vtkLine from coordinates of 2 points"""
    source = vtk.vtkLineSource()
    source.SetPoint1(p1[0], p1[1], p1[2])
    source.SetPoint2(p2[0], p2[1], p2[2])
    return source.GetOutput()

def append(polydata1, polydata2):
    """Define new polydata appending polydata1 and polydata2"""
    appender = vtk.vtkAppendPolyData()
    appender.AddInput(polydata1)
    appender.AddInput(polydata2)
    appender.Update()
    return appender.GetOutput()

def append_polys(list):
    '''Append all polys in list and return single merged polydata'''
    append = vtk.vtkAppendPolyData()
    for n in range(len(list)):
        poly = readvtk(list[n])
        if vtk.vtkVersion().GetVTKMajorVersion() < 5:
            append.AddInput(poly)
        else:
            append.AddInputData(poly)
    append.Update()
    return append.GetOutput()

def extractcells(polydata, idlist):
    """Extract cells from polydata whose cellid is in idlist."""
    cellids = vtk.vtkIdList()  # specify cellids
    cellids.Initialize()
    for i in idlist:
        cellids.InsertNextId(i)

    extract = vtk.vtkExtractCells()  # extract cells with specified cellids
    if vtk.vtkVersion.GetVTKMajorVersion() > 5:
        extract.SetInputData(polydata)
    else:
        extract.SetInput(polydata)
    extract.AddCellList(cellids)
    extraction = extract.GetOutput()

    geometry = vtk.vtkGeometryFilter()  # unstructured grid to polydata
    if vtk.vtkVersion.GetVTKMajorVersion() > 5:
        geometry.SetInputData(extraction)
    else:
        geometry.SetInput(extraction)
    geometry.Update()
    return geometry.GetOutput()

def extractboundaryedge(polydata):
    edge = vtk.vtkFeatureEdges()
    if vtk.vtkVersion.GetVTKMajorVersion() > 5:
        edge.SetInputData(polydata)
    else:
        edge.SetInput(polydata)
    edge.FeatureEdgesOff()
    edge.NonManifoldEdgesOff()
    edge.Update()
    return edge.GetOutput()

def extractlargestregion(polydata):
    """Keep only biggest region"""
    surfer = vtk.vtkDataSetSurfaceFilter()
    if vtk.vtkVersion.GetVTKMajorVersion() > 5:
        surfer.SetInputData(polydata)
    else:
        surfer.SetInput(polydata)
    surfer.Update()

    cleaner = vtk.vtkCleanPolyData()
    if vtk.vtkVersion.GetVTKMajorVersion() > 5:
        cleaner.SetInputData(surfer.GetOutput())
    else:
        cleaner.SetInput(surfer.GetOutput())
    cleaner.Update()

    connect = vtk.vtkPolyDataConnectivityFilter()
    if vtk.vtkVersion.GetVTKMajorVersion() > 5:
        connect.SetInputData(cleaner.GetOutput())
    else:
        connect.SetInput(cleaner.GetOutput())
    connect.SetExtractionModeToLargestRegion()
    connect.Update()

    cleaner = vtk.vtkCleanPolyData()
    if vtk.vtkVersion.GetVTKMajorVersion() > 5:
        cleaner.SetInputData(connect.GetOutput())
    else:
        cleaner.SetInput(connect.GetOutput())
    cleaner.Update()
    return cleaner.GetOutput()

def countregions(polydata):
    """Count number of connected components/regions"""
    # preventive measures: clean before connectivity filter to avoid artificial regionIds
    surfer = vtk.vtkDataSetSurfaceFilter()
    if vtk.vtkVersion.GetVTKMajorVersion() > 5:
        surfer.SetInputData(polydata)
    else:
        surfer.SetInput(polydata)
    surfer.Update()

    cleaner = vtk.vtkCleanPolyData()
    if vtk.vtkVersion.GetVTKMajorVersion() > 5:
        cleaner.SetInputData(surfer.GetOutput())
    else:
        cleaner.SetInput(surfer.GetOutput())
    cleaner.Update()

    connect = vtk.vtkPolyDataConnectivityFilter()
    if vtk.vtkVersion.GetVTKMajorVersion() > 5:
        connect.SetInputData(cleaner.GetOutput())
    else:
        connect.SetInput(cleaner.GetOutput())
    connect.Update()
    return connect.GetNumberOfExtractedRegions()

def extractclosestpointregion(polydata, point=[0, 0, 0]):
    # NOTE: preventive measures: clean before connectivity filter
    # to avoid artificial regionIds
    # It slices the surface down the middle
    surfer = vtk.vtkDataSetSurfaceFilter()
    if vtk.vtkVersion.GetVTKMajorVersion() > 5:
        surfer.SetInputData(polydata)
    else:
        surfer.SetInput(polydata)
    surfer.Update()

    cleaner = vtk.vtkCleanPolyData()
    if vtk.vtkVersion.GetVTKMajorVersion() > 5:
        cleaner.SetInputData(surfer.GetOutput())
    else:
        cleaner.SetInput(surfer.GetOutput())
    cleaner.Update()

    connect = vtk.vtkPolyDataConnectivityFilter()
    if vtk.vtkVersion.GetVTKMajorVersion() > 5:
        connect.SetInputData(cleaner.GetOutput())
    else:
        connect.SetInput(cleaner.GetOutput())
    connect.SetExtractionModeToClosestPointRegion()
    connect.SetClosestPoint(point)
    connect.Update()
    return connect.GetOutput()

def extractconnectedregion(polydata, regionid):
    """Extract connected region with label = regionid """
    surfer = vtk.vtkDataSetSurfaceFilter()
    if vtk.vtkVersion.GetVTKMajorVersion() > 5:
        surfer.SetInputData(polydata)
    else:
        surfer.SetInput(polydata)
    surfer.Update()

    cleaner = vtk.vtkCleanPolyData()
    if vtk.vtkVersion.GetVTKMajorVersion() > 5:
        cleaner.SetInputData(surfer.GetOutput())
    else:
        cleaner.SetInput(surfer.GetOutput())
    cleaner.Update()

    connect = vtk.vtkPolyDataConnectivityFilter()
    if vtk.vtkVersion.GetVTKMajorVersion() > 5:
        connect.SetInputData(cleaner.GetOutput())
    else:
        connect.SetInput(cleaner.GetOutput())

    connect.SetExtractionModeToAllRegions()
    connect.ColorRegionsOn()
    connect.Update()
    surface = pointthreshold(connect.GetOutput(), 'RegionId', float(regionid), float(regionid))
    return surface

def get_connected_components(polydata):
    """Extract all connected regions"""
    connect = vtk.vtkPolyDataConnectivityFilter()
    if vtk.vtkVersion.GetVTKMajorVersion() > 5:
        connect.SetInputData(polydata)
    else:
        connect.SetInput(polydata)
    connect.SetExtractionModeToAllRegions()
    connect.ColorRegionsOn()
    connect.Update()
    return connect

def find_create_path(mesh, p1, p2):
    """Get shortest path (using Dijkstra algorithm) between p1 and p2 on the mesh. Returns a polydata"""
    dijkstra = vtk.vtkDijkstraGraphGeodesicPath()
    if vtk.vtkVersion().GetVTKMajorVersion() > 5:
        dijkstra.SetInputData(mesh)
    else:
        dijkstra.SetInput(mesh)
    dijkstra.SetStartVertex(p1)
    dijkstra.SetEndVertex(p2)
    dijkstra.Update()
    return dijkstra.GetOutput()

def compute_geodesic_distance(mesh, id_p1, id_p2):
    """Compute geodesic distance from point id_p1 to id_p2 on surface 'mesh'
    It first computes the path across the edges and then the corresponding distance adding up point to point distances)"""
    path = find_create_path(mesh, id_p1, id_p2)
    total_dist = 0
    n = path.GetNumberOfPoints()
    for i in range(n-1):   # Ids are ordered in the new polydata, from 0 to npoints_in_path
        p0 = path.GetPoint(i)
        p1 = path.GetPoint(i+1)
        dist = math.sqrt(math.pow(p0[0]-p1[0], 2) + math.pow(p0[1]-p1[1], 2) + math.pow(p0[2]-p1[2], 2) )
        total_dist = total_dist + dist
    return total_dist, path

def transfer_array(ref, target, arrayname, targetarrayname):
    """Transfer scalar array using closest point approximation"""
    locator = vtk.vtkPointLocator()
    locator.SetDataSet(ref)
    locator.BuildLocator()

    refarray = ref.GetPointData().GetArray(arrayname)  # get array from reference

    numberofpoints = target.GetNumberOfPoints()
    newarray = vtk.vtkDoubleArray()
    newarray.SetName(targetarrayname)
    newarray.SetNumberOfTuples(numberofpoints)
    target.GetPointData().AddArray(newarray)

    # go through each point of target surface, determine closest point on surface, copy value
    for i in range(target.GetNumberOfPoints()):
        point = target.GetPoint(i)
        closestpoint_id = locator.FindClosestPoint(point)
        value = refarray.GetValue(closestpoint_id)
        newarray.SetValue(i, value)
    return target


def transfer_all_scalar_arrays(m1, m2):
    """ Transfer all scalar arrays from m1 to m2"""
    for i in range(m1.GetPointData().GetNumberOfArrays()):
        print('Transferring scalar array: {}'.format(m1.GetPointData().GetArray(i).GetName()))
        transfer_array(m1, m2, m1.GetPointData().GetArray(i).GetName(), m1.GetPointData().GetArray(i).GetName())


def transfer_all_scalar_arrays_by_point_id(m1, m2):
    """ Transfer all scalar arrays from m1 to m2 by point id"""
    for i in range(m1.GetPointData().GetNumberOfArrays()):
        print('Transferring scalar array: {}'.format(m1.GetPointData().GetArray(i).GetName()))
        m2.GetPointData().AddArray(m1.GetPointData().GetArray(i))


def transfer_info_points2cells_threshold(mesh, arrayname_points, arrayname_cells, threshold):
    # Create binary cell array. Compute sum of array_name_points in the 3 vertices.
    # Set cell to 1 if sum is higher than threshold, and set to 0 otherwise. Use threshold = 1.5 with scars.
    ncells = mesh.GetNumberOfCells()
    cell_array = vtk.vtkFloatArray()
    cell_array.SetName(arrayname_cells)
    cell_array.SetNumberOfComponents(1)
    cell_array.SetNumberOfTuples(ncells)

    for j in range(ncells):
        ptids = mesh.GetCell(j).GetPointIds()    # Get the 3 point ids
        if (ptids.GetNumberOfIds() != 3):
            print("Non triangular cell")
        val0 = mesh.GetPointData().GetArray(arrayname_points).GetTuple1(ptids.GetId(0))
        val1 = mesh.GetPointData().GetArray(arrayname_points).GetTuple1(ptids.GetId(1))
        val2 = mesh.GetPointData().GetArray(arrayname_points).GetTuple1(ptids.GetId(2))
        suma = (val0 + val1 + val2)
        if (suma > threshold):
            cell_array.SetTuple1(j, 1)
        else:
            cell_array.SetTuple1(j, 0)
    mesh.GetCellData().AddArray(cell_array)
    return mesh


def get_ordered_cont_ids_based_on_distance(mesh):
    """ Given a contour, get the ordered list of Ids (not ordered by default).
    Open the mesh duplicating the point with id = 0. Compute distance transform of point 0
    and get a ordered list of points (starting in 0) """
    m = vtk.vtkMath()
    m.RandomSeed(0)
    # copy the original mesh point by point
    points = vtk.vtkPoints()
    polys = vtk.vtkCellArray()
    cover = vtk.vtkPolyData()
    nver = mesh.GetNumberOfPoints()
    points.SetNumberOfPoints(nver+1)

    new_pid = nver  # id of the duplicated point
    added = False

    for j in range(mesh.GetNumberOfCells()):
        # get the 2 point ids
        ptids = mesh.GetCell(j).GetPointIds()
        cell = mesh.GetCell(j)
        if (ptids.GetNumberOfIds() != 2):
            # print "Non contour mesh (lines)"
            break

        # read the 2 involved points
        pid0 = ptids.GetId(0)
        pid1 = ptids.GetId(1)
        p0 = mesh.GetPoint(ptids.GetId(0))   # coordinates
        p1 = mesh.GetPoint(ptids.GetId(1))

        if pid0 == 0:
            if added == False:
                # Duplicate point 0. Add gaussian noise to the original point
                new_p = [p0[0] + m.Gaussian(0.0, 0.0005), p0[1] + m.Gaussian(0.0, 0.0005), p0[2] + m.Gaussian(0.0, 0.0005)]
                points.SetPoint(new_pid, new_p)
                points.SetPoint(pid1, p1)
                polys.InsertNextCell(2)
                polys.InsertCellPoint(pid1)
                polys.InsertCellPoint(new_pid)
                added = True
            else:  # act normal
                points.SetPoint(ptids.GetId(0), p0)
                points.SetPoint(ptids.GetId(1), p1)
                polys.InsertNextCell(2)
                polys.InsertCellPoint(cell.GetPointId(0))
                polys.InsertCellPoint(cell.GetPointId(1))
        elif pid1 == 0:
            if added == False:
                new_p = [p1[0] + m.Gaussian(0.0, 0.0005), p1[1] + m.Gaussian(0.0, 0.0005), p1[2] + m.Gaussian(0.0, 0.0005)]
                points.SetPoint(new_pid, new_p)
                points.SetPoint(pid0, p0)
                polys.InsertNextCell(2)
                polys.InsertCellPoint(pid0)
                polys.InsertCellPoint(new_pid)
                added = True
            else:  # act normal
                points.SetPoint(ptids.GetId(0), p0)
                points.SetPoint(ptids.GetId(1), p1)
                polys.InsertNextCell(2)
                polys.InsertCellPoint(cell.GetPointId(0))
                polys.InsertCellPoint(cell.GetPointId(1))

        else:
            points.SetPoint(ptids.GetId(0), p0)
            points.SetPoint(ptids.GetId(1), p1)
            polys.InsertNextCell(2)
            polys.InsertCellPoint(cell.GetPointId(0))
            polys.InsertCellPoint(cell.GetPointId(1))

    if added == False:
        print('Warning: I have not added any point, list of indexes may not be correct.')
    cover.SetPoints(points)
    cover.SetPolys(polys)
    if not vtk.vtkVersion.GetVTKMajorVersion() > 5:
        cover.Update()
    # compute distance from point with id 0 to all the rest
    npoints = cover.GetNumberOfPoints()
    dists = np.zeros(npoints)
    for i in range(npoints):
        [dists[i], poly] = compute_geodesic_distance(cover, int(0), i)
    list_ = np.argsort(dists).astype(int)
    return list_[0:len(list_)-1]    # skip last one, duplicated


def define_pv_segments_proportions(t_v5, t_v6, t_v7, alpha):
    """define number of points of each pv hole segment to ensure appropriate distribution"""
    props = np.zeros([4, 3])
    props[0, 0] = np.divide(1.0, 4.0)  # proportion of the total number of points of the pv contour according to the proportion of circle
    props[0, 1] = np.divide(1.0, 4.0) + t_v5 * np.divide(1.0, 2.0*np.pi)
    props[0, 2] = 1.0 - props[0, 0] - props[0, 1]
    # print('Proportions sum up:', props[0, 0]+props[0, 1]+props[0, 2])
    props[1, 0] = np.divide(t_v6, 2.0*np.pi) - np.divide(1.0, 2.0)
    props[1, 2] = np.divide(1.0, 4.0)  # s3
    props[1, 1] = 1.0 - props[1, 0] - props[1, 2]   # s2
    # print('Proportions sum up:', props[1, 0]+props[1, 1]+props[1, 2])
    props[2, 0] = np.divide(1.0, 4.0)
    props[2, 1] = np.divide(t_v7, 2.0*np.pi) - props[2, 0]
    props[2, 2] = 1.0 - props[2, 0] - props[2, 1]
    # print('Proportions sum up:', props[1, 0]+props[1, 1]+props[1, 2])
    props[3, 0] = np.divide(1.0, 4.0)   # a bit more if the LAA is displaced to the left
    props[3, 1] = np.divide(1.0, 2.0)   # a bit less if the LAA is displaced to the left
    # props[3, 0] = np.divide(1.0, 4.0) + alpha * np.divide(1.0, 2.0*np.pi)  # a bit more if the LAA is displaced to the left
    # props[3, 1] = np.divide(1.0, 2.0) - alpha * np.divide(1.0, 2.0*np.pi)  # a bit less if the LAA is displaced to the left
    props[3, 2] = np.divide(1.0, 4.0)
    return props

def define_disk_template(rdisk, rhole_rspv, rhole_ripv, rhole_lipv, rhole_lspv, rhole_laa, xhole_center, yhole_center,
                         laa_hole_center_x, laa_hole_center_y, t_v5, t_v6, t_v7, t_v8):
    """Define target positions in the disk template, return coordinates (x,y) corresponding to:
    v1r, v1d, v1l, v2u, v2r, v2l, v3u, v3r, v3l, v4r, v4u, v4d, vlaad, vlaau, p5, p6, p7, p8 """
    coordinates = np.zeros([2, 18])
    complete_circumf_t = np.linspace(0, 2 * np.pi, 1000, endpoint=False)
    rspv_hole_x = np.cos(complete_circumf_t) * rhole_rspv + xhole_center[0]
    rspv_hole_y = np.sin(complete_circumf_t) * rhole_rspv + yhole_center[0]
    ripv_hole_x = np.cos(complete_circumf_t) * rhole_ripv + xhole_center[1]
    ripv_hole_y = np.sin(complete_circumf_t) * rhole_ripv + yhole_center[1]
    lipv_hole_x = np.cos(complete_circumf_t) * rhole_lipv + xhole_center[2]
    lipv_hole_y = np.sin(complete_circumf_t) * rhole_lipv + yhole_center[2]
    lspv_hole_x = np.cos(complete_circumf_t) * rhole_lspv + xhole_center[3]
    lspv_hole_y = np.sin(complete_circumf_t) * rhole_lspv + yhole_center[3]
    laa_hole_x = np.cos(complete_circumf_t) * rhole_laa + laa_hole_center_x
    laa_hole_y = np.sin(complete_circumf_t) * rhole_laa + laa_hole_center_y
    # define (x,y) positions where I put v5, v6, v7 and v8
    coordinates[0, 14] = np.cos(t_v5) * rdisk  # p5_x
    coordinates[1, 14] = np.sin(t_v5) * rdisk  # p5_y
    coordinates[0, 15] = np.cos(t_v6) * rdisk  # p6_x
    coordinates[1, 15] = np.sin(t_v6) * rdisk  # p6_y
    coordinates[0, 16] = np.cos(t_v7) * rdisk  # p7_x
    coordinates[1, 16] = np.sin(t_v7) * rdisk  # p7_y
    coordinates[0, 17] = np.cos(t_v8) * rdisk  # p8_x
    coordinates[1, 17] = np.sin(t_v8) * rdisk  # p8_y

    # define target points corresponding to the pv holes
    # RSPV (right (in the line connecting to MV; left (horizontal line), down, vertical line))
    coordinates[0, 0] = rspv_hole_x[
        np.abs(complete_circumf_t - t_v5).argmin()]  # v1r_x, x in rspv circumf where angle is pi/4
    coordinates[1, 0] = rspv_hole_y[np.abs(complete_circumf_t - t_v5).argmin()]
    coordinates[0, 1] = rspv_hole_x[np.abs(complete_circumf_t - (3 * np.pi / 2)).argmin()]
    coordinates[1, 1] = rspv_hole_y[np.abs(complete_circumf_t - (3 * np.pi / 2)).argmin()]
    coordinates[0, 2] = rspv_hole_x[(np.abs(complete_circumf_t - np.pi)).argmin()]
    coordinates[1, 2] = rspv_hole_y[(np.abs(complete_circumf_t - np.pi)).argmin()]
    # RIPV
    coordinates[0, 3] = ripv_hole_x[np.abs(complete_circumf_t - (np.pi / 2)).argmin()]  # x in ripv circumf UP
    coordinates[1, 3] = ripv_hole_y[np.abs(complete_circumf_t - (np.pi / 2)).argmin()]
    coordinates[0, 4] = ripv_hole_x[np.abs(complete_circumf_t - t_v6).argmin()]
    coordinates[1, 4] = ripv_hole_y[np.abs(complete_circumf_t - t_v6).argmin()]
    coordinates[0, 5] = ripv_hole_x[np.abs(complete_circumf_t - (np.pi)).argmin()]
    coordinates[1, 5] = ripv_hole_y[np.abs(complete_circumf_t - (np.pi)).argmin()]
    # LIPV
    coordinates[0, 6] = lipv_hole_x[np.abs(complete_circumf_t - (np.pi / 2)).argmin()]
    coordinates[1, 6] = lipv_hole_y[np.abs(complete_circumf_t - (np.pi / 2)).argmin()]
    coordinates[0, 7] = lipv_hole_x[complete_circumf_t.argmin()]  # angle = 0
    coordinates[1, 7] = lipv_hole_y[complete_circumf_t.argmin()]
    coordinates[0, 8] = lipv_hole_x[np.abs(complete_circumf_t - t_v7).argmin()]
    coordinates[1, 8] = lipv_hole_y[np.abs(complete_circumf_t - t_v7).argmin()]
    # LSPV
    coordinates[0, 9] = lspv_hole_x[complete_circumf_t.argmin()]  # angle = 0
    coordinates[1, 9] = lspv_hole_y[complete_circumf_t.argmin()]
    coordinates[0, 10] = lspv_hole_x[np.abs(complete_circumf_t - (np.pi / 2)).argmin()]
    coordinates[1, 10] = lspv_hole_y[np.abs(complete_circumf_t - (np.pi / 2)).argmin()]
    coordinates[0, 11] = lspv_hole_x[np.abs(complete_circumf_t - (3 * np.pi / 2)).argmin()]
    coordinates[1, 11] = lspv_hole_y[np.abs(complete_circumf_t - (3 * np.pi / 2)).argmin()]
    # LAA
    coordinates[0, 12] = laa_hole_x[np.abs(complete_circumf_t - (3 * np.pi / 2)).argmin()]
    coordinates[1, 12] = laa_hole_y[np.abs(complete_circumf_t - (3 * np.pi / 2)).argmin()]
    coordinates[0, 13] = laa_hole_x[np.abs(complete_circumf_t - t_v8).argmin()]  # angle = pi/2 + pi/8
    coordinates[1, 13] = laa_hole_y[np.abs(complete_circumf_t - t_v8).argmin()]
    return coordinates

def get_coords(c):
    """Given all coordinates in a matrix, identify and return them separately"""
    return c[0,0], c[1,0], c[0,1], c[1,1], c[0,2], c[1,2], c[0,3], c[1,3], c[0,4], c[1,4], c[0,5], c[1,5], c[0,6], c[1,6], c[0,7], c[1,7], c[0,8], c[1,8], c[0,9], c[1,9], c[0,10], c[1,10], c[0,11], c[1,11], c[0,12], c[1,12], c[0,13], c[1,13], c[0,14], c[1,14], c[0,15], c[1,15], c[0,16], c[1,16], c[0,17], c[1,17]

def extract_LA_contours(m_open, filename, save=False):
    """Given LA with clipped PVs, LAA and MV identify and classify all 5 contours using 'autolabels' array.
    Save contours if save=True"""
    edges = extractboundaryedge(m_open)
    conn = get_connected_components(edges)
    poly_edges = conn.GetOutput()
    if save==True:
        writevtk(poly_edges, filename[0:len(filename) - 4] + '_detected_edges.vtk')

    print('Detected {} regions'.format(conn.GetNumberOfExtractedRegions()))
    if conn.GetNumberOfExtractedRegions() != 6:
        print('WARNING: the number of contours detected is not the expected. The classification of contours may be wrong')

    for i in range(6):
        print('Detecting region index: {}'.format(i))
        c = pointthreshold(poly_edges, 'RegionId', i, i)
        autolabels = vtk_to_numpy(c.GetPointData().GetArray('autolabels'))
        counts = np.bincount(autolabels.astype(int))
        mostcommon = np.argmax(counts)

        if mostcommon == 36:  # use the most repeated label since some of they are 36 (body). Can be 36 more common in the other regions?
            print('Detected MV')
            if save == True:
                writevtk(c, filename[0:len(filename) - 4] + '_cont_mv.vtk')
            cont_mv = c
        if mostcommon == 37:
            print('Detected LAA')
            if save == True:
                writevtk(c, filename[0:len(filename) - 4] + '_cont_laa.vtk')
            cont_laa = c
        if mostcommon == 76:
            print('Detected RSPV')
            if save == True:
                writevtk(c, filename[0:len(filename) - 4] + '_cont_rspv.vtk')
            cont_rspv = c
        if mostcommon == 77:
            print('Detected RIPV')
            if save == True:
                writevtk(c, filename[0:len(filename) - 4] + '_cont_ripv.vtk')
            cont_ripv = c
        if mostcommon == 78:
            print('Detected LSPV')
            if save == True:
                writevtk(c, filename[0:len(filename) - 4] + '_cont_lspv.vtk')
            cont_lspv = c
        if mostcommon == 79:
            print('Detected LIPV')
            if save == True:
                writevtk(c, filename[0:len(filename) - 4] + '_cont_lipv.vtk')
            cont_lipv = c
    return cont_rspv, cont_ripv, cont_lipv, cont_lspv, cont_mv, cont_laa

def build_locators(mesh, m_open, cont_rspv, cont_ripv, cont_lipv, cont_lspv, cont_laa):
    """Build different locators to find corresponding points between different meshes (open/closed, open/contours, etc)"""
    locator = vtk.vtkPointLocator()
    locator.SetDataSet(mesh)  # clipped + CLOSED - where the seeds are marked
    locator.BuildLocator()

    locator_open = vtk.vtkPointLocator()
    locator_open.SetDataSet(m_open)
    locator_open.BuildLocator()

    locator_rspv = vtk.vtkPointLocator()
    locator_rspv.SetDataSet(cont_rspv)
    locator_rspv.BuildLocator()

    locator_ripv = vtk.vtkPointLocator()
    locator_ripv.SetDataSet(cont_ripv)
    locator_ripv.BuildLocator()

    locator_lipv = vtk.vtkPointLocator()
    locator_lipv.SetDataSet(cont_lipv)
    locator_lipv.BuildLocator()

    locator_lspv = vtk.vtkPointLocator()
    locator_lspv.SetDataSet(cont_lspv)
    locator_lspv.BuildLocator()

    locator_laa = vtk.vtkPointLocator()
    locator_laa.SetDataSet(cont_laa)
    locator_laa.BuildLocator()
    return locator, locator_open, locator_rspv, locator_ripv, locator_lipv, locator_lspv, locator_laa

def read_paths(filename, npaths):
    """read the paths (lines) defined in the 3D mesh using 3_divide_LA.py"""
    for i in range(npaths):
        if os.path.isfile(filename[0:len(filename)-4]+'path'+ str(i+1) +'.vtk')==False:
            sys.exit('ERROR: dividing line path' + str(i+1) + ' not found. Run 3_divide_LA.py')
        else:
            if i == 0:
                path1 = readvtk(filename[0:len(filename) - 4] + 'path' + str(i + 1) + '.vtk')
            elif i == 1:
                path2 = readvtk(filename[0:len(filename) - 4] + 'path' + str(i + 1) + '.vtk')
            elif i == 2:
                path3 = readvtk(filename[0:len(filename) - 4] + 'path' + str(i + 1) + '.vtk')
            elif i == 3:
                path4 = readvtk(filename[0:len(filename) - 4] + 'path' + str(i + 1) + '.vtk')
            elif i == 4:
                path5 = readvtk(filename[0:len(filename) - 4] + 'path' + str(i + 1) + '.vtk')
            elif i == 5:
                path6 = readvtk(filename[0:len(filename) - 4] + 'path' + str(i + 1) + '.vtk')
            elif i == 6:
                path7 = readvtk(filename[0:len(filename) - 4] + 'path' + str(i + 1) + '.vtk')
    # read LAA related paths: line from lspv to laa and from laa to mv
    if os.path.isfile(filename[0:len(filename)-4] + 'path_laa1.vtk')==False:
        sys.exit('ERROR: dividing line path_laa1 not found. Run 3_divide_LA.py')
    else:
        path_laa1 = readvtk(filename[0:len(filename)-4] + 'path_laa1.vtk')

    if os.path.isfile(filename[0:len(filename)-4] + 'path_laa2.vtk')==False:
        sys.exit('ERROR: dividing line path_laa2 not found. Run 3_divide_LA.py')
    else:
        path_laa2 = readvtk(filename[0:len(filename)-4] + 'path_laa2.vtk')

    if os.path.isfile(filename[0:len(filename)-4] + 'path_laa3.vtk')==False:
        sys.exit('ERROR: dividing line path_laa3 not found. Run 3_divide_LA.py')
    else:
        path_laa3 = readvtk(filename[0:len(filename)-4] + 'path_laa3.vtk')
    return path1, path2, path3, path4, path5, path6, path7, path_laa1, path_laa2, path_laa3

def get_mv_contour_ids(cont_mv, locator_open):
    """Obtain ids of the MV contour"""
    edge_cont_ids = get_ordered_cont_ids_based_on_distance(cont_mv)
    mv_cont_ids = np.zeros(edge_cont_ids.size)
    for i in range(mv_cont_ids.shape[0]):
        p = cont_mv.GetPoint(edge_cont_ids[i])
        mv_cont_ids[i] = locator_open.FindClosestPoint(p)
    return mv_cont_ids

def identify_segments_extremes(path1, path2, path3, path4, path5, path6, path7, path_laa1, path_laa2, path_laa3,
                               locator_open, locator_rspv, locator_ripv, locator_lipv, locator_lspv, locator_laa,
                               cont_rspv, cont_ripv, cont_lipv, cont_lspv, cont_laa,
                               v5, v6, v7, v8):
    """Identify ids in the to_be_flat mesh corresponding to the segment extremes: v1d, v1r, ect."""
    # start with segments of PVs because they will modify the rest of segments (we try to have uniform number of points in the 3 segments of the veins)
    # first identify ALL pv segments extremes (v1d, v2u etc.)

    # s1 - Find ids corresponding to v1d and v2u as intersection of rspv (ripv) contour and path1
    dists1 = np.zeros(path1.GetNumberOfPoints())
    dists2 = np.zeros(path1.GetNumberOfPoints())
    for i in range(path1.GetNumberOfPoints()):
        p = path1.GetPoint(i)
        dists1[i] = euclideandistance(p, cont_rspv.GetPoint(locator_rspv.FindClosestPoint(p)))
        dists2[i] = euclideandistance(p, cont_ripv.GetPoint(locator_ripv.FindClosestPoint(p)))
    v1d_in_path1 = np.argmin(dists1)
    v2u_in_path1 = np.argmin(dists2)
    v1d = locator_open.FindClosestPoint(path1.GetPoint(v1d_in_path1))
    v2u = locator_open.FindClosestPoint(path1.GetPoint(v2u_in_path1))

    # s2 - Find 2l and v3r
    dists1 = np.zeros(path2.GetNumberOfPoints())
    dists2 = np.zeros(path2.GetNumberOfPoints())
    for i in range(path2.GetNumberOfPoints()):
        p = path2.GetPoint(i)
        dists1[i] = euclideandistance(p, cont_ripv.GetPoint(locator_ripv.FindClosestPoint(p)))
        dists2[i] = euclideandistance(p, cont_lipv.GetPoint(locator_lipv.FindClosestPoint(p)))
    v2l_in_path2 = np.argmin(dists1)
    v3r_in_path2 = np.argmin(dists2)
    v2l = locator_open.FindClosestPoint(path2.GetPoint(v2l_in_path2))
    v3r = locator_open.FindClosestPoint(path2.GetPoint(v3r_in_path2))

    # s3 - Find v3u and v4d
    dists1 = np.zeros(path3.GetNumberOfPoints())
    dists2 = np.zeros(path3.GetNumberOfPoints())
    for i in range(path3.GetNumberOfPoints()):
        p = path3.GetPoint(i)
        dists1[i] = euclideandistance(p, cont_lipv.GetPoint(locator_lipv.FindClosestPoint(p)))
        dists2[i] = euclideandistance(p, cont_lspv.GetPoint(locator_lspv.FindClosestPoint(p)))
    v3u_in_path3 = np.argmin(dists1)
    v4d_in_path3 = np.argmin(dists2)
    v3u = locator_open.FindClosestPoint(path3.GetPoint(v3u_in_path3))
    v4d = locator_open.FindClosestPoint(path3.GetPoint(v4d_in_path3))

    # s4 - Find v4r and v1l
    dists1 = np.zeros(path4.GetNumberOfPoints())
    dists2 = np.zeros(path4.GetNumberOfPoints())
    for i in range(path4.GetNumberOfPoints()):
        p = path4.GetPoint(i)
        dists1[i] = euclideandistance(p, cont_lspv.GetPoint(locator_lspv.FindClosestPoint(p)))
        dists2[i] = euclideandistance(p, cont_rspv.GetPoint(locator_rspv.FindClosestPoint(p)))
    v4r_in_path4 = np.argmin(dists1)
    v1l_in_path4 = np.argmin(dists2)
    v4r = locator_open.FindClosestPoint(path4.GetPoint(v4r_in_path4))
    v1l = locator_open.FindClosestPoint(path4.GetPoint(v1l_in_path4))

    # find ids in the MV
    id_v5 = locator_open.FindClosestPoint(v5)
    id_v6 = locator_open.FindClosestPoint(v6)
    id_v7 = locator_open.FindClosestPoint(v7)
    id_v8 = locator_open.FindClosestPoint(v8)

    # Next 4 segments: s5, s6, s7, s8 : FROM pvs (v1r,v2r,v3l,v4l) TO points in the MV
    dists1 = np.zeros(path5.GetNumberOfPoints())
    for i in range(path5.GetNumberOfPoints()):
        p = path5.GetPoint(i)
        dists1[i] = euclideandistance(p, cont_rspv.GetPoint(locator_rspv.FindClosestPoint(p)))
    v1r_in_path5 = np.argmin(dists1)
    v1r = locator_open.FindClosestPoint(path5.GetPoint(v1r_in_path5))

    # s6
    dists1 = np.zeros(path6.GetNumberOfPoints())
    for i in range(path6.GetNumberOfPoints()):
        p = path6.GetPoint(i)
        dists1[i] = euclideandistance(p, cont_ripv.GetPoint(locator_ripv.FindClosestPoint(p)))
    v2r_in_path6 = np.argmin(dists1)
    v2r = locator_open.FindClosestPoint(path6.GetPoint(v2r_in_path6))

    # s7
    dists1 = np.zeros(path7.GetNumberOfPoints())
    for i in range(path7.GetNumberOfPoints()):
        p = path7.GetPoint(i)
        dists1[i] = euclideandistance(p, cont_lipv.GetPoint(locator_lipv.FindClosestPoint(p)))
    v3l_in_path7 = np.argmin(dists1)
    v3l = locator_open.FindClosestPoint(path7.GetPoint(v3l_in_path7))

    # S8a -> segment from v4 (lspv) to LAA
    dists1 = np.zeros(path_laa1.GetNumberOfPoints())
    dists2 = np.zeros(path_laa1.GetNumberOfPoints())
    for i in range(path_laa1.GetNumberOfPoints()):
        p = path_laa1.GetPoint(i)
        dists1[i] = euclideandistance(p, cont_lspv.GetPoint(locator_lspv.FindClosestPoint(p)))
        dists2[i] = euclideandistance(p, cont_laa.GetPoint(locator_laa.FindClosestPoint(p)))
    v4u_in_pathlaa1 = np.argmin(dists1)
    vlaad_in_pathlaa1 = np.argmin(dists2)
    v4u = locator_open.FindClosestPoint(path_laa1.GetPoint(v4u_in_pathlaa1))
    vlaad = locator_open.FindClosestPoint(path_laa1.GetPoint(vlaad_in_pathlaa1))

    # S8b -> segment from LAA to V8 (MV)
    dists1 = np.zeros(path_laa2.GetNumberOfPoints())
    for i in range(path_laa2.GetNumberOfPoints()):
        p = path_laa2.GetPoint(i)
        dists1[i] = euclideandistance(p, cont_laa.GetPoint(locator_laa.FindClosestPoint(p)))
    vlaau_in_pathlaa2 = np.argmin(dists1)
    vlaau = locator_open.FindClosestPoint(path_laa2.GetPoint(vlaau_in_pathlaa2))

    # aux point vlaar (connecting laa and rspv - auxiliary to know laa contour direction)
    dists1 = np.zeros(path_laa3.GetNumberOfPoints())
    for i in range(path_laa3.GetNumberOfPoints()):
        p = path_laa3.GetPoint(i)
        dists1[i] = euclideandistance(p, cont_laa.GetPoint(locator_laa.FindClosestPoint(p)))
    vlaar_in_pathlaa3 = np.argmin(dists1)
    vlaar = locator_open.FindClosestPoint(path_laa3.GetPoint(vlaar_in_pathlaa3))
    return v1r, v1d, v1l, v2u, v2r, v2l, v3u, v3r, v3l, v4r, v4u, v4d, vlaad, vlaau, vlaar, id_v5, id_v6, id_v7, id_v8

def get_rspv_segments_ids(cont_rspv, locator_open, v1l, v1d, v1r, propn_rspv_s1, propn_rspv_s2, propn_rspv_s3):
    """ Return 3 arrays with ids of each of the 3 segments in rspv contour.
        Return also the modified (to have proportional number of points in the segments) extreme ids"""
    edge_cont_rspv = get_ordered_cont_ids_based_on_distance(cont_rspv)
    rspv_cont_ids = np.zeros(edge_cont_rspv.size)
    for i in range(rspv_cont_ids.shape[0]):
        p = cont_rspv.GetPoint(edge_cont_rspv[i])
        rspv_cont_ids[i] = locator_open.FindClosestPoint(p)
    pos_v1l = int(np.where(rspv_cont_ids == v1l)[0])
    rspv_ids = np.append(rspv_cont_ids[pos_v1l:rspv_cont_ids.size], rspv_cont_ids[0:pos_v1l])
    pos_v1d = int(np.where(rspv_ids == v1d)[0])
    pos_v1r = int(np.where(rspv_ids == v1r)[0])
    if pos_v1r < pos_v1d:   # flip
        aux = np.zeros(rspv_ids.size)
        for i in range(rspv_ids.size):
            aux[rspv_ids.size - 1 - i] = rspv_ids[i]
        # maintain the v1l as the first one (after the flip is the last one)
        flipped = np.append(aux[aux.size - 1], aux[0:aux.size - 1])
        rspv_ids = flipped.astype(int)
    rspv_s1 = rspv_ids[0:int(np.where(rspv_ids == v1d)[0])]
    rspv_s2 = rspv_ids[int(np.where(rspv_ids == v1d)[0]): int(np.where(rspv_ids == v1r)[0])]
    rspv_s3 = rspv_ids[int(np.where(rspv_ids == v1r)[0]): rspv_ids.size]

    # # correct to have proportional segments length
    # s1_prop_length = round(propn_rspv_s1*len(rspv_ids))
    # s2_prop_length = round(propn_rspv_s2*len(rspv_ids))
    # s3_prop_length = round(propn_rspv_s3*len(rspv_ids))
    # v1l_prop = v1l   # stays the same, reference
    # v1d_prop = rspv_ids[int(s1_prop_length)]
    # v1r_prop = rspv_ids[int(s1_prop_length + s2_prop_length)]
    # rspv_s1_prop = rspv_ids[0:int(s1_prop_length)]
    # rspv_s2_prop = rspv_ids[int(s1_prop_length): int(s1_prop_length + s2_prop_length)]
    # rspv_s3_prop = rspv_ids[int(s1_prop_length + s2_prop_length): rspv_ids.size]

    # INTERMEDIATE (final) solution. Offset
    s1_prop_length = round(propn_rspv_s1 * len(rspv_ids))
    s2_prop_length = round(propn_rspv_s2 * len(rspv_ids))
    s3_prop_length = round(propn_rspv_s3 * len(rspv_ids))
    v1l_prop = v1l   # stays the same, reference
    rspv_s1_offset = round((s1_prop_length - rspv_s1.size)/2)        # If negative, I'll shorten s1 in that case
    v1d_prop = rspv_ids[int(rspv_s1.size + rspv_s1_offset)]
    rspv_s1_prop = rspv_ids[0:int(rspv_s1.size + rspv_s1_offset)]
    new_s2_size = rspv_s2.size - rspv_s1_offset   # initial minus points now given to s1
    rspv_s2_offset = np.floor((s2_prop_length - new_s2_size)/2)    # I will add an offset of half the difference. Floor, otherwise s3 is always shorter since it is the remaining part
    v1r_prop = rspv_ids[int(rspv_s1_prop.size + new_s2_size + rspv_s2_offset)]
    rspv_s2_prop = rspv_ids[int(rspv_s1.size + rspv_s1_offset):int(rspv_s1.size + rspv_s1_offset + new_s2_size + rspv_s2_offset)]
    rspv_s3_prop = rspv_ids[int(rspv_s1.size + rspv_s1_offset + new_s2_size + rspv_s2_offset): rspv_ids.size]
    # # print('RSPV original lengths', rspv_s1.size, rspv_s2.size, rspv_s3.size)
    # # print('Proportional lengths', rspv_s1_prop.size, rspv_s2_prop.size, rspv_s3_prop.size)
    return rspv_ids, rspv_s1_prop, rspv_s2_prop, rspv_s3_prop, v1l_prop, v1d_prop, v1r_prop

def get_ripv_segments_ids(cont_ripv, locator_open, v2l, v2r, v2u, propn_ripv_s1, propn_ripv_s2, propn_ripv_s3):
    """ Return 3 arrays with ids of each of the 3 segments in ripv contour.
        Return also the modified (to have proportional number of points in the segments) extreme ids"""
    edge_cont_ripv = get_ordered_cont_ids_based_on_distance(cont_ripv)
    ripv_cont_ids = np.zeros(edge_cont_ripv.size)
    for i in range(ripv_cont_ids.shape[0]):
        p = cont_ripv.GetPoint(edge_cont_ripv[i])
        ripv_cont_ids[i] = locator_open.FindClosestPoint(p)
    pos_v2l = int(np.where(ripv_cont_ids == v2l)[0])
    ripv_ids = np.append(ripv_cont_ids[pos_v2l:ripv_cont_ids.size], ripv_cont_ids[0:pos_v2l])
    pos_v2r = int(np.where(ripv_ids == v2r)[0])
    pos_v2u = int(np.where(ripv_ids == v2u)[0])
    if pos_v2u < pos_v2r:  # flip
        aux = np.zeros(ripv_ids.size)
        for i in range(ripv_ids.size):
            aux[ripv_ids.size - 1 - i] = ripv_ids[i]
        flipped = np.append(aux[aux.size - 1], aux[0:aux.size - 1])
        ripv_ids = flipped.astype(int)
    ripv_s1 = ripv_ids[0:int(np.where(ripv_ids == v2r)[0])]
    ripv_s2 = ripv_ids[int(np.where(ripv_ids == v2r)[0]): int(np.where(ripv_ids == v2u)[0])]
    ripv_s3 = ripv_ids[int(np.where(ripv_ids == v2u)[0]): ripv_ids.size]

    # # # correct to have proportional segments length
    # s1_prop_length = round(propn_ripv_s1 * len(ripv_ids))
    # s2_prop_length = round(propn_ripv_s2 * len(ripv_ids))
    # s3_prop_length = round(propn_ripv_s3 * len(ripv_ids))
    # v2l_prop = v2l  # stays the same, reference
    # v2r_prop = ripv_ids[int(s1_prop_length)]
    # v2u_prop = ripv_ids[int(s1_prop_length + s2_prop_length)]
    # ripv_s1_prop = ripv_ids[0:int(s1_prop_length)]
    # ripv_s2_prop = ripv_ids[int(s1_prop_length): int(s1_prop_length + s2_prop_length)]
    # ripv_s3_prop = ripv_ids[int(s1_prop_length + s2_prop_length): ripv_ids.size]

    # INTERMEDIATE solution.
    s1_prop_length = round(propn_ripv_s1 * len(ripv_ids))
    s2_prop_length = round(propn_ripv_s2 * len(ripv_ids))
    s3_prop_length = round(propn_ripv_s3 * len(ripv_ids))
    v2l_prop = v2l   # stays the same, reference
    ripv_s1_offset = round((s1_prop_length - ripv_s1.size)/2)
    v2r_prop = ripv_ids[int(ripv_s1.size + ripv_s1_offset)]
    ripv_s1_prop = ripv_ids[0:int(ripv_s1.size + ripv_s1_offset)]
    new_s2_size = ripv_s2.size - ripv_s1_offset
    ripv_s2_offset = np.floor((s2_prop_length - new_s2_size)/2)
    v2u_prop = ripv_ids[int(ripv_s1_prop.size + new_s2_size + ripv_s2_offset)]
    ripv_s2_prop = ripv_ids[int(ripv_s1.size + ripv_s1_offset):int(ripv_s1.size + ripv_s1_offset + new_s2_size + ripv_s2_offset)]
    ripv_s3_prop = ripv_ids[int(ripv_s1.size + ripv_s1_offset + new_s2_size + ripv_s2_offset): ripv_ids.size]
    # print('RIPV original lengths', ripv_s1.size, ripv_s2.size, ripv_s3.size)
    # print('Proportional lengths', ripv_s1_prop.size, ripv_s2_prop.size, ripv_s3_prop.size)
    return ripv_ids, ripv_s1_prop, ripv_s2_prop, ripv_s3_prop, v2l_prop, v2r_prop, v2u_prop

def get_lipv_segments_ids(cont_lipv, locator_open, v3r, v3u, v3l, propn_lipv_s1, propn_lipv_s2, propn_lipv_s3):
    """ Return 3 arrays with ids of each of the 3 segments in lipv contour.
        Return also the modified (to have proportional number of points in the segments) extreme ids"""
    edge_cont_lipv = get_ordered_cont_ids_based_on_distance(cont_lipv)
    lipv_cont_ids = np.zeros(edge_cont_lipv.size)
    for i in range(lipv_cont_ids.shape[0]):
        p = cont_lipv.GetPoint(edge_cont_lipv[i])
        lipv_cont_ids[i] = locator_open.FindClosestPoint(p)
    pos_v3r = int(np.where(lipv_cont_ids == v3r)[0])
    lipv_ids = np.append(lipv_cont_ids[pos_v3r:lipv_cont_ids.size], lipv_cont_ids[0:pos_v3r])
    pos_v3u = int(np.where(lipv_ids == v3u)[0])
    pos_v3l = int(np.where(lipv_ids == v3l)[0])
    if pos_v3l < pos_v3u:  # flip
        aux = np.zeros(lipv_ids.size)
        for i in range(lipv_ids.size):
            aux[lipv_ids.size - 1 - i] = lipv_ids[i]
        flipped = np.append(aux[aux.size - 1], aux[0:aux.size - 1])
        lipv_ids = flipped.astype(int)
    lipv_s1 = lipv_ids[0:int(np.where(lipv_ids == v3u)[0])]
    lipv_s2 = lipv_ids[int(np.where(lipv_ids == v3u)[0]): int(np.where(lipv_ids == v3l)[0])]
    lipv_s3 = lipv_ids[int(np.where(lipv_ids == v3l)[0]): lipv_ids.size]

    # # # correct to have proportional segments length
    # s1_prop_length = round(propn_lipv_s1 * len(lipv_ids))
    # s2_prop_length = round(propn_lipv_s2 * len(lipv_ids))
    # s3_prop_length = round(propn_lipv_s3 * len(lipv_ids))
    # v3r_prop = v3r  # stays the same, reference
    # v3u_prop = lipv_ids[int(s1_prop_length)]
    # v3l_prop = lipv_ids[int(s1_prop_length + s2_prop_length)]
    # lipv_s1_prop = lipv_ids[0:int(s1_prop_length)]
    # lipv_s2_prop = lipv_ids[int(s1_prop_length): int(s1_prop_length + s2_prop_length)]
    # lipv_s3_prop = lipv_ids[int(s1_prop_length + s2_prop_length): lipv_ids.size]

    # INTERMEDIATE solution.
    s1_prop_length = round(propn_lipv_s1 * len(lipv_ids))
    s2_prop_length = round(propn_lipv_s2 * len(lipv_ids))
    s3_prop_length = round(propn_lipv_s3 * len(lipv_ids))
    v3r_prop = v3r   # stays the same, reference
    lipv_s1_offset = round((s1_prop_length - lipv_s1.size)/2)
    v3u_prop = lipv_ids[int(lipv_s1.size + lipv_s1_offset)]
    lipv_s1_prop = lipv_ids[0:int(lipv_s1.size + lipv_s1_offset)]
    new_s2_size = lipv_s2.size - lipv_s1_offset
    lipv_s2_offset = np.floor((s2_prop_length - new_s2_size)/2)
    v3l_prop = lipv_ids[int(lipv_s1_prop.size + new_s2_size + lipv_s2_offset)]
    lipv_s2_prop = lipv_ids[int(lipv_s1.size + lipv_s1_offset):int(lipv_s1.size + lipv_s1_offset + new_s2_size + lipv_s2_offset)]
    lipv_s3_prop = lipv_ids[int(lipv_s1.size + lipv_s1_offset + new_s2_size + lipv_s2_offset): lipv_ids.size]
    # print('LIPV original lengths', lipv_s1.size, lipv_s2.size, lipv_s3.size)
    # print('Proportional lengths', lipv_s1_prop.size, lipv_s2_prop.size, lipv_s3_prop.size)
    return lipv_ids, lipv_s1_prop, lipv_s2_prop, lipv_s3_prop, v3r_prop, v3u_prop, v3l_prop

def get_lspv_segments_ids(cont_lspv, locator_open, v4r, v4u, v4d, propn_lspv_s1, propn_lspv_s2, propn_lspv_s3):
    """ Return 3 arrays with ids of each of the 3 segments in lspv contour.
        Return also the modified (to have proportional number of points in the segments) extreme ids"""
    edge_cont_lspv = get_ordered_cont_ids_based_on_distance(cont_lspv)
    lspv_cont_ids = np.zeros(edge_cont_lspv.size)
    for i in range(lspv_cont_ids.shape[0]):
        p = cont_lspv.GetPoint(edge_cont_lspv[i])
        lspv_cont_ids[i] = locator_open.FindClosestPoint(p)
    pos_v4r = int(np.where(lspv_cont_ids == v4r)[0])
    lspv_ids = np.append(lspv_cont_ids[pos_v4r:lspv_cont_ids.size], lspv_cont_ids[0:pos_v4r])
    pos_v4u = int(np.where(lspv_ids == v4u)[0])
    pos_v4d = int(np.where(lspv_ids == v4d)[0])
    if pos_v4d < pos_v4u:   # flip
        aux = np.zeros(lspv_ids.size)
        for i in range(lspv_ids.size):
            aux[lspv_ids.size - 1 - i] = lspv_ids[i]
        flipped = np.append(aux[aux.size - 1], aux[0:aux.size - 1])
        lspv_ids = flipped.astype(int)
    lspv_s1 = lspv_ids[0:int(np.where(lspv_ids == v4u)[0])]
    lspv_s2 = lspv_ids[int(np.where(lspv_ids == v4u)[0]): int(np.where(lspv_ids == v4d)[0])]
    lspv_s3 = lspv_ids[int(np.where(lspv_ids == v4d)[0]): lspv_ids.size]

    ## correct to have proportional segments length
    # s1_prop_length = round(propn_lspv_s1*len(lspv_ids))
    # s2_prop_length = round(propn_lspv_s2*len(lspv_ids))
    # s3_prop_length = round(propn_lspv_s3*len(lspv_ids))
    # v4r_prop = v4r   # stays the same, reference
    # v4u_prop = lspv_ids[int(s1_prop_length)]
    # v4d_prop = lspv_ids[int(s1_prop_length + s2_prop_length)]
    # lspv_s1_prop = lspv_ids[0:int(s1_prop_length)]
    # lspv_s2_prop = lspv_ids[int(s1_prop_length): int(s1_prop_length + s2_prop_length)]
    # lspv_s3_prop = lspv_ids[int(s1_prop_length + s2_prop_length): lspv_ids.size]

    # INTERMEDIATE solution.
    s1_prop_length = round(propn_lspv_s1*len(lspv_ids))
    s2_prop_length = round(propn_lspv_s2*len(lspv_ids))
    s3_prop_length = round(propn_lspv_s3*len(lspv_ids))
    v4r_prop = v4r   # stays the same, reference
    lspv_s1_offset = round((s1_prop_length - lspv_s1.size)/2)
    v4u_prop = lspv_ids[int(lspv_s1.size + lspv_s1_offset)]
    lspv_s1_prop = lspv_ids[0:int(lspv_s1.size + lspv_s1_offset)]
    new_s2_size = lspv_s2.size - lspv_s1_offset
    lspv_s2_offset = np.floor((s2_prop_length - new_s2_size)/2)
    v4d_prop = lspv_ids[int(lspv_s1_prop.size + new_s2_size + lspv_s2_offset)]
    lspv_s2_prop = lspv_ids[int(lspv_s1.size + lspv_s1_offset):int(lspv_s1.size + lspv_s1_offset + new_s2_size + lspv_s2_offset)]
    lspv_s3_prop = lspv_ids[int(lspv_s1.size + lspv_s1_offset + new_s2_size + lspv_s2_offset): lspv_ids.size]
    # print('LSPV Original lengths', lspv_s1.size, lspv_s2.size, lspv_s3.size)
    # print('Proportional lengths', lspv_s1_prop.size, lspv_s2_prop.size, lspv_s3_prop.size)
    return lspv_ids, lspv_s1_prop, lspv_s2_prop, lspv_s3_prop, v4r_prop, v4u_prop, v4d_prop

def get_laa_segments_ids(cont_laa, locator_open, vlaau, vlaad, vlaar):
    """ Return 2 arrays with ids of each of the 2 segments in LAA contour."""
    edge_cont_laa = get_ordered_cont_ids_based_on_distance(cont_laa)
    laa_cont_ids = np.zeros(edge_cont_laa.size)
    for i in range(laa_cont_ids.shape[0]):
        p = cont_laa.GetPoint(edge_cont_laa[i])
        laa_cont_ids[i] = locator_open.FindClosestPoint(p)
    pos_vlaad = int(np.where(laa_cont_ids == vlaad)[0])  # intersection of laa contour and path 8a (from lspv to laa)
    laa_ids = np.append(laa_cont_ids[pos_vlaad:laa_cont_ids.size], laa_cont_ids[0:pos_vlaad])

    pos_vlaar = int(np.where(laa_ids == vlaar)[0])
    pos_vlaau = int(np.where(laa_ids == vlaau)[0])
    if pos_vlaau < pos_vlaar:  # flip
        aux = np.zeros(laa_ids.size)
        for i in range(laa_ids.size):
            aux[laa_ids.size - 1 - i] = laa_ids[i]
        flipped = np.append(aux[aux.size - 1], aux[0:aux.size - 1])
        laa_ids = flipped.astype(int)

    laa_s1 = laa_ids[0:int(np.where(laa_ids == vlaau)[0])]
    laa_s2 = laa_ids[int(np.where(laa_ids == vlaau)[0]): laa_ids.size]
    return laa_ids, laa_s1, laa_s2

def get_segment_ids_in_to_be_flat_mesh(path, locator, intersect_end, intersect_beginning):
    s = np.zeros(path.GetNumberOfPoints())
    for i in range(path.GetNumberOfPoints()):
        p = path.GetPoint(i)
        s[i] = int(locator.FindClosestPoint(p))
    intersect_wlast = np.intersect1d(s, intersect_end)   # find repeated values (s1 merges with rspv contour)
    nlasts_to_delete = len(intersect_wlast)
    index1 = np.arange(len(s) - nlasts_to_delete, len(s))
    final_s = np.delete(s, index1)

    intersect_wfirst = np.intersect1d(final_s, intersect_beginning)
    nfirst_to_delete = len(intersect_wfirst)
    index2 = np.arange(0, nfirst_to_delete)
    s = np.delete(final_s, index2)
    return s

def define_boundary_positions(rdisk, rhole_rspv, rhole_ripv, rhole_lipv, rhole_lspv, rhole_laa, xhole_center, yhole_center, laa_hole_center_x, laa_hole_center_y,
                              s9size, s10size, s11size, s12size, pv_laa_segment_lengths, t_v5, t_v6, t_v7, t_v8):
    """Define BOUNDARY target (x0,y0) coordinates given template parameters (hole radii and positions) and number of points of segments"""
    p_bound = s9size + s10size + s11size + s12size + np.sum(pv_laa_segment_lengths)
    x0_bound = np.zeros(int(p_bound))
    y0_bound = np.zeros(int(p_bound))
    # start with BOUNDARY (disk contour) 4 segments of the mv <-> contour of the disk
    # s9: left
    ind1 = 0
    ind2 = s9size
    t = np.linspace(-(2*np.pi - t_v6), t_v5, s9size+1, endpoint=True)   # +1 because later I will exclude the last point
    # flip to have clock wise direction in the angle
    aux = np.zeros(t.size)
    for i in range(t.size):
        aux[t.size-1-i] = t[i]
    t = aux
    final_t = t[0:len(t)-1]  # exclude extreme, only one, last
    x0_bound[ind1: ind2] = np.cos(final_t) * rdisk
    y0_bound[ind1: ind2] = np.sin(final_t) * rdisk

    # s10: bottom
    ind1 = ind2
    ind2 = ind2 + s10size
    t = np.linspace(t_v7, t_v6, s10size+1, endpoint=True)
    # flip to have clock wise direction in the angle
    aux = np.zeros(t.size)
    for i in range(t.size):
        aux[t.size-1-i] = t[i]
    t = aux
    final_t = t[0:len(t)-1]  # exclude extreme, only one, last
    x0_bound[ind1: ind2] = np.cos(final_t) * rdisk
    y0_bound[ind1: ind2] = np.sin(final_t) * rdisk

    # s11: left - from v7 to v8
    ind1 = ind2
    ind2 = ind2 + s11size
    t = np.linspace(t_v8, t_v7, s11size+1, endpoint=True)
    # flip to have clock wise direction in the angle
    aux = np.zeros(t.size)
    for i in range(t.size):
        aux[t.size-1-i] = t[i]
    t = aux
    final_t = t[0:len(t)-1]  # exclude extreme, only one, last
    x0_bound[ind1: ind2] = np.cos(final_t) * rdisk
    y0_bound[ind1: ind2] = np.sin(final_t) * rdisk

    # s12: top
    ind1 = ind2
    ind2 = ind2 + s12size
    t = np.linspace(t_v5, t_v8, s12size+1, endpoint=True)
    # flip to have clock wise direction in the angle
    aux = np.zeros(t.size)
    for i in range(t.size):
        aux[t.size-1-i] = t[i]
    t = aux
    final_t = t[0:len(t)-1]  # exclude extreme, only one, last
    x0_bound[ind1: ind2] = np.cos(final_t) * rdisk
    y0_bound[ind1: ind2] = np.sin(final_t) * rdisk

    # PV HOLES
    # RSPV, starts in pi
    # rspv_s1
    ind1 = ind2
    ind2 = ind2 + pv_laa_segment_lengths[0, 0]
    t = np.linspace(np.pi, 3*np.pi/2, pv_laa_segment_lengths[0, 0]+1, endpoint=True)  # skip last one later
    t = t[0:len(t)-1]
    x0_bound[ind1: ind2] = np.cos(t) * rhole_rspv + xhole_center[0]
    y0_bound[ind1: ind2] = np.sin(t) * rhole_rspv + yhole_center[0]
    # rspv_s2
    ind1 = ind2
    ind2 = ind2 + pv_laa_segment_lengths[0,1]
    t = np.linspace(3*np.pi/2, t_v5 + 2*np.pi, pv_laa_segment_lengths[0, 1]+1, endpoint=True)  # skip last one later
    t = t[0:len(t)-1]
    x0_bound[ind1: ind2] = np.cos(t) * rhole_rspv + xhole_center[0]
    y0_bound[ind1: ind2] = np.sin(t) * rhole_rspv + yhole_center[0]
    # rspv_s3
    ind1 = ind2
    ind2 = ind2 + pv_laa_segment_lengths[0,2]
    t = np.linspace(t_v5, np.pi, pv_laa_segment_lengths[0, 2]+1, endpoint=True)  # skip last one later
    t = t[0:len(t)-1]
    x0_bound[ind1: ind2] = np.cos(t) * rhole_rspv + xhole_center[0]
    y0_bound[ind1: ind2] = np.sin(t) * rhole_rspv + yhole_center[0]

    # RIPV, starts in pi
    ind1 = ind2
    ind2 = ind2 + pv_laa_segment_lengths[1,0]
    t = np.linspace(np.pi, t_v6, pv_laa_segment_lengths[1, 0]+1, endpoint=True)  # skip last one later
    t = t[0:len(t)-1]
    x0_bound[ind1: ind2] = np.cos(t) * rhole_ripv + xhole_center[1]
    y0_bound[ind1: ind2] = np.sin(t) * rhole_ripv + yhole_center[1]
    # ripv_s2
    ind1 = ind2
    ind2 = ind2 + pv_laa_segment_lengths[1,1]
    t = np.linspace(t_v6, 2*np.pi + np.pi/2, pv_laa_segment_lengths[1, 1]+1, endpoint=True)  # skip last one later
    t = t[0:len(t)-1]
    x0_bound[ind1: ind2] = np.cos(t) * rhole_ripv + xhole_center[1]
    y0_bound[ind1: ind2] = np.sin(t) * rhole_ripv + yhole_center[1]
    # ripv_s3
    ind1 = ind2
    ind2 = ind2 + pv_laa_segment_lengths[1,2]
    t = np.linspace(np.pi/2, np.pi, pv_laa_segment_lengths[1, 2]+1, endpoint=True)  # skip last one later
    t = t[0:len(t)-1]
    x0_bound[ind1: ind2] = np.cos(t) * rhole_ripv + xhole_center[1]
    y0_bound[ind1: ind2] = np.sin(t) * rhole_ripv + yhole_center[1]

    # LIPV, starts in 0
    # lipv_s1
    ind1 = ind2
    ind2 = ind2 + pv_laa_segment_lengths[2, 0]
    t = np.linspace(0, np.pi/2, pv_laa_segment_lengths[2, 0]+1, endpoint=True)  # skip last one later
    t = t[0:len(t)-1]
    x0_bound[ind1: ind2] = np.cos(t) * rhole_lipv + xhole_center[2]
    y0_bound[ind1: ind2] = np.sin(t) * rhole_lipv + yhole_center[2]
    # lipv_s2
    ind1 = ind2
    ind2 = ind2 + pv_laa_segment_lengths[2, 1]
    t = np.linspace(np.pi/2, t_v7, pv_laa_segment_lengths[2, 1]+1, endpoint=True)  # skip last one later
    t = t[0:len(t)-1]
    x0_bound[ind1: ind2] = np.cos(t) * rhole_lipv + xhole_center[2]
    y0_bound[ind1: ind2] = np.sin(t) * rhole_lipv + yhole_center[2]
    # lipv_s3
    ind1 = ind2
    ind2 = ind2 + pv_laa_segment_lengths[2, 2]
    t = np.linspace(t_v7, 2*np.pi, pv_laa_segment_lengths[2, 2]+1, endpoint=True)  # skip last one later
    t = t[0:len(t)-1]
    x0_bound[ind1: ind2] = np.cos(t) * rhole_lipv + xhole_center[2]
    y0_bound[ind1: ind2] = np.sin(t) * rhole_lipv + yhole_center[2]

    # LSPV, starts in 0
    # lspv_s1
    ind1 = ind2
    ind2 = ind2 + pv_laa_segment_lengths[3, 0]
    t = np.linspace(0, np.pi/2, pv_laa_segment_lengths[3, 0]+1, endpoint=True)  # skip last one later
    t = t[0:len(t)-1]
    x0_bound[ind1: ind2] = np.cos(t) * rhole_lspv + xhole_center[3]
    y0_bound[ind1: ind2] = np.sin(t) * rhole_lspv + yhole_center[3]
    # lspv_s2
    ind1 = ind2
    ind2 = ind2 + pv_laa_segment_lengths[3, 1]
    t = np.linspace(np.pi/2, 3*np.pi/2, pv_laa_segment_lengths[3, 1]+1, endpoint=True)  # skip last one later
    t = t[0:len(t)-1]
    x0_bound[ind1: ind2] = np.cos(t) * rhole_lspv + xhole_center[3]
    y0_bound[ind1: ind2] = np.sin(t) * rhole_lspv + yhole_center[3]
    # lspv_s3
    ind1 = ind2
    ind2 = ind2 + pv_laa_segment_lengths[3, 2]
    t = np.linspace(3*np.pi/2, 2*np.pi, pv_laa_segment_lengths[3, 2]+1, endpoint=True)  # skip last one later
    t = t[0:len(t)-1]
    x0_bound[ind1: ind2] = np.cos(t) * rhole_lspv + xhole_center[3]
    y0_bound[ind1: ind2] = np.sin(t) * rhole_lspv + yhole_center[3]

    # LAA hole, circumf
    # laa s1, starts in 3*pi/2
    ind1 = ind2
    ind2 = ind2 + pv_laa_segment_lengths[4, 0]
    t = np.linspace(3*np.pi/2, t_v8 + 2*np.pi, pv_laa_segment_lengths[4, 0]+1, endpoint=True)  # skip last one later
    t = t[0:len(t)-1]
    x0_bound[ind1: ind2] = np.cos(t) * rhole_laa + laa_hole_center_x
    y0_bound[ind1: ind2] = np.sin(t) * rhole_laa + laa_hole_center_y
    # laa s2
    ind1 = ind2
    ind2 = ind2 + pv_laa_segment_lengths[4, 1]
    t = np.linspace(t_v8, 3*np.pi/2, pv_laa_segment_lengths[4, 1]+1, endpoint=True)  # skip last one later
    t = t[0:len(t)-1]
    x0_bound[ind1: ind2] = np.cos(t) * rhole_laa + laa_hole_center_x
    y0_bound[ind1: ind2] = np.sin(t) * rhole_laa + laa_hole_center_y
    return x0_bound, y0_bound


def define_constraints_positions(s1, s2, s3, s4, s5, s6, s7, s8a, s8b, v1l_x, v1l_y, v1d_x, v1d_y, v1r_x, v1r_y, v2l_x,
                                 v2l_y, v2r_x, v2r_y, v2u_x, v2u_y, v3r_x, v3r_y, v3u_x, v3u_y, v3l_x, v3l_y,
                                 v4r_x, v4r_y, v4u_x, v4u_y, v4d_x, v4d_y, vlaad_x, vlaad_y, vlaau_x, vlaau_y, p5_x,
                                 p5_y, p6_x, p6_y, p7_x, p7_y, p8_x, p8_y):
    """Define target (x0,y0) coordinates of regional constraints given segments and template parameters (extreme coordinates of segments)"""
    p_const = s1.shape[0] + s2.shape[0] + s3.shape[0] + s4.shape[0] + s5.shape[0] + s6.shape[0] + s7.shape[0] + s8a.shape[0] + s8b.shape[0]
    x0_const = np.zeros(p_const)
    y0_const = np.zeros(p_const)
    # s1, vert line, right
    ind1 = 0
    ind2 = s1.shape[0]
    # vert line
    x0_const[ind1:ind2] = v1d_x
    aux = np.linspace(v1d_y, v2u_y, s1.shape[0] + 2, endpoint=True)
    y0_const[ind1:ind2] = aux[1:aux.size - 1]  # skip first and last

    # s2,  bottom line
    ind1 = ind2
    ind2 = ind2 + s2.shape[0]
    # crosswise lines (all with direction starting in the PV ending in the MV). General rule:
    # m = (y2-y1)/(x2-x1)
    # b = y - m*x
    # y = m*x + b   (any x and y in the line)
    aux = np.linspace(v2l_x, v3r_x, s2.size + 2, endpoint=True)
    x0_const[ind1: ind2] = aux[1:aux.size - 1]
    m = (v3r_y - v2l_y) / (v3r_x - v2l_x)
    b = v3r_y - m * v3r_x
    aux2 = m * aux + b
    y0_const[ind1: ind2] = aux2[1:aux2.size - 1]

    # s3, vert line left
    ind1 = ind2
    ind2 = ind2 + s3.shape[0]
    x0_const[ind1: ind2] = v3u_x
    aux = np.linspace(v3u_y, v4d_y, s3.shape[0] + 2, endpoint=True)
    y0_const[ind1: ind2] = aux[1:aux.size - 1]

    # s4, hori top line
    ind1 = ind2
    ind2 = ind2 + s4.shape[0]
    aux = np.linspace(v4r_x, v1l_x, s4.shape[0] + 2, endpoint=True)
    x0_const[ind1: ind2] = aux[1:aux.size - 1]
    m = (v1l_y - v4r_y) / (v1l_x - v4r_x)
    b = v4r_y - m * v4r_x
    aux2 = m * aux + b
    y0_const[ind1: ind2] = aux2[1:aux2.size - 1]

    # s5 - line crosswise line from v1r to v5
    ind1 = ind2
    ind2 = ind2 + s5.shape[0]
    m = (p5_y - v1r_y) / (p5_x - v1r_x)
    b = v1r_y - m * v1r_x
    aux = np.linspace(v1r_x, p5_x, s5.shape[0] + 2, endpoint=True)
    aux2 = m * aux + b
    x0_const[ind1: ind2] = aux[1:aux.size - 1]
    y0_const[ind1: ind2] = aux2[1:aux2.size - 1]

    # s6 - line crosswise line from v2r to v6
    ind1 = ind2
    ind2 = ind2 + s6.shape[0]
    m = (p6_y - v2r_y) / (p6_x - v2r_x)
    b = v2r_y - m * v2r_x
    aux = np.linspace(v2r_x, p6_x, s6.shape[0] + 2, endpoint=True)
    aux2 = m * aux + b
    x0_const[ind1: ind2] = aux[1:aux.size - 1]
    y0_const[ind1: ind2] = aux2[1:aux2.size - 1]

    # s7 - line crosswise line from v3l to v7
    ind1 = ind2
    ind2 = ind2 + s7.shape[0]
    m = (p7_y - v3l_y) / (p7_x - v3l_x)
    b = v3l_y - m * v3l_x
    aux = np.linspace(v3l_x, p7_x, s7.shape[0] + 2, endpoint=True)
    aux2 = m * aux + b
    x0_const[ind1: ind2] = aux[1:aux.size - 1]
    y0_const[ind1: ind2] = aux2[1:aux2.size - 1]

    # # s8a  - vertical line from lspv (v4u) to laa
    # ind1 = ind2
    # ind2 = ind2 + s8a.shape[0]    # vertical line
    # aux = np.linspace(v4u_y, vlaad_y, s8a.shape[0] + 2, endpoint=True)
    # x0_const[ind1: ind2] = xhole_center[3]
    # y0_const[ind1: ind2] = aux[1:aux.size-1]

    # s8a  - crosswise line from lspv (v4u) to laa
    ind1 = ind2
    ind2 = ind2 + s8a.shape[0]
    aux = np.linspace(v4u_x, vlaad_x, s8a.shape[0] + 2, endpoint=True)
    x0_const[ind1: ind2] = aux[1:aux.size - 1]
    m = (vlaad_y - v4u_y) / (vlaad_x - v4u_x)
    b = v4u_y - m * v4u_x
    aux2 = m * aux + b
    y0_const[ind1: ind2] = aux2[1:aux2.size - 1]

    # s8b- line crosswise line from vlaau to v8
    ind1 = ind2
    ind2 = ind2 + s8b.shape[0]
    m = (p8_y - vlaau_y) / (p8_x - vlaau_x)
    b = vlaau_y - m * vlaau_x
    if p8_x > vlaau_x:
        print('Warning: v8 is greater (in absolute value) than v_laa_up, consider select a different angle for point V8')
    aux = np.linspace(vlaau_x, p8_x, s8b.shape[0] + 2, endpoint=True)
    aux2 = m * aux + b
    x0_const[ind1: ind2] = aux[1:aux.size - 1]
    y0_const[ind1: ind2] = aux2[1:aux2.size - 1]
    return x0_const, y0_const


def ExtractVTKPoints(mesh):
    """Extract points from vtk structures. Return the Nx3 numpy.array of the vertices."""
    n = mesh.GetNumberOfPoints()
    vertex = np.zeros((n, 3))
    for i in range(n):
        mesh.GetPoint(i, vertex[i, :])
    return vertex


def ExtractVTKTriFaces(mesh):
    """Extract triangular faces from vtkPolyData. Return the Nx3 numpy.array of the faces (make sure there are only triangles)."""
    m = mesh.GetNumberOfCells()
    faces = np.zeros((m, 3), dtype=int)
    for i in range(m):
        ptIDs = vtk.vtkIdList()
        mesh.GetCellPoints(i, ptIDs)
        if ptIDs.GetNumberOfIds() != 3:
            raise Exception("Nontriangular cell!")
        faces[i, 0] = ptIDs.GetId(0)
        faces[i, 1] = ptIDs.GetId(1)
        faces[i, 2] = ptIDs.GetId(2)
    return faces


def ComputeLaplacian(vertex, faces):
    """Calculates the laplacian of a mesh
    vertex 3xN numpy.array: vertices
    faces 3xM numpy.array: faces"""
    n = vertex.shape[1]
    m = faces.shape[1]

    # compute mesh weight matrix
    W = sparse.coo_matrix((n, n))
    for i in np.arange(1, 4, 1):
        i1 = np.mod(i - 1, 3)
        i2 = np.mod(i, 3)
        i3 = np.mod(i + 1, 3)
        pp = vertex[:, faces[i2, :]] - vertex[:, faces[i1, :]]
        qq = vertex[:, faces[i3, :]] - vertex[:, faces[i1, :]]
        # normalize the vectors
        pp = pp / np.sqrt(np.sum(pp ** 2, axis=0))
        qq = qq / np.sqrt(np.sum(qq ** 2, axis=0))

        # compute angles
        ang = np.arccos(np.sum(pp * qq, axis=0))
        W = W + sparse.coo_matrix((1 / np.tan(ang), (faces[i2, :], faces[i3, :])), shape=(n, n))
        W = W + sparse.coo_matrix((1 / np.tan(ang), (faces[i3, :], faces[i2, :])), shape=(n, n))

    # compute Laplacian
    d = W.sum(axis=0)
    D = sparse.dia_matrix((d, 0), shape=(n, n))
    L = D - W
    return L


def flat(m, boundary_ids, x0, y0):
    """Conformal flattening fitting boundary to (x0,y0) coordinate positions"""
    vertex = ExtractVTKPoints(m).T
    faces = ExtractVTKTriFaces(m).T
    n = vertex.shape[1]
    L = ComputeLaplacian(vertex, faces)

    L = L.tolil()
    L[boundary_ids, :] = 0
    for i in range(boundary_ids.shape[0]):
        L[boundary_ids[i], boundary_ids[i]] = 1

    Rx = np.zeros(n)
    Rx[boundary_ids] = x0
    Ry = np.zeros(n)
    Ry[boundary_ids] = y0
    L = L.tocsr()

    result = np.zeros((Rx.size, 2))
    result[:, 0] = linalg_sp.spsolve(L, Rx)  # x
    result[:, 1] = linalg_sp.spsolve(L, Ry)  # y

    pd = vtk.vtkPolyData()
    pts = vtk.vtkPoints()

    pts.SetNumberOfPoints(n)
    for i in range(n):
        pts.SetPoint(i, result[i, 0], result[i, 1], 0)

    pd.SetPoints(pts)
    pd.SetPolys(m.GetPolys())
    pd.Modified()
    return pd


def flat_w_constraints(m, boundary_ids, constraints_ids, x0_b, y0_b, x0_c, y0_c):
    """ Conformal flattening fitting boundary points to (x0_b,y0_b) coordinate positions
    and additional contraint points to (x0_c,y0_c).
    Solve minimization problem using quadratic programming: https://en.wikipedia.org/wiki/Quadratic_programming"""

    penalization = 1000
    vertex = ExtractVTKPoints(m).T    # 3 x n_vertices
    faces = ExtractVTKTriFaces(m).T
    n = vertex.shape[1]
    L = ComputeLaplacian(vertex, faces)
    L = L.tolil()
    L[boundary_ids, :] = 0.0     # Not conformal there
    for i in range(boundary_ids.shape[0]):
         L[boundary_ids[i], boundary_ids[i]] = 1

    L = L*penalization

    Rx = np.zeros(n)
    Ry = np.zeros(n)
    Rx[boundary_ids] = x0_b * penalization
    Ry[boundary_ids] = y0_b * penalization

    L = L.tocsr()
    # result = np.zeros((Rx.size, 2))

    nconstraints = constraints_ids.shape[0]
    M = np.zeros([nconstraints, n])   # M, zero rows except 1 in constraint point
    for i in range(nconstraints):
        M[i, constraints_ids[i]] = 1
    dx = x0_c
    dy = y0_c

    block1 = hstack([L.T.dot(L), M.T])

    zeros_m = coo_matrix(np.zeros([len(dx), len(dx)]))
    block2 = hstack([M, zeros_m])

    C = vstack([block1, block2])

    prodx = coo_matrix([L.T.dot(Rx)])
    dxx = coo_matrix([dx])
    cx = hstack([prodx, dxx])

    prody = coo_matrix([L.T.dot(Ry)])
    dyy = coo_matrix([dy])
    cy = hstack([prody, dyy])

    solx = linalg_sp.spsolve(C, cx.T)
    soly = linalg_sp.spsolve(C, cy.T)

    # print('There are: ', len(np.argwhere(np.isnan(solx))), ' nans')
    # print('There are: ', len(np.argwhere(np.isnan(soly))), ' nans')
    if len(np.argwhere(np.isnan(solx))) > 0:
        print('WARNING!!! matrix is singular. It is probably due to the convergence of 2 different division lines in the same point.')
        print('Trying to assign different 2D possition to same 3D point. Try to create new division lines or increase resolution of mesh.')

    pd = vtk.vtkPolyData()
    pts = vtk.vtkPoints()

    pts.SetNumberOfPoints(n)
    for i in range(n):
        pts.SetPoint(i, solx[i], soly[i], 0)

    pd.SetPoints(pts)
    pd.SetPolys(m.GetPolys())
    pd.Modified()
    return pd

def transfer_carinas_and_lines(mesh, path_1, path_2, path_3, path_4):
    """Given mesh and dividing lines (polydatas) project the lines to the mesh by adding 2 scalar arrays:
    - "carinas" with value = 1 in the left carina and value = 2 in the right carina (0 elsewhere)
    - "lines" with value = 1 in the line connecting the 2 superior veins and value = 2 in the line connecting the
    2 inferior veins (0 elsewhere) """

    carina_array = np.zeros(mesh.GetNumberOfPoints())
    locator = vtk.vtkPointLocator()
    locator.SetDataSet(mesh)
    locator.BuildLocator()

    for p in range(path_1.GetNumberOfPoints()):
        id_p = locator.FindClosestPoint(path_1.GetPoint(p))
        carina_array[id_p] = 1

    for p in range(path_2.GetNumberOfPoints()):
        id_p = locator.FindClosestPoint(path_2.GetPoint(p))
        carina_array[id_p] = 2

    newarray = numpy_to_vtk(carina_array)
    newarray.SetName("carinas")
    mesh.GetPointData().AddArray(newarray)
    if vtk.vtkVersion().GetVTKMajorVersion() < 5:
        mesh.Update()

    lines_array = np.zeros(mesh.GetNumberOfPoints())
    locator = vtk.vtkPointLocator()
    locator.SetDataSet(mesh)
    locator.BuildLocator()

    for p in range(path_3.GetNumberOfPoints()):
        id_p = locator.FindClosestPoint(path_3.GetPoint(p))
        lines_array[id_p] = 1

    for p in range(path_4.GetNumberOfPoints()):
        id_p = locator.FindClosestPoint(path_4.GetPoint(p))
        lines_array[id_p] = 2

    newarray = numpy_to_vtk(lines_array)
    newarray.SetName("lines")
    mesh.GetPointData().AddArray(newarray)
    if vtk.vtkVersion().GetVTKMajorVersion() < 5:
        mesh.Update()
    return mesh


def cut_mesh_carina(mesh, array1, lim1, lim2, array2, value):
    """ Cut (open) mesh by adding duplicated points in the limits. It adds small gaussian noise to the points.
    Similar to cut_mesh() but in cases where there are two limits and we want to open only one of them: open mesh in the
    limits where array1 is lim1 and lim2 AND do not open if array2 == value (open only in the carina, not the other limit) """

    ids_lim1 = np.array([])
    ids_lim2 = np.array([])
    m = vtk.vtkMath()
    m.RandomSeed(0)

    # copy the original mesh point by point.
    points = vtk.vtkPoints()
    polys = vtk.vtkCellArray()
    cover = vtk.vtkPolyData()
    nver = mesh.GetNumberOfPoints()
    points.SetNumberOfPoints(nver)

    point_array1 = mesh.GetPointData().GetArray(array1)
    point_array2 = mesh.GetPointData().GetArray(array2)
    for j in range(mesh.GetNumberOfCells()):
        ptids = mesh.GetCell(j).GetPointIds()  # Get the 3 point ids
        cell = mesh.GetCell(j)

        if (ptids.GetNumberOfIds() != 3):
            print("Non triangular cell")
            break
        val0 = point_array1.GetValue(ptids.GetId(0))
        val1 = point_array1.GetValue(ptids.GetId(1))
        val2 = point_array1.GetValue(ptids.GetId(2))

        p0 = mesh.GetPoint(ptids.GetId(0))   # Coordinates
        points.SetPoint(ptids.GetId(0), p0)
        p1 = mesh.GetPoint(ptids.GetId(1))
        points.SetPoint(ptids.GetId(1), p1)
        p2 = mesh.GetPoint(ptids.GetId(2))
        points.SetPoint(ptids.GetId(2), p2)

        polys.InsertNextCell(3)

        # cases where we have to duplicate: 1. two with value = lim1 and 1 with value = lim2
        if ((val0==lim1) and (val1 == lim1) and (val2 == lim2)):
            #create new p2. Add gaussian noise to the original point. ONLY IF ITS A POINT IN THE CARINA
            if (point_array2.GetValue(ptids.GetId(0)) or point_array2.GetValue(ptids.GetId(1)) or point_array2.GetValue(ptids.GetId(2))) != value:
                new_p2 = [p2[0]+m.Gaussian(0.0, 0.0005), p2[1]+m.Gaussian(0.0, 0.0005), p2[2]+m.Gaussian(0.0, 0.0005)]
                p_id = points.InsertNextPoint(new_p2)
                polys.InsertCellPoint(cell.GetPointId(0))
                polys.InsertCellPoint(cell.GetPointId(1))
                polys.InsertCellPoint(p_id)
                # save info about the points in the limits
                ids_lim1 = np.append(ids_lim1, p_id)
                ids_lim2 = np.append(ids_lim2, ptids.GetId(2))
            else:
                polys.InsertCellPoint(cell.GetPointId(0))
                polys.InsertCellPoint(cell.GetPointId(1))
                polys.InsertCellPoint(cell.GetPointId(2))

        else:
            if ((val0==lim1) and (val1==lim2) and (val2==lim1)):
                if (point_array2.GetValue(ptids.GetId(0)) or point_array2.GetValue(ptids.GetId(1)) or point_array2.GetValue(ptids.GetId(2))) != value:
                    #create new p1. Add gaussian noise to the original point
                    new_p1 = [p1[0]+m.Gaussian(0.0, 0.0005), p1[1]+m.Gaussian(0.0, 0.0005), p1[2]+m.Gaussian(0.0, 0.0005)]
                    p_id = points.InsertNextPoint(new_p1)
                    polys.InsertCellPoint(cell.GetPointId(0))
                    polys.InsertCellPoint(p_id)
                    polys.InsertCellPoint(cell.GetPointId(2))
                    ids_lim1 = np.append(ids_lim1, p_id)
                    ids_lim2 = np.append(ids_lim2, ptids.GetId(1))
                else:
                    polys.InsertCellPoint(cell.GetPointId(0))
                    polys.InsertCellPoint(cell.GetPointId(1))
                    polys.InsertCellPoint(cell.GetPointId(2))

            else:
                if ((val0==lim2) and (val1==lim1) and (val2==lim1)):
                    if (point_array2.GetValue(ptids.GetId(0)) or point_array2.GetValue(ptids.GetId(1)) or point_array2.GetValue(ptids.GetId(2))) != value:
                        #create new p0. Add gaussian noise to the original point
                        new_p0 = [p0[0]+m.Gaussian(0.0, 0.0005), p0[1]+m.Gaussian(0.0, 0.0005), p0[2]+m.Gaussian(0.0, 0.0005)]
                        p_id = points.InsertNextPoint(new_p0)
                        polys.InsertCellPoint(p_id)
                        polys.InsertCellPoint(cell.GetPointId(1))
                        polys.InsertCellPoint(cell.GetPointId(2))
                        ids_lim1 = np.append(ids_lim1, p_id)
                        ids_lim2 = np.append(ids_lim2, ptids.GetId(0))
                    else:
                        polys.InsertCellPoint(cell.GetPointId(0))
                        polys.InsertCellPoint(cell.GetPointId(1))
                        polys.InsertCellPoint(cell.GetPointId(2))
                else:
                #2. two with value = lim2 and 1 with value = lim1
                    if ((val0==lim2) and (val1 == lim2) and (val2 == lim1)):
                        if (point_array2.GetValue(ptids.GetId(0)) or point_array2.GetValue(ptids.GetId(1)) or point_array2.GetValue(ptids.GetId(2))) != value:
                            #create new p2. Add gaussian noise to the original point
                            new_p2 = [p2[0]+m.Gaussian(0.0, 0.0005), p2[1]+m.Gaussian(0.0, 0.0005), p2[2]+m.Gaussian(0.0, 0.0005)]
                            p_id = points.InsertNextPoint(new_p2)
                            polys.InsertCellPoint(cell.GetPointId(0))
                            polys.InsertCellPoint(cell.GetPointId(1))
                            polys.InsertCellPoint(p_id)
                            ids_lim2 = np.append(ids_lim2, p_id)
                            ids_lim1 = np.append(ids_lim1, ptids.GetId(2))
                        else:
                            polys.InsertCellPoint(cell.GetPointId(0))
                            polys.InsertCellPoint(cell.GetPointId(1))
                            polys.InsertCellPoint(cell.GetPointId(2))
                    else:
                        if ((val0==lim2) and (val1 == lim1) and (val2 ==lim2)):
                            if (point_array2.GetValue(ptids.GetId(0)) or point_array2.GetValue(ptids.GetId(1)) or point_array2.GetValue(ptids.GetId(2))) != value:
                                new_p1 = [p1[0]+m.Gaussian(0.0, 0.0005), p1[1]+m.Gaussian(0.0, 0.0005), p1[2]+m.Gaussian(0.0, 0.0005)]
                                p_id = points.InsertNextPoint(new_p1)
                                polys.InsertCellPoint(cell.GetPointId(0))
                                polys.InsertCellPoint(p_id)
                                polys.InsertCellPoint(cell.GetPointId(2))
                                ids_lim2 = np.append(ids_lim2, p_id)
                                ids_lim1 = np.append(ids_lim1, ptids.GetId(1))
                            else:
                                polys.InsertCellPoint(cell.GetPointId(0))
                                polys.InsertCellPoint(cell.GetPointId(1))
                                polys.InsertCellPoint(cell.GetPointId(2))

                        else:
                            if ((val0==lim1) and (val1 == lim2) and (val2 ==lim2)):
                                if (point_array2.GetValue(ptids.GetId(0)) or point_array2.GetValue(ptids.GetId(1)) or point_array2.GetValue(ptids.GetId(2))) != value:
                                    new_p0=[p0[0]+m.Gaussian(0.0, 0.0005), p0[1]+m.Gaussian(0.0, 0.0005), p0[2]+m.Gaussian(0.0, 0.0005)]
                                    p_id = points.InsertNextPoint(new_p0)
                                    polys.InsertCellPoint(p_id)
                                    polys.InsertCellPoint(cell.GetPointId(1))
                                    polys.InsertCellPoint(cell.GetPointId(2))
                                    ids_lim2 = np.append(ids_lim2, p_id)
                                    ids_lim1 = np.append(ids_lim1, ptids.GetId(0))
                                else:
                                    polys.InsertCellPoint(cell.GetPointId(0))
                                    polys.InsertCellPoint(cell.GetPointId(1))
                                    polys.InsertCellPoint(cell.GetPointId(2))

                            else:   # Simply copy current ids
                                polys.InsertCellPoint(cell.GetPointId(0))
                                polys.InsertCellPoint(cell.GetPointId(1))
                                polys.InsertCellPoint(cell.GetPointId(2))
    cover.SetPoints(points)
    cover.SetPolys(polys)
    return cover, ids_lim1, ids_lim2


def cut_mesh_carina_and_up(mesh, array1, lim1, lim2, array2, value1, array3, value2):
    """ Cut mesh adding duplicated points in the limits. It adds small gaussian noise to the points.
        Similar to cut_mesh and cut_mesh_carina but in cases where there are 3 limits and we want to:
        1. open limit that is carina but do not save the points in the limits
        2. open limit that is one of the up lines (they are actually surrounding the posterior wall) and save the points in the limits because they are going to be blob_0 and blob_n
        3. do not open the other limit

        Find limit:
        clip if array2 == value1.  value1 1 or 2 (open in the carina, 1 -> left veins, 2 -> right veins). Do not save the points
        clip if array3 == value2.  value2 = 2
        do not clip the other limit """

    ids_lim1 = np.array([])
    ids_lim2 = np.array([])

    m = vtk.vtkMath()
    m.RandomSeed(0)

    # copy the original mesh point by point
    points = vtk.vtkPoints()
    polys = vtk.vtkCellArray()
    cover = vtk.vtkPolyData()
    nver = mesh.GetNumberOfPoints()
    points.SetNumberOfPoints(nver)

    point_array1 = mesh.GetPointData().GetArray(array1)
    point_array2 = mesh.GetPointData().GetArray(array2)
    point_array3 = mesh.GetPointData().GetArray(array3)
    for j in range(mesh.GetNumberOfCells()):
        ptids = mesh.GetCell(j).GetPointIds()
        cell = mesh.GetCell(j)
        if (ptids.GetNumberOfIds() != 3):
            print("Non triangular cell")
            break
        val0 = point_array1.GetValue(ptids.GetId(0))
        val1 = point_array1.GetValue(ptids.GetId(1))
        val2 = point_array1.GetValue(ptids.GetId(2))

        p0 = mesh.GetPoint(ptids.GetId(0))
        points.SetPoint(ptids.GetId(0), p0)
        p1 = mesh.GetPoint(ptids.GetId(1))
        points.SetPoint(ptids.GetId(1), p1)
        p2 = mesh.GetPoint(ptids.GetId(2))
        points.SetPoint(ptids.GetId(2), p2)

        polys.InsertNextCell(3)

        # cases where we have to duplicate: 1. two with value = lim1 and 1 with value = lim2
        if ((val0==lim1) and (val1 == lim1) and (val2 == lim2)):
            #create new p2. Add gaussian noise to the original point. ONLY IF ITS A POINT IN THE CARINA
            if (point_array3.GetValue(ptids.GetId(0)) or point_array3.GetValue(ptids.GetId(1)) or point_array3.GetValue(ptids.GetId(2))) == value2:
                # clip and save limits
                new_p2 = [p2[0]+m.Gaussian(0.0, 0.0005), p2[1]+m.Gaussian(0.0, 0.0005), p2[2]+m.Gaussian(0.0, 0.0005)]
                p_id = points.InsertNextPoint(new_p2)
                polys.InsertCellPoint(cell.GetPointId(0))
                polys.InsertCellPoint(cell.GetPointId(1))
                polys.InsertCellPoint(p_id)
                # save info about the points in the limits
                ids_lim1 = np.append(ids_lim1, p_id)
                ids_lim2 = np.append(ids_lim2, ptids.GetId(2))
            else:
                if (point_array2.GetValue(ptids.GetId(0)) or point_array2.GetValue(ptids.GetId(1)) or point_array2.GetValue(ptids.GetId(2))) == value1:
                    # open mesh but do not save info about points
                    new_p2 = [p2[0]+m.Gaussian(0.0, 0.0005), p2[1]+m.Gaussian(0.0, 0.0005), p2[2]+m.Gaussian(0.0, 0.0005)]
                    p_id = points.InsertNextPoint(new_p2)
                    polys.InsertCellPoint(cell.GetPointId(0))
                    polys.InsertCellPoint(cell.GetPointId(1))
                    polys.InsertCellPoint(p_id)
                else:
                    # do not open
                    polys.InsertCellPoint(cell.GetPointId(0))
                    polys.InsertCellPoint(cell.GetPointId(1))
                    polys.InsertCellPoint(cell.GetPointId(2))

        else:
            if ((val0==lim1) and (val1==lim2) and (val2==lim1)):
                if (point_array3.GetValue(ptids.GetId(0)) or point_array3.GetValue(ptids.GetId(1)) or point_array3.GetValue(ptids.GetId(2))) == value2:
                    #create new p1. Add gaussian noise to the original point
                    new_p1 = [p1[0]+m.Gaussian(0.0, 0.0005), p1[1]+m.Gaussian(0.0, 0.0005), p1[2]+m.Gaussian(0.0, 0.0005)]
                    p_id = points.InsertNextPoint(new_p1)
                    polys.InsertCellPoint(cell.GetPointId(0))
                    polys.InsertCellPoint(p_id)
                    polys.InsertCellPoint(cell.GetPointId(2))
                    ids_lim1 = np.append(ids_lim1, p_id)
                    ids_lim2 = np.append(ids_lim2, ptids.GetId(1))
                else:
                    if (point_array2.GetValue(ptids.GetId(0)) or point_array2.GetValue(ptids.GetId(1)) or point_array2.GetValue(ptids.GetId(2))) == value1:
                        new_p1 = [p1[0]+m.Gaussian(0.0, 0.0005), p1[1]+m.Gaussian(0.0, 0.0005), p1[2]+m.Gaussian(0.0, 0.0005)]
                        p_id = points.InsertNextPoint(new_p1)
                        polys.InsertCellPoint(cell.GetPointId(0))
                        polys.InsertCellPoint(p_id)
                        polys.InsertCellPoint(cell.GetPointId(2))
                    else:
                        polys.InsertCellPoint(cell.GetPointId(0))
                        polys.InsertCellPoint(cell.GetPointId(1))
                        polys.InsertCellPoint(cell.GetPointId(2))

            else:
                if ((val0==lim2) and (val1==lim1) and (val2==lim1)):
                    if (point_array3.GetValue(ptids.GetId(0)) or point_array3.GetValue(ptids.GetId(1)) or point_array3.GetValue(ptids.GetId(2))) == value2:
                        #create new p0. Add gaussian noise to the original point
                        new_p0 = [p0[0]+m.Gaussian(0.0, 0.0005), p0[1]+m.Gaussian(0.0, 0.0005), p0[2]+m.Gaussian(0.0, 0.0005)]
                        p_id = points.InsertNextPoint(new_p0)
                        polys.InsertCellPoint(p_id)
                        polys.InsertCellPoint(cell.GetPointId(1))
                        polys.InsertCellPoint(cell.GetPointId(2))
                        ids_lim1 = np.append(ids_lim1, p_id)
                        ids_lim2 = np.append(ids_lim2, ptids.GetId(0))
                    else:
                        if (point_array2.GetValue(ptids.GetId(0)) or point_array2.GetValue(ptids.GetId(1)) or point_array2.GetValue(ptids.GetId(2))) == value1:
                            new_p0 = [p0[0]+m.Gaussian(0.0, 0.0005), p0[1]+m.Gaussian(0.0, 0.0005), p0[2]+m.Gaussian(0.0, 0.0005)]
                            p_id = points.InsertNextPoint(new_p0)
                            polys.InsertCellPoint(p_id)
                            polys.InsertCellPoint(cell.GetPointId(1))
                            polys.InsertCellPoint(cell.GetPointId(2))
                        else:
                            polys.InsertCellPoint(cell.GetPointId(0))
                            polys.InsertCellPoint(cell.GetPointId(1))
                            polys.InsertCellPoint(cell.GetPointId(2))
                else:
                #2. two with value = lim2 and 1 with value = lim1
                    if ((val0==lim2) and (val1 == lim2) and (val2 == lim1)):
                        if (point_array3.GetValue(ptids.GetId(0)) or point_array3.GetValue(ptids.GetId(1)) or point_array3.GetValue(ptids.GetId(2))) == value2:
                            #create new p2. Add gaussian noise to the original point
                            new_p2 = [p2[0]+m.Gaussian(0.0, 0.0005), p2[1]+m.Gaussian(0.0, 0.0005), p2[2]+m.Gaussian(0.0, 0.0005)]
                            p_id = points.InsertNextPoint(new_p2)
                            polys.InsertCellPoint(cell.GetPointId(0))
                            polys.InsertCellPoint(cell.GetPointId(1))
                            polys.InsertCellPoint(p_id)
                            ids_lim2 = np.append(ids_lim2, p_id)
                            ids_lim1 = np.append(ids_lim1, ptids.GetId(2))
                        else:
                            if (point_array2.GetValue(ptids.GetId(0)) or point_array2.GetValue(ptids.GetId(1)) or point_array2.GetValue(ptids.GetId(2))) == value1:
                                new_p2 = [p2[0]+m.Gaussian(0.0, 0.0005), p2[1]+m.Gaussian(0.0, 0.0005), p2[2]+m.Gaussian(0.0, 0.0005)]
                                p_id = points.InsertNextPoint(new_p2)
                                polys.InsertCellPoint(cell.GetPointId(0))
                                polys.InsertCellPoint(cell.GetPointId(1))
                                polys.InsertCellPoint(p_id)
                            else:
                                polys.InsertCellPoint(cell.GetPointId(0))
                                polys.InsertCellPoint(cell.GetPointId(1))
                                polys.InsertCellPoint(cell.GetPointId(2))
                    else:
                        if ((val0==lim2) and (val1 == lim1) and (val2 ==lim2)):
                            if (point_array3.GetValue(ptids.GetId(0)) or point_array3.GetValue(ptids.GetId(1)) or point_array3.GetValue(ptids.GetId(2))) == value2:
                                new_p1 = [p1[0]+m.Gaussian(0.0, 0.0005), p1[1]+m.Gaussian(0.0, 0.0005), p1[2]+m.Gaussian(0.0, 0.0005)]
                                p_id = points.InsertNextPoint(new_p1)
                                polys.InsertCellPoint(cell.GetPointId(0))
                                polys.InsertCellPoint(p_id)
                                polys.InsertCellPoint(cell.GetPointId(2))
                                ids_lim2 = np.append(ids_lim2, p_id)
                                ids_lim1 = np.append(ids_lim1, ptids.GetId(1))
                            else:
                                if (point_array2.GetValue(ptids.GetId(0)) or point_array2.GetValue(ptids.GetId(1)) or point_array2.GetValue(ptids.GetId(2))) == value1:
                                    new_p1 = [p1[0]+m.Gaussian(0.0, 0.0005), p1[1]+m.Gaussian(0.0, 0.0005), p1[2]+m.Gaussian(0.0, 0.0005)]
                                    p_id = points.InsertNextPoint(new_p1)
                                    polys.InsertCellPoint(cell.GetPointId(0))
                                    polys.InsertCellPoint(p_id)
                                    polys.InsertCellPoint(cell.GetPointId(2))
                                else:
                                    polys.InsertCellPoint(cell.GetPointId(0))
                                    polys.InsertCellPoint(cell.GetPointId(1))
                                    polys.InsertCellPoint(cell.GetPointId(2))

                        else:
                            if ((val0==lim1) and (val1 == lim2) and (val2 ==lim2)):
                                if (point_array3.GetValue(ptids.GetId(0)) or point_array3.GetValue(ptids.GetId(1)) or point_array3.GetValue(ptids.GetId(2))) == value2:
                                    new_p0=[p0[0]+m.Gaussian(0.0, 0.0005), p0[1]+m.Gaussian(0.0, 0.0005), p0[2]+m.Gaussian(0.0, 0.0005)]
                                    p_id = points.InsertNextPoint(new_p0)
                                    polys.InsertCellPoint(p_id)
                                    polys.InsertCellPoint(cell.GetPointId(1))
                                    polys.InsertCellPoint(cell.GetPointId(2))
                                    ids_lim2 = np.append(ids_lim2, p_id)
                                    ids_lim1 = np.append(ids_lim1, ptids.GetId(0))
                                else:
                                    if (point_array2.GetValue(ptids.GetId(0)) or point_array2.GetValue(ptids.GetId(1)) or point_array2.GetValue(ptids.GetId(2))) == value1:
                                        new_p0=[p0[0]+m.Gaussian(0.0, 0.0005), p0[1]+m.Gaussian(0.0, 0.0005), p0[2]+m.Gaussian(0.0, 0.0005)]
                                        p_id = points.InsertNextPoint(new_p0)
                                        polys.InsertCellPoint(p_id)
                                        polys.InsertCellPoint(cell.GetPointId(1))
                                        polys.InsertCellPoint(cell.GetPointId(2))
                                    else:
                                        polys.InsertCellPoint(cell.GetPointId(0))
                                        polys.InsertCellPoint(cell.GetPointId(1))
                                        polys.InsertCellPoint(cell.GetPointId(2))

                            else:   # Simply copy current ids
                                polys.InsertCellPoint(cell.GetPointId(0))
                                polys.InsertCellPoint(cell.GetPointId(1))
                                polys.InsertCellPoint(cell.GetPointId(2))
    cover.SetPoints(points)
    cover.SetPolys(polys)
    return cover, ids_lim1, ids_lim2


def find_blobs(mesh):
    """ Find all scar patches (blobs) (having cut/opened the vein previously, if there is a scar patch in the limit it must be duplicated)
       1. Transfer info about scar from points to cells
       2. Apply threshold (keep only cells that are scar)
       3. Connectivity filter"""

    transfer_info_points2cells_threshold(mesh, "scar", "scar_cell", 1.5)
    thresh_vein = cellthreshold(mesh, 'scar_cell', 1, 1)

    connect = vtk.vtkPolyDataConnectivityFilter()
    connect.SetInputData(thresh_vein)
    connect.ScalarConnectivityOn()
    connect.SetExtractionModeToAllRegions()
    connect.ColorRegionsOn()
    connect.Update()
    # ncontours = connect.GetNumberOfExtractedRegions()  # number of scar patches
    cc = connect.GetOutput()    # mesh with array 'RegionId' (points) with the label of each region. In the CUT I have the same label

    #transfer array with the labels of the BLOBS to the COMPLETE mesh (vein not thresholded by scar)
    transfer_blob_array(cc, mesh, 'RegionId', 'blob')
    return mesh

def transfer_blob_array(ref, target, arrayname, targetarrayname):

    # initiate point locator
    locator = vtk.vtkPointLocator()
    locator.SetDataSet(target)
    locator.BuildLocator()

    # get array from reference
    refarray = ref.GetPointData().GetArray(arrayname)
    numberofpoints = target.GetNumberOfPoints()
    newarray = vtk.vtkDoubleArray()
    newarray.SetName(targetarrayname)
    newarray.SetNumberOfTuples(numberofpoints)
    #initialize to -1
    for i in range(numberofpoints):
        point = target.GetPoint(i)
        newarray.SetValue(i, -1)

    # go through each point of ref surface
    for i in range(ref.GetNumberOfPoints()):
        point = ref.GetPoint(i)
        closestpoint_id = locator.FindClosestPoint(point)
        value = refarray.GetValue(i)
        newarray.SetValue(closestpoint_id, value)
    target.GetPointData().AddArray(newarray)

    return target


def find_dual_points_limits(mesh, limit1, limit2):
    """ Find the "dual point" in the mesh. The closest point but in the other part of the mesh (after opening)
        Use the information saved during the cutting/opening part and compute simple euclidean distance"""

    longest = np.argmax([len(limit1), len(limit2)])   # 0 or 1
    if longest == 0:
        ids_1 = limit1
        ids_2 = limit2
    else:
        ids_1 = limit2
        ids_2 = limit1
    dual_points = np.zeros([2, len(ids_1)])
    for i in range(len(ids_1)):   # go over the longest array
        p1 = mesh.GetPoint(ids_1[i])
        d = np.zeros(len(ids_2))
        for j in range(len(ids_2)):
            p2 = mesh.GetPoint(ids_2[j])
            d[j] = euclideandistance(p1, p2)
        min_dis_id = np.argmin(d)
        dual_points[0, i] = ids_1[i]
        dual_points[1, i] = ids_2[min_dis_id]
    return dual_points


def find_closed_shortest_path(mesh, distance_transforms, dual_points, blob_ids, new_blob_array):
    """ For each point in the limit where I opened the mesh create the graph assuming that the path starts/end there.
        All points in the limit are potential candidates.
        The final shortest path will we the shortest of those shortest paths"""

    min_total_gap_length = 1000000   # initialize to high value
    all_min_gap_lengths = np.zeros(dual_points.shape[1])  # save here all the min gap length for each of the crossing points
    nblobs = len(np.unique(blob_ids))-1  # -1 because that is the value when there is no scar (and not numerated blob therefore)

    for p in range(dual_points.shape[1]):   # number of points in the limits
        p1 = dual_points[0, p]              # crossing points, one from artificial blob_0 and the other one from artificial blob_n
        p2 = dual_points[1, p]

        min_distances = np.zeros((nblobs+2, nblobs+2))
        all_closest_points = np.zeros((nblobs+2, nblobs+2))

        # Compute distances from p1 (from blob0) to the rest of the blobs. Using the already calculated distance transforms of all the blobs (also artificial)
        # first row and first column of the graph
        for blobi in range(nblobs+2):
            if blobi == 0:
                for blobj in range(nblobs):  # 0 .. nblobs-1
                    dist_to_blob_j = distance_transforms[:, blobj+1]   # Not the 0, until nblobs + 1 (neither artificial n, because the point it should go trhough is fixed now)
                    min_distances[0, blobj+1] = dist_to_blob_j[p1]
                    min_distances[blobj+1, 0] = dist_to_blob_j[p1]
            elif blobi == nblobs+1:  # last iteration
                for blobj in range(nblobs):  # 0 .. nblobs-1
                    dist_to_blob_j = distance_transforms[:, blobj+1]
                    min_distances[nblobs+1, blobj+1] = dist_to_blob_j[p2]  # last row and last column
                    min_distances[blobj+1, nblobs+1] = dist_to_blob_j[p2]

            else:   # intermediate blobs
                dist_to_blob_i = distance_transforms[:, blobi]
                closest_points_to_blob_i = np.zeros(nblobs+2)
                for blobj in range(nblobs+2):   # skip case blobj = 0 (is one of the artificial blobs)
                    if blobj > 0:
                        if (blobi!=blobj):
                            #blob_j = np.where(new_blob_array == blobj)[0]
                            blob_j = np.where(blob_ids == blobj-1)[0]  # Better with this array that shoudnt be empty for real blobs
                            if blob_j.size != 0:   # This can be zero if a very small blob disappears when I create the artificial blobs. (I change its label to artificial blob0 for example) np.min will fail if the array is empty
                                                   # For the artificial blobs it will still be 0, blobj goes until nblobs + 2
                                min_distances[blobi, blobj] = np.min(dist_to_blob_i[blob_j])
                                pos_min = np.argmin(dist_to_blob_i[blob_j])
                                closest_points_to_blob_i[blobj] = blob_j[pos_min]

                all_closest_points[:, blobi] = closest_points_to_blob_i
        # first row: closest points of b0 to the other blobs, this is always p1
        closest_points_to_blob_i = p1*np.ones(nblobs+2)
        all_closest_points[0, :] = closest_points_to_blob_i
        # last row: closest points of bn to the other blobs, this is always pn
        closest_points_to_blob_i = p2*np.ones(nblobs+2)
        all_closest_points[nblobs+1, :] = closest_points_to_blob_i

        # distance between the crossing points (full loop)
        [distance_p1_p2, p1_p2_path] = compute_geodesic_distance(mesh, p1, p2)
        min_distances[0, nblobs+1] = distance_p1_p2
        min_distances[nblobs+1, 0] = distance_p1_p2

        # Check if the point that is now artificial blob0 or blobn was part of a scar patch before.
        # I have to put distance = 0 in that case (when a point was originally part of blob_i and later I put blob = 0 because is part of an "artificial blob")

        ori_blob_id_p1 = blob_ids[p1]
        if ori_blob_id_p1 > -1:  # it was part of a scar patch, set distance between the point and the scar patch to 0
            min_distances[0, ori_blob_id_p1+1] = 0
            min_distances[ori_blob_id_p1+1, 0] = 0

        ori_blob_id_p2 = blob_ids[p2]
        if ori_blob_id_p2 > -1:
            min_distances[nblobs+1, ori_blob_id_p2+1] = 0
            min_distances[ori_blob_id_p2+1, nblobs+1] = 0

        #############  Dijkstra algorithm  #############
        # 1. Create the graph
        g = Graph()

        vertex_names = np.empty(nblobs+1, dtype=object)
        vertex_names[0] = 'limit1'
        for i in range(nblobs):
            vertex_names[i+1] = 'blob_' + str(i)

        g.add_vertex(vertex_names[0])
        for blobi in range(nblobs):
            g.add_vertex(vertex_names[blobi+1])
        g.add_vertex('limit2')

        for blobi in range(np.size(min_distances, 1)-1):   # The last one independently because the name will be different
            for blobj in range(np.size(min_distances, 1)-1):
                g.add_edge(vertex_names[blobi], vertex_names[blobj], min_distances[blobi, blobj])
            g.add_edge(vertex_names[blobi], 'limit2', min_distances[blobi, np.size(min_distances, 1)-1])

        for j in range(np.size(min_distances, 1)-1):
            g.add_edge('limit2', vertex_names[blobj], min_distances[np.size(min_distances, 1)-1, blobj])
        g.add_edge('limit2', 'limit2', min_distances[np.size(min_distances, 1)-1, np.size(min_distances, 1)-1])

        # Get shortest path and minimum distance from limit1 to limit2
        dijkstra(g, g.get_vertex('limit1'), g.get_vertex('limit2'))
        target = g.get_vertex('limit2')
        path = [target.get_id()]
        shortest(target, path)
        total_gap_length = target.get_distance()

        all_min_gap_lengths[p] = total_gap_length    # save here, for each crossing point the gap length. Check later if there are several crossing points that give the minimum possible gap length

        if total_gap_length < min_total_gap_length:  # update current shortest path
            min_total_gap_length = total_gap_length
            min_path = path
            min_min_distances = min_distances
            best_all_closest_points = all_closest_points

            cross_point_0 = p1
            cross_point_n = p2

    if min_total_gap_length == 1000000:
        print("Not closed surface around the vein")
        min_path = "not_closed"
        cross_point_0 = dual_points[0, 0]    # just to assign something
        cross_point_n = dual_points[1, 0]

    return min_path, min_total_gap_length, cross_point_0, cross_point_n, min_min_distances, best_all_closest_points, all_min_gap_lengths


def find_closest_point_limit(mesh, new_blob_array, p1_id, blob_id):
    """ Find the closest point belonging to blob with id = blob_id to p1_id (in general this is a point of the limit).
        Return closest point id and corresponding geodesic distance."""

    blob = np.where(new_blob_array == blob_id)[0]    # point ids belonging to blob_id
    min_dist = 10000
    for ind in range(len(blob)):
        p2_id = blob[ind]
        [dist_p1_p2, p1_p2_path] = compute_geodesic_distance(mesh, p1_id, p2_id)
        dist = dist_p1_p2
        if dist < min_dist:
            min_dist = dist
            min_path = p1_p2_path
            min_ind = ind
    closest_p = blob[min_ind]
    return closest_p, min_path


def create_polydatas_path(mesh, path_indices, min_distances, closest_points_matrix, cross_point_0, cross_point_n, blob_ids, new_blob_array, directory, pv_label,  n_mins_pos, dual_points):
    """ Go through the path an create polydatas (write them directly) corresponding to all scar_paths or gap_paths
    Return also total_scar_length within the path to compute RGM later. """

    total_scar_path = 0
    min_dist = 100000
    n_mins = np.shape(n_mins_pos)[1]  # array with the positions of dual_points of the points that give min GAP length
    ntotal_nodes_graph = min_distances.shape[0]
    nblobs_in_final_path = path_indices.shape[0]
    last_scar_done = False   # flag to check if I've already created last scar polydata in the case where the first and last ones must be calculated at the same time
    last_gap_done = False
    first_scar_done = False
    first_gap_done = False

    if min_distances[0, path_indices[1]] == 0:    # that means that the first polydata must be 'scar' type
        # if there is scar in the limit and the path has length = 2 means that the vein is isolated. In any other case, the length is higher
        if len(path_indices) == 2:
            [blob_i_width, within_blob_path] = compute_geodesic_distance(mesh, cross_point_0, cross_point_n)
            name_poly_path = os.path.join(directory, pv_label + '_dist_within_blob_0.vtk')
            writevtk(within_blob_path, name_poly_path)
            total_scar_path = total_scar_path + blob_i_width
            first_scar_done = True
        else:
            if min_distances[ntotal_nodes_graph-1, path_indices[nblobs_in_final_path-2]] == 0:  # that means that the last polydata must be 'scar' type
                # Create two polydatas from the crossing points to the closest point from the cut blob (they are actually the first and last blob in the path) to the next blobs
                p0 = cross_point_0
                p1 = closest_points_matrix[path_indices[1], path_indices[2]]    # point of the first real blob in the path, which is the closest to the next blob
                pn = cross_point_n

                last_blob = np.where(new_blob_array == path_indices[len(path_indices)-2])[0]
                if len(last_blob) > 0:    # this can be zero if a small patch in the limit disappears ('else' case)
                    p2 = closest_points_matrix[path_indices[len(path_indices)-2], path_indices[len(path_indices)-3]]
                else:
                    [p2, poly_p2] = find_closest_point_limit(mesh, blob_ids, pn, path_indices[len(path_indices)-2]-1)  # use original blob_ids array and original indexing of blobs
                    p2 = closest_points_matrix[path_indices[len(path_indices)-2], path_indices[len(path_indices)-3]]

                if n_mins > 1:   # there are more than 1 min, I'll maybe have to change the crossing points (both obviously, I need a closed path)
                    n_mins_pos = n_mins_pos[0]
                    for index in range(n_mins):
                        possible_p0 = dual_points[0, n_mins_pos[index]]
                        possible_pn = dual_points[1, n_mins_pos[index]]
                        [dist0, dist_path0] = compute_geodesic_distance(mesh, possible_p0, p1)
                        [distn, dist_pathn] = compute_geodesic_distance(mesh, possible_pn, p2)
                        if (dist0 + distn) < min_dist:
                            min_dist = dist0 + distn
                            p0 = possible_p0
                            pn = possible_pn

                [blob_i_width, within_blob_path] = compute_geodesic_distance(mesh, p0, p1)
                name_poly_path = os.path.join(directory, pv_label + '_dist_within_blob_0.vtk')
                writevtk(within_blob_path, name_poly_path)
                total_scar_path = total_scar_path + blob_i_width
                first_scar_done = True

                [blob_i_width, within_blob_path] = compute_geodesic_distance(mesh, p2, pn)
                name_poly_path = os.path.join(directory, pv_label + '_dist_within_blob_' + str(len(path_indices)-3) + '.vtk')
                writevtk(within_blob_path, name_poly_path)
                total_scar_path = total_scar_path + blob_i_width
                last_scar_done = True
            else:   # I only start with scar (I do not finish). Create only first scar path.
                p0 = cross_point_0
                p1 = closest_points_matrix[path_indices[1], path_indices[2]]
                [blob_i_width, within_blob_path] = compute_geodesic_distance(mesh, p0, p1)
                name_poly_path = os.path.join(directory, pv_label + '_dist_within_blob_0.vtk')
                writevtk(within_blob_path, name_poly_path)
                total_scar_path = total_scar_path + blob_i_width
                first_scar_done = True

    else:   # the first polydata must be 'gap' type
        if len(path_indices) == 2:     # path = 'limit1', 'limit2' -> Everything is gap
            name_poly_path = os.path.join(directory, pv_label + '_gap_0_path.vtk')
            # [distance_p1_p2, p1_p2_path] = compute_geodesic_distance(mesh, cross_point_0, cross_point_n)
            [distance_p1_p2, p1_p2_path] = compute_geodesic_distance(mesh, cross_point_n, cross_point_0)
            writevtk(p1_p2_path, name_poly_path)
            first_gap_done = True
            last_gap_done = True
        else:
            if min_distances[ntotal_nodes_graph-1, path_indices[nblobs_in_final_path-2]] != 0:   # path finishes also with gap
                [first_p, first_poly_gap] = find_closest_point_limit(mesh, new_blob_array, cross_point_0, path_indices[1])  # find the closest point belonging to the first real blob to the crossing point 0
                name_poly_path = os.path.join(directory, pv_label + '_gap_0_path.vtk')
                writevtk(first_poly_gap, name_poly_path)
                first_gap_done = True

                [last_p, last_poly_gap] = find_closest_point_limit(mesh, new_blob_array, cross_point_n, path_indices[len(path_indices)-2])
                name_poly_path = os.path.join(directory, pv_label + '_gap_' + str(len(path_indices)-2) + '_path.vtk')
                writevtk(last_poly_gap, name_poly_path)
                last_gap_done = True

            else:  # create just the first gap polydata
                [first_p, first_poly_gap] = find_closest_point_limit(mesh, new_blob_array, cross_point_0, path_indices[1])  # find the closest point belonging to the first real blob to the crossing point 0
                name_poly_path = os.path.join(directory, pv_label + '_gap_0_path.vtk')
                writevtk(first_poly_gap, name_poly_path)
                first_gap_done = True

    if (min_distances[ntotal_nodes_graph-1, path_indices[nblobs_in_final_path-2]] == 0):  # that means that the last polydata must be 'scar' type
        if last_scar_done:
            print("last scar already done")
        else:   # do the last scar path
            pn = cross_point_n
            last_blob_id = path_indices[len(path_indices)-2]
            prev_blob_id = path_indices[len(path_indices)-3]
            last_blob = np.where(new_blob_array == last_blob_id)[0]
            if len(last_blob) > 0:    # this can be zero if a small scar patch in the limit disappears
                p2 = closest_points_matrix[last_blob_id, prev_blob_id]
            else:
                [p2, poly_p2] = find_closest_point_limit(mesh, blob_ids, pn, path_indices[len(path_indices)-2]-1)  # use original blob_ids array and original indexing of blobs

            [blob_i_width, within_blob_path] = compute_geodesic_distance(mesh, p2, pn)
            name_poly_path = os.path.join(directory, pv_label + '_dist_within_blob_' + str(len(path_indices)-3) + '.vtk')
            writevtk(within_blob_path, name_poly_path)
            total_scar_path = total_scar_path + blob_i_width
            last_scar_done = True

    else:   # the last polydata must be 'gap' type
        if not last_gap_done:
            [last_p, last_poly_gap] = find_closest_point_limit(mesh, new_blob_array, cross_point_n, path_indices[len(path_indices)-2])
            if first_gap_done:
                name_poly_path = os.path.join(directory, pv_label + '_gap_' + str(len(path_indices)-2) + '_path.vtk')
            else:
                name_poly_path = os.path.join(directory, pv_label + '_gap_' + str(len(path_indices)-3) + '_path.vtk')
            writevtk(last_poly_gap, name_poly_path)
            last_gap_done = True

    # For the rest of the path:
    # 1. remaining GAPS
    for gap in range(len(path_indices)-3):

        p1 = closest_points_matrix[path_indices[gap+1], path_indices[gap+2]]
        p2 = closest_points_matrix[path_indices[gap+2], path_indices[gap+1]]

        [distance_p1_p2, p1_p2_path] = compute_geodesic_distance(mesh, int(p1), int(p2))
        if not first_gap_done:  # path started with scar, the first gap is this one, in the middle of the path
            name_poly_path = os.path.join(directory, pv_label + '_gap_' + str(gap) + '_path.vtk')
        else:
            name_poly_path = os.path.join(directory, pv_label + '_gap_' + str(gap+1) + '_path.vtk')
        writevtk(p1_p2_path, name_poly_path)

    # 2. Within blobs distances and polydatas
    if not first_scar_done:
        for blob_i in range(len(path_indices)-2):
            if (blob_i == len(path_indices)-3 and last_scar_done):  # last
                 print('Last scar already done')
            else:
                blob_n = path_indices[blob_i+1]
                blob_prev_n = path_indices[blob_i]
                blob_next_n = path_indices[blob_i+2]
                if (blob_i == 0):  #first
                    [p1, poly_p1] = find_closest_point_limit(mesh, new_blob_array, cross_point_0, blob_n)
                else:
                    p1 = closest_points_matrix[blob_n, blob_prev_n]   # point belonging to blob_n closest to blob n - 1 (previous in the shortest path)

                p2 = closest_points_matrix[blob_n, blob_next_n]   # point belonging to blob_n closest to blob n + 1 (next one in the shortest path)

                # correct for the last case
                if not last_scar_done and blob_i == len(path_indices) - 3:
                    [p2, poly_p2] = find_closest_point_limit(mesh, new_blob_array, cross_point_n, blob_n)

                # compute width of the blob (according to our definition of isolating path), create the polydata AND update the total_scar_path variable
                [blob_i_width, within_blob_path] = compute_geodesic_distance(mesh, int(p1), int(p2))
                name_poly_path = os.path.join(directory, pv_label + '_dist_within_blob_' + str(blob_i) + '.vtk')
                writevtk(within_blob_path, name_poly_path)
                total_scar_path = total_scar_path + blob_i_width
    else:
        if last_scar_done:  # First scar and last scar already done (in the limit were scar in the both directions)
            for blob_i in range(len(path_indices)-4):       # number of blobs I go through
                blob_n = path_indices[blob_i+2]   # not artificial, not the first real one, the next one
                blob_prev_n = path_indices[blob_i+1]
                blob_next_n = path_indices[blob_i+3]
                p1 = closest_points_matrix[blob_n, blob_prev_n]   # point belonging to blob_n closest to blob n - 1 (previous in the shortest path)
                p2 = closest_points_matrix[blob_n, blob_next_n]   # point belonging to blob_n closest to blob n + 1 (next one in the shortest path)
                [blob_i_width, within_blob_path] = compute_geodesic_distance(mesh, p1, p2)
                name_poly_path = os.path.join(directory, pv_label + '_dist_within_blob_' + str(blob_i+1) + '.vtk')

                writevtk(within_blob_path, name_poly_path)
                total_scar_path = total_scar_path + blob_i_width
        else:   # first scar done (start numbering with 1 from now) but not last one (the path finishes with a gap)
            for blob_i in range(len(path_indices)-3):       # number of blobs I go through
                blob_n = path_indices[blob_i+2]   # this is now the last one before the last gap (in this situation the path finishes with gap)
                blob_prev_n = path_indices[blob_i+1]
                blob_next_n = path_indices[blob_i+3]
                p1 = closest_points_matrix[blob_n, blob_prev_n]
                if blob_i == len(path_indices)-4:   #last iteration
                    [p2, poly_p2] = find_closest_point_limit(mesh, new_blob_array, cross_point_n, blob_n)
                else:
                    p2 = closest_points_matrix[blob_n, blob_next_n]   # point belonging to blob_n closest to blob n + 1 (next one in the shortest path)
                [blob_i_width, within_blob_path] = compute_geodesic_distance(mesh, p1, p2)

                name_poly_path = os.path.join(directory, pv_label + '_dist_within_blob_' + str(blob_i+1) + '.vtk')
                writevtk(within_blob_path, name_poly_path)
                total_scar_path = total_scar_path + blob_i_width

    return total_scar_path


def find_min_euclidean_dist_between_extreme_gaps(m1, m2):
    distances = np.zeros([1, 4])
    p1 = m1.GetPoint(0)
    p2 = m2.GetPoint(0)
    distances[0, 0] = math.sqrt(math.pow(p1[0] - p2[0], 2) + math.pow(p1[1] - p2[1], 2) + math.pow(p1[2] - p2[2], 2))

    p1 = m1.GetPoint(0)
    p2 = m2.GetPoint(m2.GetNumberOfPoints() - 1)
    distances[0, 1] = math.sqrt(math.pow(p1[0] - p2[0], 2) + math.pow(p1[1] - p2[1], 2) + math.pow(p1[2] - p2[2], 2))

    p1 = m1.GetPoint(m1.GetNumberOfPoints() - 1)
    p2 = m2.GetPoint(0)
    distances[0, 2] = math.sqrt(math.pow(p1[0] - p2[0], 2) + math.pow(p1[1] - p2[1], 2) + math.pow(p1[2] - p2[2], 2))

    p1 = m1.GetPoint(m1.GetNumberOfPoints() - 1)
    p2 = m2.GetPoint(m2.GetNumberOfPoints() - 1)
    distances[0, 3] = math.sqrt(math.pow(p1[0] - p2[0], 2) + math.pow(p1[1] - p2[1], 2) + math.pow(p1[2] - p2[2], 2))

    min_dist = np.min(distances)
    return min_dist


def merge_polydatas_gaps_blobs_4veins(path):
    """ Merge all (corresponding to same PV) polydatas gaps and blobs found in path with appropriate name"""

    # Gaps
    name = "rspv_gap_*.vtk"
    rspv_list = glob.glob(os.path.join(path, name))
    name = "ripv_gap_*.vtk"
    ripv_list = glob.glob(os.path.join(path, name))
    name = "lipv_gap_*.vtk"
    lipv_list = glob.glob(os.path.join(path, name))
    name = "lspv_gap_*.vtk"
    lspv_list = glob.glob(os.path.join(path, name))

    # print 'Number of gaps concatenated (independent PVs):', len(rspv_list), len(ripv_list), len(lspv_list), len(lipv_list)
    if len(rspv_list) > 0:
        m_out = append_polys(rspv_list)
        writevtk(m_out, os.path.join(path, 'rspv_all_gaps.vtk'))
    if len(ripv_list) > 0:
        m_out = append_polys(ripv_list)
        writevtk(m_out, os.path.join(path, 'ripv_all_gaps.vtk'))
    if len(lipv_list) > 0:
        m_out = append_polys(lipv_list)
        writevtk(m_out, os.path.join(path, 'lipv_all_gaps.vtk'))
    if len(lspv_list) > 0:
        m_out = append_polys(lspv_list)
        writevtk(m_out, os.path.join(path, 'lspv_all_gaps.vtk'))

    # Scar blobs
    name = "rspv_dist_within_blob_*.vtk"
    rspv_list = glob.glob(os.path.join(path, name))
    name = "ripv_dist_within_blob_*.vtk"
    ripv_list = glob.glob(os.path.join(path, name))
    name = "lipv_dist_within_blob_*.vtk"
    lipv_list = glob.glob(os.path.join(path, name))
    name = "lspv_dist_within_blob_*.vtk"
    lspv_list = glob.glob(os.path.join(path, name))

    # print 'Number of blobs concatenated (independent PVs):', len(rspv_list), len(ripv_list), len(lspv_list), len(lipv_list)

    if len(rspv_list) > 0:
        m_out = append_polys(rspv_list)
        writevtk(m_out, os.path.join(path, 'rspv_all_blobs.vtk'))
    if len(ripv_list) > 0:
        m_out = append_polys(ripv_list)
        writevtk(m_out, os.path.join(path, 'ripv_all_blobs.vtk'))
    if len(lipv_list) > 0:
        m_out = append_polys(lipv_list)
        writevtk(m_out, os.path.join(path, 'lipv_all_blobs.vtk'))
    if len(lspv_list) > 0:
        m_out = append_polys(lspv_list)
        writevtk(m_out, os.path.join(path, 'lspv_all_blobs.vtk'))


def merge_polydatas_gaps_blobs_lateral_veins(path):
    """ Merge all (corresponding to same lateral searching area (right/left)) polydatas gaps and blobs found in path"""

    # Gaps
    name = "left_gap_*.vtk"
    left_list = glob.glob(os.path.join(path, name))
    name = "right_gap_*.vtk"
    right_list = glob.glob(os.path.join(path, name))

    if len(left_list) > 0:
        m_out = append_polys(left_list)
        writevtk(m_out, os.path.join(path, 'joint_left_all_gaps.vtk'))
    if len(right_list) > 0:
        m_out = append_polys(right_list)
        writevtk(m_out, os.path.join(path, 'joint_right_all_gaps.vtk'))

    # Scar blobs
    name = "left_dist_within_blob_*.vtk"
    left_list = glob.glob(os.path.join(path, name))
    name = "right_dist_within_blob_*.vtk"
    right_list = glob.glob(os.path.join(path, name))

    if len(left_list) > 0:
        m_out = append_polys(left_list)
        writevtk(m_out, os.path.join(path, 'joint_left_all_blobs.vtk'))
    if len(right_list) > 0:
        m_out = append_polys(right_list)
        writevtk(m_out, os.path.join(path, 'joint_right_all_blobs.vtk'))



def count_gaps_4veins(path):
    """ Count the number of gaps in each searching area counting the different polydatas created and checking if one
    gap has been splited due to the cut (but it's actually 1 gap). The name of possible splited gaps will always be
    gap_0 and gap_n being n the number of gaps - 1

    Input: path where the polydatas are
    Output: numpy array with number of gaps in [RSPV, RIPV, LIPV, LSPV] """

    all_ngaps = np.zeros(4).astype(int)
    vein = ['rspv', 'ripv', 'lipv', 'lspv' ]

    for i in range(4):
        file_path = path + vein[i] + '_all_gaps.vtk'
        if (os.path.exists(file_path)):
            all_ngaps[i] = len(glob.glob1(path, vein[i] + '_gap*'))   # each one of these files have a gap
            if all_ngaps[i] > 1:  # ngaps can be correctly only 1, don't substract 1 if there is only 1
                # check if the first and last gaps are actually the same gap
                m1 = readvtk(path + vein[i] + '_gap_0_path.vtk')
                m2 = readvtk(path + vein[i] + '_gap_' + str(all_ngaps[i] - 1) + '_path.vtk')
                # if m1 and m2 have a very close point they are the same gap
                # the closest point will always be the first or the last (they are lines)
                min_dist = find_min_euclidean_dist_between_extreme_gaps(m1, m2)
                if min_dist < 0.01:
                    all_ngaps[i] = all_ngaps[i] - 1
    return all_ngaps


def count_gaps_lateral_veins(path):
    """ Count the number of gaps in each searching area (only right/left here) counting the different polydatas created and checking if one
    gap has been splited due to the cut (but it's actually 1 gap). The name of possible splited gaps will always be
    gap_0 and gap_n being n the number of gaps - 1

    Input: path where the polydatas are
    Output: numpy array with number of gaps in [Right, Left] """

    all_ngaps = np.zeros(2).astype(int)
    vein = ['right', 'left']
    for i in range(2):    #right/left only
        file_path = path + 'joint_' + vein[i] + '_all_gaps.vtk'
        if (os.path.exists(file_path)):
            all_ngaps[i] = len(glob.glob1(path, vein[i] + '_gap*'))   # each one of these files have a gap
            if all_ngaps[i] > 1:  # ngaps can be correctly only 1, don't substract 1 if there is only 1
                # check if the first and last gaps are actually the same gap
                m1 = readvtk(path + vein[i] + '_gap_0_path.vtk')
                m2 = readvtk(path + vein[i] + '_gap_' + str(all_ngaps[i] - 1) + '_path.vtk')
                # if m1 and m2 have a very close point they are the same gap
                # the closest point will always be the first or the last (they are lines)
                min_dist = find_min_euclidean_dist_between_extreme_gaps(m1, m2)
                if min_dist < 0.01:
                    all_ngaps[i] = all_ngaps[i] - 1
    return all_ngaps

# From here, Dijkstra algorithm, taken from http://www.bogotobogo.com/python/python_Dijkstras_Shortest_Path_Algorithm.php
class Vertex:
    def __init__(self, node):
        self.id = node
        self.adjacent = {}
        # Set distance to infinity for all nodes
        # self.distance = sys.maxint
        self.distance = np.inf
        # Mark all nodes unvisited
        self.visited = False
        # Predecessor
        self.previous = None

    def add_neighbor(self, neighbor, weight=0):
        self.adjacent[neighbor] = weight

    def get_connections(self):
        return self.adjacent.keys()

    def get_id(self):
        return self.id

    def get_weight(self, neighbor):
        return self.adjacent[neighbor]

    def set_distance(self, dist):
        self.distance = dist

    def get_distance(self):
        return self.distance

    def set_previous(self, prev):
        self.previous = prev

    def set_visited(self):
        self.visited = True

    def __str__(self):
        return str(self.id) + ' adjacent: ' + str([x.id for x in self.adjacent])

    # Added this for python 3
    def __lt__(self, other):
        return self.distance < other.distance


class Graph:
    def __init__(self):
        self.vert_dict = {}
        self.num_vertices = 0

    def __iter__(self):
        return iter(self.vert_dict.values())

    def add_vertex(self, node):
        self.num_vertices = self.num_vertices + 1
        new_vertex = Vertex(node)
        self.vert_dict[node] = new_vertex
        return new_vertex

    def get_vertex(self, n):
        if n in self.vert_dict:
            return self.vert_dict[n]
        else:
            return None

    def add_edge(self, frm, to, cost = 0):
        if frm not in self.vert_dict:
            self.add_vertex(frm)
        if to not in self.vert_dict:
            self.add_vertex(to)

        self.vert_dict[frm].add_neighbor(self.vert_dict[to], cost)
        self.vert_dict[to].add_neighbor(self.vert_dict[frm], cost)

    def get_vertices(self):
        return self.vert_dict.keys()

    def set_previous(self, current):
        self.previous = current

    def get_previous(self, current):
        return self.previous


def shortest(v, path):
    ''' make shortest path from v.previous'''
    if v.previous:
        path.append(v.previous.get_id())
        shortest(v.previous, path)
    return


def dijkstra(aGraph, start, target):
    #print '''Dijkstra's shortest path'''
    # Set the distance for the start node to zero
    start.set_distance(0)

    # Put tuple pair into the priority queue
    unvisited_queue = [(v.get_distance(), v) for v in aGraph]
    heapq.heapify(unvisited_queue)

    while len(unvisited_queue):
        # Pops a vertex with the smallest distance
        uv = heapq.heappop(unvisited_queue)
        current = uv[1]
        current.set_visited()

        #for next in v.adjacent:
        for next in current.adjacent:
            # if visited, skip
            if next.visited:
                continue
            new_dist = current.get_distance() + current.get_weight(next)

            if new_dist < next.get_distance():
                next.set_distance(new_dist)
                next.set_previous(current)
                #print 'updated : current = %s next = %s new_dist = %s' \
                #        %(current.get_id(), next.get_id(), next.get_distance())
            #else:
                #print 'not updated : current = %s next = %s new_dist = %s' \
                #        %(current.get_id(), next.get_id(), next.get_distance())


        # Rebuild heap
        # 1. Pop every item
        while len(unvisited_queue):
            heapq.heappop(unvisited_queue)
        # 2. Put all vertices not visited into the queue
        unvisited_queue = [(v.get_distance(), v) for v in aGraph if not v.visited]
        heapq.heapify(unvisited_queue)


