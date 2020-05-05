"""
    Copyright (c) - Marta Nunez Garcia
    This program is free software: you can redistribute it and/or modify it under the terms of the GNU General
    Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option)
    any later version. This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
    without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
    Public License for more details. You should have received a copy of the GNU General Public License along with this
    program. If not, see <http://www.gnu.org/licenses/>.
"""

"""
    Compute Relative Gap Measure considering ipsilateral PV isolation (the 2 PV on the same side very jointly isolated).

    Input: LA mesh already flattened with 'scar' scalar array. 
           Some polydatas generated during the LA flattening pipeline (steps 1, 2, 3, and 4) will be also used.
    Output: Gap quantification results summarized in an Excel file. Gaps and paths-connecting-gaps polydatas for visualization.
    Usage: python 5_compute_RGM_lateral_veins.py --meshfile data/mesh.vtk
    
    NOTE: almost same code as 5_compute_RDM_4veins.py but it changes how to cut/open the mesh to ensure closed 
    encircling paths. Results are saved separately too (separated Excel file).
"""

from aux_functions import *
from sys import platform
import argparse
import xlsxwriter

parser = argparse.ArgumentParser()
parser.add_argument('--meshfile', type=str, metavar='PATH', help='path to input mesh')
args = parser.parse_args()

if os.path.isfile(args.meshfile)==False:
    sys.exit('ERROR: input file does not exist')
else:
    mesh = readvtk(args.meshfile)     # original LA mesh (i.e. with long PVs, LAA, and MV).
fileroot = os.path.dirname(args.meshfile)
filename = os.path.basename(args.meshfile)
filename_base = os.path.splitext(filename)[0]

m2d = readvtk(fileroot + '/' + filename_base + '_clipped_c_flat.vtk')           # Input mesh after flattening
m3d = readvtk(fileroot + '/' + filename_base + '_clipped_c_to_be_flat.vtk')     # 3D version of previous mesh (i.e. 3D LA after PVs, LAA and MV clipping)
m3d_ = readvtk(fileroot + '/' + filename_base + '_clipped_c.vtk')
lc = readvtk(fileroot + '/' + filename_base + '_clipped_cpath3.vtk')            # Left carina path (polydata). CAREFUL, these paths were computed on the surface of the clipped & closed mesh
rc = readvtk(fileroot + '/' + filename_base + '_clipped_cpath1.vtk')            # Rigth carina path (polydata)
upline = readvtk(fileroot + '/' + filename_base + '_clipped_cpath4.vtk')        # Line connecting the 2 superior PVs
downline = readvtk(fileroot + '/' + filename_base + '_clipped_cpath2.vtk')      # Line connecting the 2 inferior PVs

# Create folder to save gap quantification results (including gap/encircling path polydatas)
# Create also output Excel file
# Read also 2D template
if platform == 'linux' or platform == 'linux2':
    if not os.path.isdir(fileroot + '/gap_quantification_results'):
        os.system('mkdir ' + fileroot + '/gap_quantification_results')
    workbook = xlsxwriter.Workbook(fileroot + '/gap_quantification_results/RGM_results_joint_lateral_veins.xlsx')
    m_disk_ref = readvtk('data/2D_template_divided.vtk')
    string_path = fileroot + '/gap_quantification_results/'
elif platform == 'win32':
    if not os.path.isdir(fileroot + '\gap_quantification_results'):
        os.system('mkdir ' + fileroot + '\gap_quantification_results')
    workbook = xlsxwriter.Workbook(fileroot + '\gap_quantification_results\RGM_results_joint_lateral_veins.xlsx')
    m_disk_ref = readvtk('data\\2D_template_divided.vtk')
    string_path = fileroot + '\gap_quantification_results\\'
else:
    sys.exit('Unknown operating system. Distance transform can not be computed.')


worksheet = workbook.add_worksheet()
# format cells. This is the format for text and real numbers
format = workbook.add_format()
format.set_align('center')               # center the text in the cells
format.set_num_format('0.0000')          # 4 decimals

format_int = workbook.add_format()       # format for integers (number of gaps)
format_int.set_align('center')

worksheet.write(0, 1, 'Right', format)
worksheet.write(0, 2, 'Left', format)
worksheet.write(1, 0, 'Gap length', format)
worksheet.write(2, 0, 'RGM', format)
worksheet.write(3, 0, '# gaps', format)
worksheet.set_column('A:C', 15)  # Widen columns.

# Remove ad-hoc scalar arrays from '_to_be_flat' mesh. They don't bother but are irrelevant, there is no reason to keep them.
m3d.GetPointData().RemoveArray('pv')
m3d.GetPointData().RemoveArray('autolabels')
m3d.GetPointData().RemoveArray('hole')
writevtk(m3d, fileroot + '/' + filename_base + '_clipped_c_to_be_flat.vtk')

# Detect and mark the PV surroundings using a reference 2D template
transfer_all_scalar_arrays(m_disk_ref, m2d)
transfer_all_scalar_arrays_by_point_id(m2d, m3d)

# Cut the mesh to define starting and ending points surrounding each PV. Use dividing lines computed with 3_divide_LA.py
# Use lines delimiting the posterior wall (easier to identify)
# Transfer carinas and lines first to the initial LA mesh and then to the m3d (the mesh that was flattened and has point correspondence with the disk)
mesh_aux = transfer_carinas_and_lines(m3d_, lc, rc, upline, downline)   # IMPORTANT. I will look for all points closer to ALL points in the paths, 'mesh' must be the mesh used to computed those paths.
transfer_all_scalar_arrays(mesh_aux, m3d)
# open the mesh eliminating the points where the carina's and up/down lines paths are.
open_mesh = pointthreshold(m3d, 'carinas', float(0), float(0.99999))
open_mesh2 = pointthreshold(open_mesh, 'lines', float(0), float(0.99999))


# extract connected regions: one is the posterior wall and the other one the rest of the LA
connect = get_connected_components(open_mesh2)
ncontours = connect.GetNumberOfExtractedRegions()
cc = connect.GetOutput()
if ncontours == 1:
    sys.exit('Unable to separate the posterior wall from the rest of the LA. Try with different seeds.')
if ncontours > 2:
    print('WARNING: the separation of the posterior wall gives more than 2 regions. This can cause problems with the opening of some vein/s')
transfer_array(cc, m3d, 'RegionId', 'up_down_label')


pv_labels = ['left', 'right']
for vein_i in range(2):
    pv_label = pv_labels[vein_i]
    vein = pointthreshold(m3d, pv_label, float(1), float(1))

    outfile_vein = string_path + filename_base + '_joint_' + pv_label + '.vtk'
    writevtk(vein, outfile_vein)

    # open mesh in the carinas and in the superior line. Limit in the superior line will be blob_0 and blob_n
    if vein_i == 0:  # left veins
        [cut_vein, ids_lim1, ids_lim2] = cut_mesh_carina_and_up(vein, 'up_down_label', 0, 1, 'carinas', 1, 'lines', 1)

    else:           # right veins
        [cut_vein, ids_lim1, ids_lim2] = cut_mesh_carina_and_up(vein, 'up_down_label', 0, 1, 'carinas', 2, 'lines', 1)

    indexes = np.unique(ids_lim1, return_index=True)[1]    # non-repeated values but keeping the order (unique reorders the values from min to max)
    blob_0 = ids_lim1[np.sort(indexes)]
    indexes = np.unique(ids_lim2, return_index=True)[1]
    blob_n = ids_lim2[np.sort(indexes)]
    blob_0 = blob_0.astype(int)
    blob_n = blob_n.astype(int)

    transfer_array(vein, cut_vein, 'scar', 'scar')

    # Find scar patches (blobs) and save the resulting mesh (i.e. portion of LA surrounding a PV with added duplicated
    # points in a detected border to allow the computation of a closed surrounding path)
    final_vein = find_blobs(cut_vein)
    outfile_open = string_path + filename_base + '_joint_' + pv_label + '_final.vtk'
    writevtk(final_vein, outfile_open)

#         # Extract the edges (write them) to visualize them later
#         edges = extractboundaryedge(final_vein)
#         outfile2 = os.path.join(fileroot, 'vein_'+ pv_label + '_edges.vtk')
#         writevtk(edges, outfile2)

    mesh = final_vein

    blob_ids = vtk_to_numpy(mesh.GetPointData().GetArray('blob'))
    nblobs = len(np.unique(blob_ids))-1  # -1 because that is the value when there is no scar (and not numerated blob therefore)

    # Get the matrix with inter-blob distances (graph). Use C++ code to compute Distance Transform using fast marching
    # Create new blob array incorporating artificial blobs. Now blob = 0 means artificial blob_0
    new_blob_array = blob_ids + 1
    new_blob_array[blob_ids == -1] = -1   # blob_ids = -1 means NO BLOB (no scar). Those points have to be -1
    new_blob_array[blob_0] = 0
    new_blob_array[blob_n] = nblobs + 1

    newarray = vtk.vtkDoubleArray()
    newarray = numpy_to_vtk(new_blob_array)
    newarray.SetName('all_blobs')
    newarray.SetNumberOfTuples(mesh.GetNumberOfPoints())
    mesh.GetPointData().AddArray(newarray)
    filename_mesh_art_blobs = string_path + filename_base + '_joint_' + pv_label + '_with_all_blobs.vtk'
    writevtk(mesh, filename_mesh_art_blobs)

    # first and last paths must be calculated at the same time as we want to enforce a closed path (where one path finishes the next one starts)
    dual_points = find_dual_points_limits(mesh, blob_0, blob_n)  # OK, in each column two dual points (row 0 has points from blob0, row 1 has points from blobn)
    dual_points = dual_points.astype(dtype=int)
    blob_ids = blob_ids.astype(dtype=int)
    #print 'Dual points: ', dual_points

    if dual_points.size == 0:     # Do this before DT calculation to aboid memory crash
        print('Not closed surface around the veins, not contact between artificial limits')
        continue

    # Distance transform computation. Output mesh will have in DT array a tuple with the distances of the corresponding point to each blob
    filename_mesh_DT = string_path + filename_base + '_joint_' + pv_label + '_DT.vtk'

    if platform == 'linux' or platform == 'linux2':
        os.system('./DistanceTransformMesh ' + filename_mesh_art_blobs + ' all_blobs ' + filename_mesh_DT)
    elif platform == 'win32':
        os.system('DistanceTransform_Windows\DistanceTransformMesh.exe ' + filename_mesh_art_blobs + ' all_blobs ' + filename_mesh_DT)
    else:
        sys.exit('Unknown operating system. Distance transform can not be computed.')

    matrix_DT_mesh = readvtk(filename_mesh_DT)
    matrix_DT = matrix_DT_mesh.GetPointData().GetArray('DT')
    distance_transforms = vtk_to_numpy(matrix_DT)

    # check and correct the distance_transforms matrix for the cases where a small blob in the limit dissapears because I change its label to blob0 or blobn
    for bb in range(nblobs):
        blob_j = np.where(new_blob_array == bb)[0]
        if blob_j.size == 0:
            print('Blob with index(new): ', bb, ' dissapeared')
            # I have to calculate distance transform for this blob. Use my old function
            blob_j = np.where(blob_ids == bb-1)[0]

            # Compute distance transform WITHOUT artificial blobs
            filename_mesh_auxDT = string_path + filename_base + '_joint_' + pv_label + 'aux_DT.vtk'
            if platform == 'linux' or platform == 'linux2':
                os.system('./DistanceTransformMesh ' + outfile_open + ' blob ' + filename_mesh_auxDT)
            elif platform == 'win32':
                os.system('DistanceTransform_Windows\DistanceTransformMesh.exe ' + outfile_open + ' blob ' + filename_mesh_auxDT)
            else:
                sys.exit('Unknown operating system. Distance transform can not be computed.')

            matrix_auxDT_mesh = readvtk(filename_mesh_auxDT)
            matrix_auxDT = matrix_auxDT_mesh.GetPointData().GetArray('DT')
            distance_transforms_aux = vtk_to_numpy(matrix_auxDT)
            # update with correct distance tranform of the missing blob
            distance_transforms[:, bb] = distance_transforms_aux[:, bb-1]    # aux has 2 less columns. bb cannot be 0. new_blob_array == 0 means artificial blob0 and that one cannot 'dissapear'
            # update all the distances to the points of the missing blob. Carefull with the sizes
            distance_transforms_aux2 = np.insert(distance_transforms_aux, 0, distance_transforms[:, 0], axis=1)   # copy first column
            distance_transforms_aux3 = np.insert(distance_transforms_aux2, nblobs-1, distance_transforms[:, nblobs-1], axis=1)   # copy last column
            distance_transforms[blob_j, :] = distance_transforms_aux3[blob_j, :]

    # For each point in the limit where I opened the mesh create the graph assuming that the path starts/end there.
    # The final shortest path will we the shortest of those shortest paths
    [min_path, min_total_gap_length, cross_point_0, cross_point_n, min_min_distances, closest_points_matrix, min_gap_length_array] = find_closed_shortest_path(mesh, distance_transforms, dual_points, blob_ids, new_blob_array)
    if min_path == 'not_closed':
       continue

    n_mins_pos = np.where(min_gap_length_array == np.min(min_gap_length_array))
    n_mins = np.shape(n_mins_pos)[1]

    path = min_path
    total_gap_length = min_total_gap_length

    print('The shortest path: %s' % (path[::-1]))
    print('The amount of gap in this vein is: %f' % total_gap_length)
    worksheet.write(1, vein_i + 1, total_gap_length, format)

    # Go through the whole path and create polydatas (gap or within_blob). Compute and write Relative Gap Measure (RGM)
    if (total_gap_length!=0):

        # Adapt the output of the Dijkstra's code to path indexes
        vertex_names = np.empty(nblobs+1, dtype=object)     # include 'limit1' in the first position
        vertex_names[0] = 'limit1'
        for i in range(nblobs):
            vertex_names[i+1] = 'blob_' + str(i)

        path_indices = np.zeros(len(path))
        for ind in range(len(path)-1):   # Careful! limit2 no esta en vertex_names
            path_indices[ind] = np.where(vertex_names == path[len(path)-1-ind])[0]
        # add the last one
        path_indices[len(path)-1] = np.size(min_min_distances, 1)-1
        # In path_indices I have the info about the blobs in the path. Indexes correspond to the order in matrix min_distances:
        # Example:  path_indices = 0 3 1 5 6
        # Means the path is: blob_0_artificial -- blob_2 (real) -- blob_0 (real) -- blob_4 (real) -- blob_n_artificial
        real_path_indices = path_indices-1

        # Create polydatas with the path and return also the length of the total scar_path to compute RGM
        total_scar_path = create_polydatas_path(mesh, path_indices.astype(int), min_min_distances,
                                                closest_points_matrix.astype(int), int(cross_point_0),
                                                int(cross_point_n), blob_ids, new_blob_array,
                                                string_path, pv_label, n_mins_pos, dual_points.astype(int))
        RGM = total_gap_length/(total_gap_length+total_scar_path)
    else:
        RGM = 0

    print('RGM = %f' %RGM + '\n' + '\n')
    worksheet.write(2, vein_i + 1, RGM, format)

# merge all polydatas
merge_polydatas_gaps_blobs_lateral_veins(string_path)
# count gaps, write info
nb_gaps = count_gaps_lateral_veins(string_path)
worksheet.write(3, 1, nb_gaps[0], format_int)
worksheet.write(3, 2, nb_gaps[1], format_int)
workbook.close()

# Delete intermediate results polydatas. rm/del warning if a file does not exist but it will continue the loop
if platform == 'linux' or platform == 'linux2':
    command = 'rm '
else:
    command = 'del '
for i in range(len(pv_labels)):    # pv_labels = ['left', 'right']
    os.system(command + string_path + pv_labels[i] + '_gap_*.vtk')
    os.system(command + string_path + pv_labels[i] + '_dist_within_*.vtk')
    os.system(command + string_path + 'mesh_joint_' + pv_labels[i] + '.vtk')
    os.system(command + string_path + 'mesh_joint_' + pv_labels[i] + '_DT.vtk')
    os.system(command + string_path + 'mesh_joint_' + pv_labels[i] + '_final.vtk')
    os.system(command + string_path + 'mesh_joint_' + pv_labels[i] + '_with_all_blobs.vtk')

print('Gap quantification done. \n')