""" fibers.py

This module contains code for representation of tractography using a
fixed-length parameterization.

class FiberArray

"""

import vtk
import numpy
import os
import torch.utils.data as data
import torch

class Fiber:
    """A class for fiber tractography data, represented with a fixed length"""

    def __init__(self):
        self.r = None
        self.a = None
        self.s = None
        self.points_per_fiber = None
        self.hemisphere_percent_threshold = 0.95
        
    def get_equivalent_fiber(self):
        """ Get the reverse order of current line (trajectory), as the
        fiber can be equivalently represented in either order."""
        
        fiber = Fiber()

        fiber.r = self.r[::-1]
        fiber.a = self.a[::-1]
        fiber.s = self.s[::-1]

        fiber.points_per_fiber = self.points_per_fiber

        return fiber

    def get_reflected_fiber(self):
        """ Returns reflected version of current fiber by reflecting
        fiber across midsagittal plane. Just sets output R coordinate to -R."""
 
        fiber = Fiber()

        fiber.r = - self.r
        fiber.a = self.a
        fiber.s = self.s

        fiber.points_per_fiber = self.points_per_fiber

        return fiber

    def match_order(self, other):
        """ Reverse order of fiber to match this one if needed """
        # compute correlation
        corr = numpy.multiply(self.r, other.r) + \
            numpy.multiply(self.a, other.a) + \
            numpy.multiply(self.s, other.s)

        other2 = other.get_equivalent_fiber()
        corr2 = numpy.multiply(self.r, other2.r) + \
            numpy.multiply(self.a, other2.a) + \
            numpy.multiply(self.s, other2.s)
        
        if numpy.sum(corr) > numpy.sum(corr2):
            return other
        else:
            return other2
        
    def __add__(self, other):
        """This is the + operator for fibers"""
        other_matched = self.match_order(other)
        fiber = Fiber()
        fiber.r = self.r + other_matched.r
        fiber.a = self.a + other_matched.a
        fiber.s = self.s + other_matched.s
        return fiber
    
    def __div__(self, other):
        """ This is to divide a fiber by a number"""
        fiber = Fiber()
        fiber.r = numpy.divide(self.r, other)
        fiber.a = numpy.divide(self.a, other)
        fiber.s = numpy.divide(self.s, other)
        return fiber

    def __mul__(self, other):
        """ This is to multiply a fiber by a number"""
        fiber = Fiber()
        fiber.r = numpy.multiply(self.r, other)
        fiber.a = numpy.multiply(self.a, other)
        fiber.s = numpy.multiply(self.s, other)
        return fiber
    
    def __subtract__(self, other):
        """This is the - operator for fibers"""
        other_matched = self.match_order(other)
        fiber = Fiber()
        fiber.r = self.r - other_matched.r
        fiber.a = self.a - other_matched.a
        fiber.s = self.s - other_matched.s
        #fiber.r = self.r + other_matched.r
        #fiber.a = self.a + other_matched.a
        #fiber.s = self.s + other_matched.s
        return fiber
    
class FiberArray:

    """A class for arrays of fiber tractography data, represented with
    a fixed length"""

    def __init__(self):
        # parameters
        self.points_per_fiber = 10
        self.verbose = 0

        self.number_of_fibers = 0

        # fiber data
        self.fiber_array_r = None
        self.fiber_array_a = None
        self.fiber_array_s = None
        self.fiber_array_ras = None

        # SWM: fiber information to be output
        self.fiber_length = None        # size: n_points x 1
        self.fiber_endpoint = None      # size: n_points x 2
        self.fiber_subID = None         # size: n_points x 1
        self.fiber_Uratio = None        # size: n_points x 1
        self.fiber_hemispheres = None   # size: n_points x 1
    def combine_fiber_arrays(self, fiber_array_list):


        # Initialize combined fiber array with the first fiber array
        self.fiber_array_r = numpy.concatenate([fiber_array.fiber_array_r for fiber_array in fiber_array_list]) if fiber_array_list[0].fiber_array_r is not None else None
        self.fiber_array_a = numpy.concatenate([fiber_array.fiber_array_a for fiber_array in fiber_array_list]) if fiber_array_list[0].fiber_array_a is not None else None
        self.fiber_array_s = numpy.concatenate([fiber_array.fiber_array_s for fiber_array in fiber_array_list]) if fiber_array_list[0].fiber_array_s is not None else None
        self.fiber_length = numpy.concatenate([fiber_array.fiber_length for fiber_array in fiber_array_list]) if fiber_array_list[0].fiber_length is not None else None
        self.fiber_endpoint = numpy.concatenate([fiber_array.fiber_endpoint for fiber_array in fiber_array_list]) if fiber_array_list[0].fiber_endpoint is not None else None
        self.fiber_subID = numpy.concatenate([fiber_array.fiber_subID for fiber_array in fiber_array_list]) if fiber_array_list[0].fiber_subID is not None else None
        self.fiber_Uratio = numpy.concatenate([fiber_array.fiber_Uratio for fiber_array in fiber_array_list]) if fiber_array_list[0].fiber_Uratio is not None else None
        self.fiber_hemispheres = numpy.concatenate([fiber_array.fiber_hemispheres for fiber_array in fiber_array_list]) if fiber_array_list[0].fiber_hemispheres is not None else None
        self.fiber_array_ras = numpy.concatenate([fiber_array.fiber_array_ras for fiber_array in fiber_array_list]) if fiber_array_list[0].fiber_array_ras is not None else None

    def convert_from_polydata(self, input_vtk_polydata, points_per_fiber, verbose=False):

        """Convert input vtkPolyData to the fixed length fiber
        representation of this class.

        The polydata should contain the output of tractography.

        The output is downsampled fibers in array format and
        hemisphere info is also calculated.

        """

        # points used in discretization of each trajectory
        self.points_per_fiber = points_per_fiber
        self.verbose = verbose

        # line count. Assume all input lines are from tractography.
        self.number_of_fibers = input_vtk_polydata.GetNumberOfLines()

        # allocate array number of lines by line length
        self.fiber_array_r = numpy.zeros((self.number_of_fibers, self.points_per_fiber), dtype=numpy.float16)
        self.fiber_array_a = numpy.zeros((self.number_of_fibers, self.points_per_fiber), dtype=numpy.float16)
        self.fiber_array_s = numpy.zeros((self.number_of_fibers, self.points_per_fiber), dtype=numpy.float16)

        self.fiber_length = numpy.zeros(self.number_of_fibers, dtype=numpy.float16)
        self.fiber_endpoint = numpy.zeros((self.number_of_fibers, 2), dtype=numpy.uint16)
        self.fiber_subID = numpy.zeros(self.number_of_fibers, dtype=numpy.int16)
        self.fiber_Uratio = numpy.zeros(self.number_of_fibers, dtype=numpy.float16)
        self.fiber_hemispheres = numpy.zeros(self.number_of_fibers, dtype=numpy.int8)

        # get data from input polydata
        inpoints = input_vtk_polydata.GetPoints()
        inpointdata = input_vtk_polydata.GetPointData()

        point_data_array_indices = list(range(inpointdata.GetNumberOfArrays()))
        for idx in point_data_array_indices:
            array = inpointdata.GetArray(idx)

            if array.GetName() == 'end_label': # For HCP
                input_vtk_polydata.GetLines().InitTraversal()
                for lidx in range(0, input_vtk_polydata.GetNumberOfLines()):
                    ptids = vtk.vtkIdList()
                    input_vtk_polydata.GetLines().GetNextCell(ptids)
                    roi_line = -numpy.ones(ptids.GetNumberOfIds())
                    for pidx in range(0, ptids.GetNumberOfIds()):
                        roi_line[pidx] = array.GetTuple(ptids.GetId(pidx))[0]
                    roi_line[roi_line == 10000] = 0
                    self.fiber_endpoint[lidx, 0] = numpy.uint16(roi_line[0])
                    self.fiber_endpoint[lidx, 1] = numpy.uint16(roi_line[-1])
            # elif array.GetName() == 'ROI_label_wmparc':
            #     input_vtk_polydata.GetLines().InitTraversal()
            #     for lidx in range(0, input_vtk_polydata.GetNumberOfLines()):
            #         ptids = vtk.vtkIdList()
            #         input_vtk_polydata.GetLines().GetNextCell(ptids)
            #         roi_line = -numpy.ones(2) # roi_line = -numpy.ones(ptids.GetNumberOfIds())
            #         for ind, pidx in enumerate([0, ptids.GetNumberOfIds()-1]): #TODO now, we only consider ep.
            #             roi_line[ind] = array.GetTuple(ptids.GetId(pidx))[0]
            #         roi_line[roi_line == 10000] = 0
            #         self.fiber_endpoint[lidx, 0] = roi_line[0]
            #         self.fiber_endpoint[lidx, 1] = roi_line[-1]
            elif array.GetName() == 'cluster_idx':
                input_vtk_polydata.GetLines().InitTraversal()
                for lidx in range(0, input_vtk_polydata.GetNumberOfLines()):
                    ptids = vtk.vtkIdList()
                    input_vtk_polydata.GetLines().GetNextCell(ptids)
                    self.fiber_subID[lidx] = numpy.uint16(array.GetTuple(ptids.GetId(0))[0])
            elif array.GetName() == 'Uratio':
                input_vtk_polydata.GetLines().InitTraversal()
                for lidx in range(0, input_vtk_polydata.GetNumberOfLines()):
                    ptids = vtk.vtkIdList()
                    input_vtk_polydata.GetLines().GetNextCell(ptids)
                    self.fiber_Uratio[lidx] = array.GetTuple(ptids.GetId(0))[0]


        vtk_array_pseq = vtk.vtkDoubleArray()
        vtk_array_pseq.SetName('FiberPointSeq')

        if self.verbose:
            print("<fibers.py> Converting polydata to array representation. Total number of fibers:", self.number_of_fibers)

        # loop over lines
        input_vtk_polydata.GetLines().InitTraversal()
        for lidx in range(0, self.number_of_fibers):

            if self.verbose and lidx % 1000 == 0:
                print("<fibers.py> processing fiber %d / %d:" %(lidx, self.number_of_fibers))

            line_ptids = vtk.vtkIdList()
            input_vtk_polydata.GetLines().GetNextCell(line_ptids)
            num_of_points = line_ptids.GetNumberOfIds()

            # calculate step size based on the first fiber
            # if lidx == 0: #TODO: precompute length
            #     step_size = 0.0
            #     count = 0.0
            #     for ptidx in range(5, line_ptids.GetNumberOfIds() - 5):
            #         point0 = inpoints.GetPoint(line_ptids.GetId(ptidx))
            #         point1 = inpoints.GetPoint(line_ptids.GetId(ptidx + 1))
            #         step_size += numpy.sqrt(numpy.sum(numpy.power(numpy.subtract(point0, point1), 2)))
            #         count += 1
            #     step_size = step_size / count

            self.fiber_length[lidx] = num_of_points

            # add point order to the data
            for pidx in range(0, line_ptids.GetNumberOfIds()):
                vtk_array_pseq.InsertNextTuple1(pidx)

            # loop over the indices that we want and get those points
            # point0 = None
            # point1 = None
            for pidx, line_index in enumerate(self._calculate_line_indices(num_of_points, self.points_per_fiber)):

                # do nearest neighbor interpolation: round index
                ptidx = line_ptids.GetId(int(round(line_index)))
                point = list(inpoints.GetPoint(ptidx))

                # if pidx == 0:
                #     point0 = numpy.array(point)
                # elif pidx == self.points_per_fiber - 1:
                #     point1 = numpy.array(point)

                if point[0] < 0: # left hemisphere fibers - 1
                    self.fiber_hemispheres[lidx] = 1

                point[0] = abs(point[0]) # for bilateral clustering

                self.fiber_array_r[lidx, pidx] = point[0]
                self.fiber_array_a[lidx, pidx] = point[1]
                self.fiber_array_s[lidx, pidx] = point[2]
            #
            # endpoint_dist = numpy.sqrt(numpy.sum(numpy.power(numpy.subtract(point0, point1), 2)))
            # self.fiber_Uratio[lidx] = endpoint_dist / self.fiber_length[lidx]
            # if self.fiber_Uratio[lidx] > 1:
            #     self.fiber_Uratio[lidx] = 1

        # add vtk_array_pseq to vtk
        # inpointdata.AddArray(vtk_array_pseq)
        # inpointdata.Update()

        self.fiber_array_ras = numpy.dstack((self.fiber_array_r, self.fiber_array_a, self.fiber_array_s))
        self.fiber_array_r = None
        self.fiber_array_a = None
        self.fiber_array_s = None

    def __str__(self):
        output = "\n points_per_fiber\t" + str(self.points_per_fiber) \
            + "\n number_of_fibers\t\t" + str(self.number_of_fibers) \
            + "\n fiber_hemisphere\t\t" + str(self.fiber_hemisphere) \
            + "\n verbose\t" + str(self.verbose)

        return output

    def _calculate_line_indices(self, input_line_length, output_line_length):
        """ Figure out indices for downsampling of polyline data.

        The indices include the first and last points on the line,
        plus evenly spaced points along the line.  This code figures
        out which indices we actually want from a line based on its
        length (in number of points) and the desired length.

        """

        # this is the increment between output points
        step = (input_line_length - 1.0) / (output_line_length - 1.0)

        # these are the output point indices (0-based)
        ptlist = []
        for ptidx in range(0, output_line_length):
            ptlist.append(ptidx * step)

        return ptlist

    def get_fiber(self, fiber_index):
        """ Return fiber number fiber_index. Return value is class
        Fiber."""

        fiber = Fiber()
        fiber.r = self.fiber_array_r[fiber_index, :]
        fiber.a = self.fiber_array_a[fiber_index, :]
        fiber.s = self.fiber_array_s[fiber_index, :]

        fiber.points_per_fiber = self.points_per_fiber

        return fiber

    def get_equivalent_fiber(self, fiber_index):
        """ Return equivalent version of fiber number
        fiber_index. Return value is class Fiber. Gets the reverse
        order of line (trajectory), as the fiber can be equivalently
        represented in either order."""
  
        fiber = self.get_fiber(fiber_index)

        return fiber.get_equivalent_fiber()

    def get_fibers(self, fiber_indices):
        """ Return FiberArray containing subset of data corresponding
        to fiber_indices"""
        
        fibers = FiberArray()

        fibers.number_of_fibers = len(fiber_indices)

        # parameters
        fibers.points_per_fiber = self.points_per_fiber
        fibers.verbose = self.verbose

        # fiber data
        fibers.fiber_array_r = self.fiber_array_r[fiber_indices]
        fibers.fiber_array_a = self.fiber_array_a[fiber_indices]
        fibers.fiber_array_s = self.fiber_array_s[fiber_indices]

        # TODO: if needs to return other variables

        return fibers

    def get_oriented_fibers(self, fiber_indices, order):
        """Return FiberArray containing subset of data corresponding to
        fiber_indices. Order fibers according to the array (where 0 is no

        change, and 1 means to reverse the order and return the
        equivalent fiber)
        """

        fibers = FiberArray()

        fibers.number_of_fibers = len(fiber_indices)

        # parameters
        fibers.points_per_fiber = self.points_per_fiber
        fibers.verbose = self.verbose

        # fiber data
        fibers.fiber_array_r = self.fiber_array_r[fiber_indices]
        fibers.fiber_array_a = self.fiber_array_a[fiber_indices]
        fibers.fiber_array_s = self.fiber_array_s[fiber_indices]

        # swap orientation as requested
        for (ord, fidx) in zip(order, range(fibers.number_of_fibers)):
            if ord == 1:
                f2 = fibers.get_equivalent_fiber(fidx)
                # replace it in the array
                fibers.fiber_array_r[fidx,:] = f2.r
                fibers.fiber_array_a[fidx,:] = f2.a
                fibers.fiber_array_s[fidx,:] = f2.s

        return fibers

    def convert_to_polydata(self):
        """Convert fiber array to vtkPolyData object."""

        outpd = vtk.vtkPolyData()
        outpoints = vtk.vtkPoints()
        outlines = vtk.vtkCellArray()

        outlines.InitTraversal()

        for lidx in range(0, self.number_of_fibers):
            cellptids = vtk.vtkIdList()

            for pidx in range(0, self.points_per_fiber):

                idx = outpoints.InsertNextPoint(self.fiber_array_r[lidx, pidx],
                                                self.fiber_array_a[lidx, pidx],
                                                self.fiber_array_s[lidx, pidx])

                cellptids.InsertNextId(idx)

            outlines.InsertNextCell(cellptids)

        # put data into output polydata
        outpd.SetLines(outlines)
        outpd.SetPoints(outpoints)

        return outpd    


class FiberPair(data.Dataset):

    def __init__(self, vec, subID, index_pair,subID_counts,similarity_path,dmri_similarity_path,bundle,alpha,transform=None, embedding_surf=False,funct=True):
        self.vec = vec #8000*14*3
        self.transform = transform
        self.subID = subID
        self.embedding_surf = embedding_surf
        self.similarity=None
        self.index_pair=index_pair
        self.subID_counts=subID_counts
        self.similarity_path=similarity_path
        self.dmri_similarity_path=dmri_similarity_path
        self.bundle=bundle
        self.alpha=alpha
    def get_epoch_similarity(self, epoch):
        """
        Set the current epoch number.

        Args:
            epoch (int): The current epoch number.
        """

        if epoch == 0:
            self.pair = torch.stack(list(self.index_pair.values()),dim=0).permute(1,0,2,3).flatten(start_dim=1,end_dim=-2).cpu().numpy()
        self.epoch_index2=self.pair[epoch]
        if self.alpha!=0:
            self.similarity=torch.load(os.path.join(self.similarity_path,f"similarities_{self.bundle}_{epoch}.pt"),weights_only=True,map_location=torch.device('cuda:0'))
        elif self.dmri_similarity_path is not None:
            self.similarity=torch.load(os.path.join(self.dmri_similarity_path,f"dmri_{self.bundle}_epoch_{epoch}.pt"),weights_only=True,map_location=torch.device('cuda:0'))
        
    def __getitem__(self, index: int):
        
        index1 = index
        index2 = -self.epoch_index2[index][0]+self.epoch_index2[index][1]+index
        fiber1 = self.vec[index1]
        fiber2 = self.vec[index2]
        if self.similarity_path is not None:
            similarity = self.similarity[index1]
        else:
            similarity=torch.tensor(0,dtype=torch.float)
        subID = self.subID[index1]
        fiber1 = torch.tensor(fiber1.T, dtype=torch.float)
        fiber2 = torch.tensor(fiber2.T, dtype=torch.float)
        return fiber1, fiber2, similarity, index, subID

    def __len__(self) -> int:
        return len(self.vec)


def _fiber_distance_internal_use(fiber_r, fiber_a, fiber_s, fiber_array, threshold=0, distance_method='Mean'):
    """ Compute the total fiber distance from one fiber to an array of
    many fibers.
    This function does not handle equivalent fiber representations,
    for that use fiber_distance, above.
    """

    fiber_array_r = fiber_array[:, :, 0]
    fiber_array_a = fiber_array[:, :, 1]
    fiber_array_s = fiber_array[:, :, 2]

    # compute the distance from this fiber to the array of other fibers
    ddx = fiber_array_r - fiber_r
    ddy = fiber_array_a - fiber_a
    ddz = fiber_array_s - fiber_s

    dx = numpy.square(ddx)
    dy = numpy.square(ddy)
    dz = numpy.square(ddz)

    # sum dx dx dz at each point on the fiber and sqrt for threshold
    # distance = numpy.sqrt(dx + dy + dz)
    distance = dx + dy + dz

    # threshold if requested
    if threshold:
        # set values less than threshold to 0
        distance = distance - threshold * threshold
        idx = numpy.nonzero(distance < 0)
        distance[idx] = 0

    if distance_method == 'Mean':
        # sum along fiber
        distance = numpy.sum(numpy.sqrt(distance), 1)
        # Remove effect of number of points along fiber (mean)
        npts = float(fiber_array.shape[1])
        #print(npts)
        distance = distance / npts
        # for consistency with other methods we need to square this value
        #distance = numpy.square(distance)
    elif distance_method == 'Hausdorff':
        # take max along fiber
        distance = numpy.max(distance, 1)
    elif distance_method == 'MeanSquared':
        # sum along fiber
        distance = numpy.sum(distance, 1)
        # Remove effect of number of points along fiber (mean)
        npts = len(fiber_r)
        distance = distance / npts
    elif distance_method == 'StrictSimilarity':
        # for use in laterality
        # this is the product of all similarity values along the fiber
        # not truly a distance but it's easiest to compute here in this function
        # where we have all distances along the fiber
        # print "distance range :", numpy.min(distance), numpy.max(distance)
        #distance = distance_to_similarity(distance, sigmasq)
        # print "similarity range :", numpy.min(distance), numpy.max(distance)
        distance = numpy.prod(distance, 1)
        # print "overall similarity range:", numpy.min(distance), numpy.max(distance)
    elif distance_method == 'Mean_shape':

        # sum along fiber
        distance_square = distance
        distance = numpy.sqrt(distance_square)

        d = numpy.sum(distance, 1)
        # Remove effect of number of points along fiber (mean)
        npts = float(fiber_array.points_per_fiber)
        d = numpy.divide(d, npts)
        # for consistency with other methods we need to square this value
        d = numpy.square(d)

        distance_endpoints = (distance[:, 0] + distance[:, npts - 1]) / 2

        for i in numpy.linspace(0, numpy.size(distance, 0) - 1, numpy.size(distance, 0)):
            for j in numpy.linspace(0, numpy.size(distance, 1) - 1, numpy.size(distance, 1)):
                if distance[i, j] == 0:
                    distance[i, j] = 1
        ddx = numpy.divide(ddx, distance)
        ddy = numpy.divide(ddy, distance)
        ddz = numpy.divide(ddz, distance)
        # print ddx*ddx+ddy*ddy+ddz*ddz
        npts = float(fiber_array.points_per_fiber)
        angles = numpy.zeros([(numpy.size(distance)) / npts, npts * (npts + 1) / 2])
        s = 0
        n = numpy.linspace(0, npts - 1, npts)
        for i in n:
            m = numpy.linspace(0, i, i + 1)
            for j in m:
                angles[:, s] = (ddx[:, i] - ddx[:, j]) * (ddx[:, i] - ddx[:, j]) + (ddy[:, i] - ddy[:, j]) * (
                            ddy[:, i] - ddy[:, j]) + (ddz[:, i] - ddz[:, j]) * (ddz[:, i] - ddz[:, j])
                s = s + 1
        angles = (numpy.sqrt(angles)) / 2
        angle = numpy.max(angles, 1)

        distance = 0.5 * d + 0.4 * d / (0.5 + 0.5 * (1 - angle * angle)) + 0.1 * distance_endpoints

    else:
        print("<fibers.py> throwing Exception. Unknown input distance method (typo?):", distance_method)
        raise Exception("unknown distance method")

    return distance

def fiber_distance(fiber, fiber_array, threshold=0, distance_method='Mean', bilateral=False):
    """
    Find pairwise fiber distance from fiber to all fibers in fiber_array.
    The Mean and MeanSquared distances are the average distance per
    fiber point, to remove scaling effects (dependence on number of
    points chosen for fiber parameterization). The Hausdorff distance
    is the maximum distance between corresponding points.
    input fiber should be class Fiber. fibers should be class FiberArray
    """
    fiber_r = fiber[:, 0]
    fiber_a = fiber[:, 1]
    fiber_s = fiber[:, 2]

    # get fiber in reverse point order, equivalent representation
    fiber_r_quiv = fiber_r[::-1]
    fiber_a_quiv = fiber_a[::-1]
    fiber_s_quiv = fiber_s[::-1]

    # compute pairwise fiber distances along fibers
    distance_1 = _fiber_distance_internal_use(fiber_r, fiber_a, fiber_s, fiber_array, threshold, distance_method)
    distance_2 = _fiber_distance_internal_use(fiber_r_quiv, fiber_a_quiv ,fiber_s_quiv, fiber_array, threshold, distance_method)

    # choose the lowest distance, corresponding to the optimal fiber
    # representation (either forward or reverse order)
    if distance_method == 'StrictSimilarity':
        # for use in laterality
        # this is the product of all similarity values along the fiber
        distance = numpy.maximum(distance_1, distance_2)
    else:
        distance = numpy.minimum(distance_1, distance_2)

    if bilateral:
        fiber_r_ref = -fiber_r
        fiber_reflect=numpy.stack((fiber_r_ref,fiber_a,fiber_s)).transpose([1,0])
        # call this function again with the reflected fiber. Do NOT reflect again (bilateral=False) to avoid infinite loop.
        distance_reflect = fiber_distance(fiber_reflect, fiber_array, threshold, distance_method, bilateral=False)
        # choose the best distance, corresponding to the optimal fiber
        # representation (either reflected or not)
        if distance_method == 'StrictSimilarity':
            # this is the product of all similarity values along the fiber
            distance = numpy.maximum(distance, distance_reflect)
        else:
            distance = numpy.minimum(distance, distance_reflect)

    return distance


def fiber_pair_distance(fiber1, fiber2):

    fiber_r1 = fiber1[:, 0]
    fiber_a1 = fiber1[:, 1]
    fiber_s1 = fiber1[:, 2]
    fiber_r2 = fiber2[:, 0]
    fiber_a2 = fiber2[:, 1]
    fiber_s2 = fiber2[:, 2]

    ddx = fiber_r1 - fiber_r2
    ddy = fiber_a1 - fiber_a2
    ddz = fiber_s1 - fiber_s2

    dx = numpy.square(ddx)
    dy = numpy.square(ddy)
    dz = numpy.square(ddz)

    distance = dx + dy + dz
    distance = numpy.sum(numpy.sqrt(distance))
    npts = len(dx)
    distance = distance / npts

    return distance

def fiber_pair_similarity(fiber1, fiber2, surf1=None, surf2=None):

    distance1 = fiber_pair_distance(fiber1, fiber2)

    fiber1_equiv =numpy.array(fiber1,copy=True)
    fiber1_equiv[:, 0] = fiber1[::-1, 0]
    fiber1_equiv[:, 1] = fiber1[::-1, 1]
    fiber1_equiv[:, 2] = fiber1[::-1, 2]

    distance2 = fiber_pair_distance(fiber1_equiv, fiber2)

    distance = numpy.minimum(distance1, distance2)

    if surf1 is not None and surf2 is not None:
        eps1 = numpy.argwhere(surf1 == 1)
        if eps1.shape[0] == 1:
            eps1 = numpy.array([eps1[0], eps1[0]])
        eps2 = numpy.argwhere(surf2 == 1)
        if eps2.shape[0] == 1:
            eps2 = numpy.array([eps2[0], eps2[0]])
        eps12_int = numpy.intersect1d(eps1, eps2)
        eps12_uni = numpy.union1d(eps1, eps2)

        surf_score = 2 - eps12_int.shape[0] / eps12_uni.shape[0]

        if surf_score > 1:
            surf_score = 2

    else:
        surf_score = 1.0

    distance *= surf_score

    return  distance
