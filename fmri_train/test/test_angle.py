import numpy
import whitematteranalysis as wma
import utils.fibers as fibers

vtk1 = "/Users/fan/Desktop/test_angle/FiberBundle_1.vtk" 
vtk2 = "/Users/fan/Desktop/test_angle/FiberBundle_2.vtk" 

pd1 = wma.io.read_polydata(vtk1)
pd2 = wma.io.read_polydata(vtk2)

fiber_array = fibers.FiberArray()
fiber_array.convert_from_polydata(pd1, points_per_fiber=30)
v2_1 = fiber_array.v2
v3_1 = fiber_array.v3

fiber_array.convert_from_polydata(pd2, points_per_fiber=30)
v2_2 = fiber_array.v2
v3_2 = fiber_array.v3

# v2_1 = fiber_array.v2
# v3_1 = fiber_array.v3

print(v2_1[0, :], v3_1[0, :])
print(v2_2[0, :], v3_2[0, :])

v3_ang = numpy.abs(numpy.dot(v3_1[0, :], v3_2[1, :]))
v2_ang = numpy.dot(v2_1[0, :], v2_2[1, :])
print(v2_ang, v3_ang)

v2_ang = (1 + v2_ang) / 2
f_ang = v2_ang * v3_ang
f_ang = 1 / ((f_ang - 1) / 2 + 1)

print("v2_ang:", v2_ang, "v3_ang:", v3_ang, "f_ang:", f_ang)