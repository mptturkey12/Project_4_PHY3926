import TurkstraMichael_Project4 as proj4
from numpy import linalg as LA

print("\n### Test 1 ###\n")
Matrix = [[1,2,3],[4,5,6],[7,8,9]]
print(LA.eigvals(Matrix))
stability = proj4.check_stability(Matrix)

print("\n### Test 2 ###\n")
Matrix = [[-1,0,0],[0,-2,0],[0,0,-3]]
print(LA.eigvals(Matrix))
stability = proj4.check_stability(Matrix)