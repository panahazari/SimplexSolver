from SimplexSolver import SimplexSolver
from scipy.optimize import linprog
import numpy as np

# Maximization Problem Formulation:
# Objective: Maximize Z = 3x1 + 5x2
# Subject to:
# 1x1 + 0x2 <= 4
# 0x1 + 2x2 <= 12
# 3x1 + 2x2 <= 18
# And x1, x2 >= 0
c_max = [3, 5]
A_max = [[1, 0], [0, 2], [3, 2]]
b_max = [4, 12, 18]
max_problem = SimplexSolver(c=c_max, A=A_max, b=b_max, op_type="max")
solution_max, objective_value_max = max_problem.solve()
print("Maximization Problem Solution with SimplexSolver:")
print("Solution:", solution_max)
print("Objective Value:", objective_value_max)

# Validation with linprog for Maximization Problem
res_max = linprog(c=-1*np.array(c_max), A_ub=A_max, b_ub=b_max, method='highs')
print("Validation with linprog for Maximization:")
print("Solution:", res_max.x)
print("Objective Value:", -res_max.fun)
print()

# Minimization Problem Formulation:
# Objective: Minimize Z = 12x1 + 16x2
# Subject to:
# x1 + 2x2 >= 40
# x1 + x2 >= 30
# And x1, x2 >= 0
c_min = [12, 16]
A_min = [[1, 2], [1, 1]]  # Directly using "greater than or equal to" constraints
b_min = [40, 30]  # Direct values for "greater than or equal to"
min_problem = SimplexSolver(c=c_min, A=A_min, b=b_min, op_type="min")
solution_min, objective_value_min = min_problem.solve()
print("Minimization Problem Solution with SimplexSolver:")
print("Solution:", solution_min)
print("Objective Value:", objective_value_min)

# Validation with linprog for Minimization Problem
# Adjusting for linprog by using A * -1 to represent the "greater than" as "less than" in linprog format
res_min = linprog(c=c_min, A_ub=-1*np.array(A_min), b_ub=-1*np.array(b_min), method='highs')
print("Validation with linprog for Minimization:")
print("Solution:", res_min.x)
print("Objective Value:", res_min.fun)
