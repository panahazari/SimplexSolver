"""
Simplex Solver
Author: Mohammad Panahazari
Email: mohammad.panahazari@tufts.edu
Date: February 27, 2024
"""

import numpy as np

class SimplexSolver:
    def __init__(self, c, A, b, op_type="max"):
        """
        Initialize the SimplexSolver with problem parameters.
        
        Parameters:
        - c: Coefficients of the objective function.
        - A: Constraint coefficients.
        - b: Right-hand side values of the constraints.
        - op_type: Type of optimization ("max" for maximization, "min" for minimization).
        """
        self.c = np.array(c, dtype=float)
        self.A = np.array(A, dtype=float)
        self.b = np.array(b, dtype=float).reshape(-1, 1)  # Ensure b is a column vector
        self.op_type = op_type

    def solve(self):
        """
        Solve the linear programming problem using the simplex method.
        
        Returns:
        A tuple of the solution to the decision variables and the objective value.
        """
        if self.op_type == "min":
            self.A = np.transpose(self.A)
            b_, self.b = self.b, self.c.reshape(-1, 1)  # Swap b and c for minimization
            self.c = b_.flatten()  # Flatten b_ back to a 1D array for c
        
        num_vars = len(self.c)
        num_constraints = len(self.b)
        
        # Prepare the tableau
        tableau = np.hstack((self.A, np.eye(num_constraints), np.zeros((num_constraints, 1)), self.b))
        c_extended = np.hstack((self.c, np.zeros(num_constraints + 1)))
        tableau = np.vstack((tableau, -np.hstack((c_extended, 0))))
        tableau[-1, -2] = 1  # Set the coefficient for the objective's slack variable
        
        basis = [num_vars + i for i in range(num_constraints)]
        
        while True:
            if np.all(tableau[-1, :-1] >= 0):
                return self._extract_solution(tableau, basis, num_vars, num_constraints)
            
            entering = np.argmin(tableau[-1, :-1])
            ratios = tableau[:-1, -1] / tableau[:-1, entering]
            ratios[ratios <= 0] = np.inf
            leaving = np.argmin(ratios)
            
            if np.isinf(ratios[leaving]):
                raise Exception("Problem is unbounded.")
            
            tableau = self._pivot(tableau, entering, leaving)
            basis[leaving] = entering

    def _pivot(self, tableau, entering, leaving):
        """
        Perform pivot operation on the tableau.
        
        Parameters:
        - tableau: Current simplex tableau.
        - entering: Index of the entering variable.
        - leaving: Index of the leaving variable.
        
        Returns:
        Updated tableau after the pivot.
        """
        pivot = tableau[leaving, entering]
        tableau[leaving, :] /= pivot
        for i in range(len(tableau)):
            if i != leaving:
                tableau[i, :] -= tableau[i, entering] * tableau[leaving, :]
        return tableau

    def _extract_solution(self, tableau, basis, num_vars, num_constraints):
        """
        Extract the solution and objective value from the final tableau.
        
        Parameters:
        - tableau: Final simplex tableau.
        - basis: List of basis variable indices.
        - num_vars: Number of decision variables.
        - num_constraints: Number of constraints.
        
        Returns:
        Solution to the decision variables and the objective value.
        """
        if self.op_type == "max":
            solution = np.zeros(num_vars)
            for i in range(num_constraints):
                if basis[i] < num_vars:
                    solution[basis[i]] = tableau[i, -1]
            objective_value = tableau[-1, -1]
        else:
            
            solution = np.zeros(num_constraints)
            for i in range(num_constraints):
                solution[i] = tableau[-1, num_vars + i]
                
            objective_value = tableau[-1, -1]
        return solution, objective_value
        
