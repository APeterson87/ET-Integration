import NRPy_param_funcs as par
# The indexedexp module defines various functions for defining and managing indexed quantities like tensors and pseudotensors
import indexedexp as ixp
# The grid module defines various parameters related to a numerical grid or the dimensionality of indexed expressions
# For example, it declares the parameter DIM, which specifies the dimensionality of the indexed expression
import grid as gri
from outputC import *
import sympy
from sympy import symbols, IndexedBase, Indexed, Idx, preorder_traversal
import numpy as np


Nx, Ny, Nz, Nn= symbols('Nx Ny Nz Nn', integer=True)
i = Idx('i', Nx)
j = Idx('j', Ny)
k = Idx('k', Nz)
n = Idx('n', Nn)
dx, dy, dz = symbols('dx dy dz')

directions = ['x','y','z']

def shift(E, idx_shift):
    # This function takes a generic Sympy expression and
    # returns a new Sympy expression where every Sympy Indexed
    # object in E has been shifted by idx_shift.
    # - idx_shift should be of length D, the dimension of E
    
    def shift_indexed(S, idx_shift):
        # This function returns a new IndexedBase object with shifted indices
        # - S should be a Sympy Indexed object
        # - idx_shift should be a tuple or list of index offsets to apply
        # - idx_shift should be of length D, the dimension of S
        base = S.base
        indices = [si + di for si, di in zip(S.indices, idx_shift)]
        return base[indices]

    return E.replace(lambda expr: type(expr) == Indexed, lambda expr: shift_indexed(expr, idx_shift))

def Dc(E, direction):
    assert(direction == 'x' or direction == 'y' or direction == 'z')
    if direction == 'x':
        shift_hi = (1, 0, 0, 0)
        shift_lo = (-1, 0, 0, 0)
        delta = dx
    elif direction == 'y':
        shift_hi = (0, 1, 0, 0)
        shift_lo = (0, -1, 0, 0)
        delta = dy
    elif direction == 'z':
        shift_hi = (0, 0, 1, 0)
        shift_lo = (0, 0, -1, 0)
        delta = dz
    return (shift(E, shift_hi) - shift(E, shift_lo))/(2 * delta)

def Dc2(E, direction):
    assert(direction == 'x' or direction == 'y' or direction == 'z')
    if direction == 'x':
        shift_hi = (1, 0, 0, 0)
        shift_lo = (-1, 0, 0, 0)
        delta = dx*dx
    elif direction == 'y':
        shift_hi = (0, 1, 0, 0)
        shift_lo = (0, -1, 0, 0)
        delta = dy*dy
    elif direction == 'z':
        shift_hi = (0, 0, 1, 0)
        shift_lo = (0, 0, -1, 0)
        delta = dz*dz
    return (shift(E, shift_hi) - 2*E + shift(E, shift_lo))/(delta)

def DcTen(E, direction):
    retE = ixp.zerorank1(len(E))
    for itr in range(len(E)):
        retE[itr] = Dc(E[itr], direction)
    return retE

def Dc2Ten(E, direction):
    retE = ixp.zerorank1(len(E))
    for itr in range(len(E)):
        retE[itr] = Dc2(E[itr], direction)
    return retE

def grad(phi):
    retGradPhi = ixp.zerorank1(len(directions))
    for itr in range(len(directions)):
        retGradPhi[itr] = Dc(phi, directions[itr])
    return retGradPhi
    
def div(E):
    div = 0
    for itr in range(len(E)):
        div += Dc(E[itr],directions[itr])
    return div

def LapTen(E):
    retLapE = ixp.zerorank1()
    for itr in range(len(directions)):
        retLapE += np.array(Dc2Ten(E,directions[itr]))
    return retLapE

def AMReXcode(expr, varnames, declarevar = False, varnew = ""):
    str_expr = str(expr)
    for name in varnames:
        str_expr = str_expr.replace(name,name+"_old_fab")
    str_expr = str_expr.replace("[","(").replace("]",")")
    str_expr = str_expr.replace("dx","dx[0]")
    str_expr = str_expr.replace("dy","dx[1]")
    str_expr = str_expr.replace("dz","dx[2]")
    str_expr = str_expr.replace("dx[0]**2","(dx[0]*dx[0])")
    str_expr = str_expr.replace("dx[1]**2","(dx[1]*dx[1])")
    str_expr = str_expr.replace("dx[2]**2","(dx[2]*dx[2])")
    str_expr = str_expr+";"
    
    if declarevar == True:
        str_expr = varnew+"_new_fab(i, j, k, n) = " + str_expr
        
    return str_expr
    
    
    
    
    
    
    