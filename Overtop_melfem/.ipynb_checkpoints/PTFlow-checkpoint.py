#!/usr/bin/env python
# coding: utf-8

# In[3]:


from __future__ import division
from past.utils import old_div
from proteus import FemTools as ft
from proteus import MeshTools as mt
from scipy import integrate
from scipy.sparse import csr_matrix,linalg
import numpy as np

class Potential_field(object):

    def __init__(self,
                 grav,
                 h,
                 n,
                 ne,
                 ln,
                 x,
                 phi):
        self.grav = grav
        self.h = h
        self.n = n
        self.ne = ne
        self.ln = ln
        self.x = x
        self.phi = phi

    def loc_divN(self): # N is dN/dxi
        N = np.zeros((2,3))
        N1 = np.zeros((2,1))
        N2 = np.zeros((2,1))
        N3 = np.zeros((2,1))
        N1[0][0] = 1.
        N2[1][0] = 1.
        N3[0][0] = -1.
        N3[1][0] = -1.
        N[0][0] = 1.
        N[1][1] = 1.
        N[0][2] = -1.
        N[1][2] = -1
        return N1,N2,N3,N

    def loc_NN(self):
        # NN is local integral of Ni*Nj
        NN = np.zeros((3,3))
        f = lambda y, x: x**2
        s = integrate.dblquad(f, 0, 1, lambda x: 0, lambda x: 1-x)
        NN[0][0] = s[0]
        f = lambda y, x: x*y
        s = integrate.dblquad(f, 0, 1, lambda x: 0, lambda x: 1-x)
        NN[0][1] = s[0]
        f = lambda y, x: x*(1-x-y)
        s = integrate.dblquad(f, 0, 1, lambda x: 0, lambda x: 1-x)
        NN[0][2] = s[0]
        f = lambda y, x: y**2
        s = integrate.dblquad(f, 0, 1, lambda x: 0, lambda x: 1-x)
        NN[1][1] = s[0]
        f = lambda y, x: y*(1-x-y)
        s = integrate.dblquad(f, 0, 1, lambda x: 0, lambda x: 1-x)
        NN[1][2] = s[0]
        f = lambda y, x: (1-x-y)**2
        s = integrate.dblquad(f, 0, 1, lambda x: 0, lambda x: 1-x)
        NN[2][2] = s[0]
        NN[1][0] = NN[0][1]
        NN[2][0] = NN[0][2]
        NN[2][1] = NN[1][2]

        # NS is local integral of Ni
        NS = np.zeros(3)
        NS[0] = 1./6.
        NS[1] = 1./6.
        NS[2] = 1./6.
        return NN,NS

    def jacobian(self,N,x):
        jcb_tr = np.dot(N,x)
        invJ = np.linalg.inv(jcb_tr)
        detJ = np.linalg.det(jcb_tr)
        return invJ,detJ

    def local_d(self,N,NS,invJ):
        d1 = np.zeros((3,3))
        d2 = np.zeros((3,3))
        Nxy = np.dot(invJ,N)
        for i in range(3):
            for j in range(3):
                d1[i][j] = Nxy[0][j]*NS[i]
                d2[i][j] = Nxy[1][j]*NS[i]
        return d1,d2

    def v_Galerkin(self): #(n,ne,ln,x,phi):
        n = self.n
        ne = self.ne
        ln = self.ln 
        x = self.x 
        phi = self.phi 
        
        # build the derivative of shape function
        [N1,N2,N3,N] = self.loc_divN()

        # for nodal velocity by Galerkin method
        [NN,NS] = self.loc_NN()
    
        unod = np.zeros((n,2))
        C = np.zeros((n,n))
        D1 = np.zeros((n,n))
        D2 = np.zeros((n,n))
        for i in range(ne):

            # get inverse of JT and determinant of J
            [invJ,detJ] = self.jacobian(N,x[ln[i][:]][:])

            # get local matrices of an element
            [d1,d2] = self.local_d(N,NS,invJ)

            # assemble local matrices
            for j in range(3):
                for k in range(3):
                    C[ln[i][j]][ln[i][k]] += NN[j][k]*detJ
                    D1[ln[i][j]][ln[i][k]] += d1[j][k]*detJ
                    D2[ln[i][j]][ln[i][k]] += d2[j][k]*detJ
        # solve u
        b = np.matmul(D1,phi)
        unod[:,0] = np.linalg.solve(C,b)
        # solve v
        b = np.matmul(D2,phi)
        unod[:,1] = np.linalg.solve(C,b)
        return unod


class Bvp_solver(object):


class Time_marching(object):

    def __init__(self,
                 dt,
                 grav,
                 h,
                 n,
                 ne,
                 ln,
                 x,
                 phi):
        self.dt = dt
        self.grav = grav
        self.h = h
        self.n = n
        self.ne = ne
        self.ln = ln
        self.x = x
        self.phi = phi
        
    def get_free_surface_phit(self,flag,grav,h,n,x,u):
        grav = self.grav
        h = self.h
        n = self.n
        x = self.x
        u = self.u
        if flag in [physics.domain.boundaryTags['top']]:
            return -0.5*(u[0]**2+u[1]**2)-grav*(x[1]-h)
        
    def get_free_surface_dpdt(self,flag,grav,h,n,x,u):
        grav = self.grav
        h = self.h
        n = self.n
        x = self.x
        u = self.u
        if flag in [physics.domain.boundaryTags['top']]:
            return 0.5*(u[0]**2+u[1]**2)-grav*(x[1]-h)
            
    def forward_Euler(self):
        dt = self.dt
        x = self.x
        u = self.u
        phi = self.phi
        dpdt = self.dpdt
        new_x = np.zero(2)
        new_x[0] = x[0]+dt*u[0]
        new_x[1] = x[1]+dt*u[1]
        new_phi = phi+dt*dpdt
        return new_x,new_phi
    
#     def rk4_one_of_four(self):
#         """ this subroutine is one of the four steps of the rk4 method"""
    



def rk4(smth_fs,mtd_mesh,mtd_unod,mtd_remesh,mov_wall,dt,vel10,vel1n,vel30,vel3n,grav,h,ne,nx,ny,n,ln,ndir,idir,nfs,ifs,ngp,xg,wg,xc,x,phi):
    # define acc for intermediate step, acc0, acci, accn
    #acci = 0.5*(acc0+accn)
    vel1i = 0.5*(vel10+vel1n)
    vel3i = 0.5*(vel30+vel3n)
    
    
    # ki = nodal velocity or dpdt, get k1
    [u1,dpdt1] = bvp_solver(mtd_mesh,mtd_unod,mtd_remesh,mov_wall,vel10,vel30,grav,h,ne,nx,ny,n,ln,ndir,idir,nfs,ifs,ngp,xg,wg,xc,x,phi)
    [x1,phi1] = remesh_EL(mtd_mesh,0.5*dt,nx,ny,n,ifs,x,u1,phi,dpdt1)
    
    # ki = nodal velocity or dpdt, get k2
    [u2,dpdt2] = bvp_solver(mtd_mesh,mtd_unod,mtd_remesh,mov_wall,vel1i,vel3i,grav,h,ne,nx,ny,n,ln,ndir,idir,nfs,ifs,ngp,xg,wg,xc,x1,phi1)    
    [x2,phi2] = remesh_EL(mtd_mesh,0.5*dt,nx,ny,n,ifs,x,u2,phi,dpdt2)
    
    # ki = nodal velocity or dpdt, get k3
    [u3,dpdt3] = bvp_solver(mtd_mesh,mtd_unod,mtd_remesh,mov_wall,vel1i,vel3i,grav,h,ne,nx,ny,n,ln,ndir,idir,nfs,ifs,ngp,xg,wg,xc,x2,phi2)
    [x3,phi3] = remesh_EL(mtd_mesh,dt,nx,ny,n,ifs,x,u3,phi,dpdt3)
    
    # ki = nodal velocity or dpdt, get k4
    [u4,dpdt4] = bvp_solver(mtd_mesh,mtd_unod,mtd_remesh,mov_wall,vel1n,vel3n,grav,h,ne,nx,ny,n,ln,ndir,idir,nfs,ifs,ngp,xg,wg,xc,x3,phi3)    
     
    
    # define ki for x and phi on the free surface
    k1x = []; k1y = []; k1p = []; k2x = []; k2y = []; k2p = []
    k3x = []; k3y = []; k3p = []; k4x = []; k4y = []; k4p = []
    for i in range(nfs):
        k1x.append(u1[ifs[i],0])
        k1y.append(u1[ifs[i],1])
        k1p.append(dpdt1[ifs[i]])
        k2x.append(u2[ifs[i],0])
        k2y.append(u2[ifs[i],1])
        k2p.append(dpdt2[ifs[i]])        
        k3x.append(u3[ifs[i],0])
        k3y.append(u3[ifs[i],1])
        k3p.append(dpdt3[ifs[i]])
        k4x.append(u4[ifs[i],0])
        k4y.append(u4[ifs[i],1])
        k4p.append(dpdt4[ifs[i]])  


    [new_x,new_phi] = remesh_rk4(smth_fs,mtd_mesh,dt,nx,ny,n,nfs,ifs,k1x,k2x,k3x,k4x,k1y,k2y,k3y,k4y,k1p,k2p,k3p,k4p,x,phi)
    return new_x,new_phi



