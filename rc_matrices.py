#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 22:54:12 2020

@author: brian
"""

import numpy as np
import networkx as nx
import sys

EPS = sys.float_info.epsilon

def create_sparse(k,n,d,r):
    '''Create a sparse matrix from input R^K to output R^N'''
    win = np.zeros((n,k))
    for i in range(n):
        for j in range(k):
            p = np.random.uniform(EPS,1.0)
            if(p < d):
                win[i,j] = np.random.uniform(-r,r)          
    return win

def create_ErdosRenyi(n,p):
    return (nx.adjacency_matrix(nx.fast_gnp_random_graph(n,p, seed=11))
            .toarray().astype('float64'))

def create_NewmanWattsStrogatz(n,k,p):
    return (nx.adjacency_matrix(nx.newman_watts_strogatz_graph(n,k,p))
            .toarray().astype('float64'))

def create_pLaw(n,m,p):
    return (nx.adjacency_matrix(nx.powerlaw_cluster_graph(n,m,p))
            .toarray().astype('float64'))

def add_weight(g):
    rows, cols = g.shape
    for i in range(rows):
        for j in range(cols):
            if(g[i,j] != 0):
                #pdb.set_trace()
                s = np.random.uniform(-1,1)
                newval = g[i,j] * s
                g[i,j] = newval
    return g

def rescale(A, scaler):
    return np.real((A/np.max(np.abs(np.linalg.eig(A)[0]))*scaler))


def get_Matrix(d, r, scaler, sparsity=None):
    A = np.random.uniform(low=-r,high=r,size=(d,d))
    if(sparsity):
        for i in range(d):
            for j in range(d):
                p = np.random.uniform(0.0,1.0)
                if(p < sparsity):
                    A[i,j] = 0.0   
    else:
        pass
    A = A + A.T - np.diag(A.diagonal())
    eigs = np.linalg.eig(A)[0]
    return (A/eigs[0])*scaler