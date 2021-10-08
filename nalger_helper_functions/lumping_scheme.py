#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration scheme for P2 triangles and Quadrilaterals
used for building lumped mass matrices

@author: Jeremy Bleyer, Ecole des Ponts ParisTech,
Laboratoire Navier (ENPC,IFSTTAR,CNRS UMR 8205)
@email: jeremy.bleyer@enpc.f
"""
from numpy import array, kron
from FIAT.reference_element import UFCTriangle, UFCTetrahedron
from FIAT.quadrature import QuadratureRule
from FIAT.quadrature_schemes import create_quadrature
from ffc.analysis import _autoselect_quadrature_rule


def create_quadrature_monkey_patched(ref_el, degree, scheme="default"):
    """Monkey patched FIAT.quadrature_schemes.create_quadrature()
    for implementing lumped scheme"""
    # Our "special" scheme
    if scheme == "lumped":
        if isinstance(ref_el, UFCTriangle) and degree == 2:
            x = array([[0.0, 0.0],
                       [1.0, 0.0],
                       [0.0, 1.0],
                       [0.5, 0.0],
                       [0.5, 0.5],
                       [0.0, 0.5]])
            w = array([1/12.,]*6)
            return QuadratureRule(ref_el, x, w)
        elif isinstance(ref_el, UFCTetrahedron) and degree == 2:
            x = array([[0.0, 0.0, 0.0],
                       [1.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0],
                       [0.5, 0.0, 0.0],
                       [0.5, 0.5, 0.0],
                       [0.0, 0.5, 0.0],
                       [0.5, 0.0, 0.5],
                       [0.5, 0.5, 0.5],
                       [0.0, 0.5, 0.5],
                       [0.0, 0.0, 1.0]])
            w = array([1/60.,]*10)
            return QuadratureRule(ref_el, x, w)
        raise NotImplementedError("Scheme {} of degree {} on {} not implemented"
            .format(scheme, degree, ref_el))

    # Fallback to FIAT's normal operation
    return create_quadrature(ref_el, degree, scheme=scheme)

def _autoselect_quadrature_rule_monkey_patched(*args, **kwargs):
    """Monkey patched ffc.analysis._autoselect_quadrature_rule()
    preventing FFC to complain about non-existing quad scheme"""
    try:
        return _autoselect_quadrature_rule(*args, **kwargs)
    except Exception:
        integral_metadata = args[0]
        qr = integral_metadata["quadrature_rule"]
        return qr

# Monkey patch FIAT quadrature scheme generator
import FIAT
FIAT.create_quadrature = create_quadrature_monkey_patched

# Monkey patch FFC scheme autoselection mechanism
import ffc.analysis
ffc.analysis._autoselect_quadrature_rule = _autoselect_quadrature_rule_monkey_patched
