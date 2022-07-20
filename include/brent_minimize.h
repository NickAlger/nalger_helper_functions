#pragma once

// This is based on code in scipy.optimize, but modified for c++
//
// https://www.scipy.org/scipylib/license.html
// SciPy license
// Copyright © 2001, 2002 Enthought, Inc.
// All rights reserved.
//
// Copyright © 2003-2019 SciPy Developers.
// All rights reserved.
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
//
// Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
//
// Neither the name of Enthought nor the names of the SciPy Developers may be used to endorse or promote products derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <iostream>
#include <list>
#include <vector>
#include <queue>

#include <math.h>
#include <cmath>
#include <Eigen/Dense>

namespace BRENT {

std::tuple<double, double, int, int> brent_minimize( const std::function<double(double)> &func,
                                                     const double xmin, const double xmax,
                                                     const double tol, const int maxiter )
{
    double cg = 0.3819660;
    double mintol = 1e-11;

    double xa = xmin;
    double xc = xmax;
    double xb = (xa + xc) / 2.0;

    double funcalls = 0;

    double x = xb;
    double w = xb;
    double v = xb;
    double u;

    double fw = func(x); funcalls += 1;
    double fv = fw;
    double fx = fw;

    double a;
    double b;
    if (xa < xc)
    {
        a = xa;
        b = xc;
    }
    else
    {
        a = xc;
        b = xa;
    }

    double deltax = 0.0;
    int iter = 0;
    while (iter < maxiter)
    {
        double tol1 = tol * std::abs(x) + mintol;
        double tol2 = 2.0 * tol1;
        double xmid = 0.5 * (a + b);
        // check for convergence
        if ( std::abs(x - xmid) < (tol2 - 0.5 * (b - a)) )
        {
            break;
        }
        // XXX In the first iteration, rat is only bound in the true case
        // of this conditional. This used to cause an UnboundLocalError
        // (gh-4140). It should be set before the if (but to what?).
        double rat;
        if ( std::abs(deltax) <= tol1 )
        {
            if ( x >= xmid )
            {
                deltax = a - x;  // do a golden section step
            }
            else
            {
                deltax = b - x;
            }
            rat = cg * deltax;
        }
        else  // do a parabolic step
        {
            double tmp1 = (x - w) * (fx - fv);
            double tmp2 = (x - v) * (fx - fw);
            double p = (x - v) * tmp2 - (x - w) * tmp1;
            tmp2 = 2.0 * (tmp2 - tmp1);
            if (tmp2 > 0.0)
            {
                p = -p;
            }
            tmp2 = std::abs(tmp2);
            double dx_temp = deltax;
            deltax = rat;
            // check parabolic fit
            if ( (p > tmp2 * (a - x)) &&
                 (p < tmp2 * (b - x)) &&
                 (std::abs(p) < std::abs(0.5 * tmp2 * dx_temp)) )
            {
                rat = p * 1.0 / tmp2;  // if parabolic step is useful.
                double u = x + rat;
                if ( (u - a) < tol2 || (b - u) < tol2 )
                {
                    if ( xmid - x >= 0 )
                    {
                        rat = tol1;
                    }
                    else
                    {
                        rat = -tol1;
                    }
                }
            }
            else
            {
                if ( x >= xmid )
                {
                    deltax = a - x;  // if it's not do a golden section step
                }
                else
                {
                    deltax = b - x;
                }
                rat = cg * deltax;
            }
        }

        if ( std::abs(rat) < tol1 )  // update by at least tol1
        {
            if ( rat >= 0 )
            {
                u = x + tol1;
            }
            else
            {
                u = x - tol1;
            }
        }
        else
        {
            u = x + rat;
        }
        double fu = func(u); funcalls += 1;  // calculate new output value

        if ( fu > fx )  // if it's bigger than current
        {
            if ( u < x )
            {
                a = u;
            }
            else
            {
                b = u;
            }
            if ( (fu <= fw) || (w == x) )
            {
                v = w;
                w = u;
                fv = fw;
                fw = fu;
            }
            else if ( (fu <= fv) || (v == x) || (v == w) )
            {
                v = u;
                fv = fu;
            }
        }
        else
        {
            if ( u >= x )
            {
                a = x;
            }
            else
            {
                b = x;
            }
            v = w;
            w = x;
            x = u;
            fv = fw;
            fw = fx;
            fx = fu;
        }

        iter += 1;
    }

    return std::make_tuple(x, fx, iter, funcalls);
}

} // end namespace BRENT