import numpy as np
from scipy.optimize import line_search

def minimize(X, f, length, *args):
    RHO = 0.01
    SIG = 0.5
    INT = 0.1
    EXT = 3.0
    MAX = 20
    RATIO = 100
    
    if len(length) == 2:
        red = length[1]
        length = length[0]
    else:
        red = 1

    i = 0
    ls_failed = 0
    fX = []
    
    f1, df1 = f(X, *args)
    i = i + (length < 0)
    s = -df1
    d1 = -np.dot(s, s)
    z1 = red / (1 - d1)
    
    while i < abs(length):
        i = i + (length > 0)
        
        X0 = X.copy()
        f0 = f1
        df0 = df1
        X = X + z1 * s
        f2, df2 = f(X, *args)
        i = i + (length < 0)
        d2 = np.dot(df2, s)
        f3 = f1
        d3 = d1
        z3 = -z1
        
        if length > 0:
            M = MAX
        else:
            M = min(MAX, -length - i)
        
        success = 0
        limit = -1
        
        while True:
            while ((f2 > f1 + z1 * RHO * d1) or (d2 > -SIG * d1)) and (M > 0):
                limit = z1
                if f2 > f1:
                    z2 = z3 - (0.5 * d3 * z3 * z3) / (d3 * z3 + f2 - f3)
                else:
                    A = 6 * (f2 - f3) / z3 + 3 * (d2 + d3)
                    B = 3 * (f3 - f2) - z3 * (d3 + 2 * d2)
                    z2 = (np.sqrt(B * B - A * d2 * z3 * z3) - B) / A
                
                if np.isnan(z2) or np.isinf(z2):
                    z2 = z3 / 2
                
                z2 = max(min(z2, INT * z3), (1 - INT) * z3)
                z1 = z1 + z2
                X = X + z2 * s
                f2, df2 = f(X, *args)
                M = M - 1
                i = i + (length < 0)
                d2 = np.dot(df2, s)
                z3 = z3 - z2
            
            if f2 > f1 + z1 * RHO * d1 or d2 > -SIG * d1:
                break
            elif d2 > SIG * d1:
                success = 1
                break
            elif M == 0:
                break
            
            A = 6 * (f2 - f3) / z3 + 3 * (d2 + d3)
            B = 3 * (f3 - f2) - z3 * (d3 + 2 * d2)
            z2 = -d2 * z3 * z3 / (B + np.sqrt(B * B - A * d2 * z3 * z3))
            
            if not np.isreal(z2) or np.isnan(z2) or np.isinf(z2) or z2 < 0:
                if limit < -0.5:
                    z2 = z1 * (EXT - 1)
                else:
                    z2 = (limit - z1) / 2
            elif (limit > -0.5) and (z2 + z1 > limit):
                z2 = (limit - z1) / 2
            elif (limit < -0.5) and (z2 + z1 > z1 * EXT):
                z2 = z1 * (EXT - 1.0)
            elif z2 < -z3 * INT:
                z2 = -z3 * INT
            elif (limit > -0.5) and (z2 < (limit - z1) * (1.0 - INT)):
                z2 = (limit - z1) * (1.0 - INT)
            
            f3 = f2
            d3 = d2
            z3 = -z2
            z1 = z1 + z2
            X = X + z2 * s
            f2, df2 = f(X, *args)
            M = M - 1
            i = i + (length < 0)
            d2 = np.dot(df2, s)
        
        if success:
            f1 = f2
            fX.append(f1)
            print(f'Iteration {i:6d}: Value {f1:4.6e}')
            s = (np.dot(df2, df2) - np.dot(df1, df2)) / np.dot(df1, df1) * s - df2
            tmp = df1
            df1 = df2
            df2 = tmp
            d2 = np.dot(df1, s)
            if d2 > 0:
                s = -df1
                d2 = -np.dot(s, s)
            z1 = z1 * min(RATIO, d1 / (d2 - np.finfo(float).tiny))
            d1 = d2
            ls_failed = 0
        else:
            X = X0
            f1 = f0
            df1 = df0
            if ls_failed or i > abs(length):
                break
            tmp = df1
            df1 = df2
            df2 = tmp
            s = -df1
            d1 = -np.dot(s, s)
            z1 = 1 / (1 - d1)
            ls_failed = 1
    
    return X, fX, i