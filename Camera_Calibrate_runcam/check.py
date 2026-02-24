import math

def integrate_inv_z(x, y, sign='+'):
    """
    Computes ∫[0 -> sqrt(x^2 + y^2)] (1 / z(r)) dr
    where z(r) = -0.1068*r*cos(theta) ± 0.0057*r*sin(theta) + 599.10
    and theta = atan(y/x)
    """
    R = math.sqrt(x**2 + y**2)
    if R == 0:
        return 0.0  # avoid divide by zero

    # Compute theta components
    cos_theta = x / R
    sin_theta = y / R


    a = -0.1068 * cos_theta - 0.0057 * sin_theta

    b = 599.10



    # Compute integral
    integral = (1 / a) * math.log((a * R + b) / b)
    return integral

import numpy as np
n =10
x = np.linspace(1482,2040,n)

z = -0.1068*x -0.00574*686 +599
dr = (2040-1482)/n
print(np.sum((1/z)*dr) * 39.)


# # Example usage
# if __name__ == "__main__":
#     x_val, y_val = 355,0
#     print("Integral (+):", integrate_inv_z(x_val, y_val)*39.37)
