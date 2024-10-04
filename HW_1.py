import numpy as np

# Initialize your x array. append x0
max_iterations = 100
# Initial guess
x_n = -1.6
tolerance = 10**(-6)
x_vals = [x_n]

def f(x):
    return x * np.sin(3 * x) - np.exp(x)

def df(x):
    return np.sin(3*x) + 3*x*np.cos(3 * x) - np.exp(x)

# Newton-Raphson method
for i in range(max_iterations):
    #compute f(x), f'(x), and x_n+1
    f_x_n = f(x_vals[i])
    df_x_n = df(x_vals[i])
    new_x = x_vals[i] - (f_x_n / df_x_n)
    x_vals.append(new_x)

    if abs(f(x_vals[i])) < tolerance:
        break

x_low = -0.7
x_high = -0.4
midpoints = []
# Bisection method
for j in range(max_iterations):
    #compute
    x_mid = (x_low + x_high) / 2
    if f(x_mid)*f(x_low) < 0:
        x_high = x_mid
    elif f(x_mid)*f(x_high) < 0:
        x_low = x_mid
    else:
        print("Error")
        break
    midpoints.append(x_mid)
    if abs(f(x_mid)) < tolerance:
        break
A1 = x_vals
A2 = midpoints
A3 = [len(x_vals)-1, len(midpoints)]
print(A1)
print(A2)
print(A3)


A = np.array([[1, 2], [-1, 1]])
B = np.array([[2, 0], [0, 2]])
C = np.array([[2, 0, -3], [0, 0, -1]])
D = np.array([[1, 2], [2, 3], [-1, 0]])
x = np.array([[1], [0]])
y = np.array([[0], [1]])
z = np.array([[1], [2], [-1]])


def mult(mat1, mat2):
    return np.dot(mat1,mat2)


A4 = A + B
temp = (3*x - 4*y)
A5 = temp.flatten()
temp = mult(A, x)
A6 = temp.flatten()
temp = mult(B, (x - y))
A7 = temp.flatten()
temp = mult(D, x)
A8 = temp.flatten()
temp = mult(D, y) + z
A9 = temp.flatten()
A10 = mult(A, B)
A11 = mult(B, C)
A12 = mult(C, D)