from numpy import sqrt
x=1000000.
s_numpy = sqrt(x)
s=1.
tol = 1.0E-7

for k in range(100):
    s_old = s
    s = 0.5*(s + x/s)
    print(f"At the {k}-th iteration, the value of s is {s}")
    if abs(s-s_old) < tol:
        break

print(f"square root of {x} is {s}")
