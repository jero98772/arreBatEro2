import pulp as pl

grado = pl.LpProblem("Grado valentina", pl.LpMaximize)

# variables de decicion
# Cdt Bi anuales
w1 = pl.LpVariable("w1", lowBound=0, cat="Continues")
w2 = pl.LpVariable("w2", lowBound=0, cat="Continues")
w3 = pl.LpVariable("w3", lowBound=0, cat="Continues")
w4 = pl.LpVariable("w4", lowBound=0, cat="Continues")
# Cdt tria unales
x1 = pl.LpVariable("x1", lowBound=0, cat="Continues")
x2 = pl.LpVariable("x2", lowBound=0, cat="Continues")
x3 = pl.LpVariable("x3", lowBound=0, cat="Continues")
# Fondo semilla
y2 = pl.LpVariable("y2", lowBound=0, cat="Continues")

z5 = pl.LpVariable("z5", lowBound=0, cat="Continues")

r1 = pl.LpVariable("r1", lowBound=0, cat="Continues")
r2 = pl.LpVariable("r2", lowBound=0, cat="Continues")
r3 = pl.LpVariable("r3", lowBound=0, cat="Continues")
r4 = pl.LpVariable("r4", lowBound=0, cat="Continues")
r5 = pl.LpVariable("r5", lowBound=0, cat="Continues")

grado += r5 + 1.35 * w4 + 1.55 * x3 + 1.75 * y2 + 1.2 * z5
grado += 60 == w1 + x1 + r1
grado += r1 == w2 + x2 + y2 + r2
grado += 1.35 * w1 + r2 == w3 + x3 + r3
grado += 1.55 * x1 + 1.35 * w2 + r3 == w4 + r4
grado += 1.55 * x2 + 1.35 * w3 + r4 == z5 + r5

print(grado.solve())
