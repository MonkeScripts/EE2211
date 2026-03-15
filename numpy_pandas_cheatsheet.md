# EE2211 Cheat Sheet (Tutorials 2–6)

Quick reference for NumPy, Pandas, Probability, and ML patterns.

---

## Part 1 — NumPy

### 1.1 Imports

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import inv, matrix_rank, det
```

---

### 1.2 Array Creation

| Function | Description |
|---|---|
| `np.array([1, 2, 3])` | 1D array from list |
| `np.array([[1,2],[3,4]])` | 2D array (matrix) |
| `np.zeros((m, n))` | m×n matrix of zeros |
| `np.ones((m, n))` | m×n matrix of ones |
| `np.eye(n)` | n×n identity matrix |
| `np.linspace(a, b, n)` | n evenly spaced points in [a, b] |
| `np.arange(start, stop, step)` | integer range array |
| `np.random.randn(m, n)` | standard normal (Gaussian) |

```python
X = np.array([[1, 2, 3],
              [4, 5, 6]])       # shape (2, 3)
Z = np.zeros((3, 1))           # 3×1 column of zeros
I = np.eye(3)                  # 3×3 identity
x = np.linspace(0, 10, 100)   # 100 points from 0 to 10
```

---

### 1.3 Array Properties

| Attribute | Description |
|---|---|
| `X.shape` | tuple of dimensions, e.g. (3, 4) |
| `X.ndim` | number of dimensions |
| `X.dtype` | data type (float64, int32, …) |
| `X.size` | total number of elements |
| `len(X)` | number of rows (axis 0) |

```python
N, M = X.shape    # N = rows (samples), M = cols (features)
```

---

### 1.4 Indexing & Slicing

| Expression | Returns |
|---|---|
| `X[i, j]` | element at row i, col j |
| `X[i, :]` | row i (all columns) |
| `X[:, j]` | column j (all rows) |
| `X[1:3, :]` | rows 1 and 2 |
| `X[-1, :]` | last row |
| `X[X > 2]` | elements satisfying condition (1D) |
| `np.where(X > 2, 1, 0)` | element-wise conditional (same shape) |

```python
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

A[0, :]        # [1, 2, 3]  — first row
A[:, 2]        # [3, 6, 9]  — third column
A[0:2, :]      # rows 0-1
A[A > 5]       # [6, 7, 8, 9]
np.where(A > 5, 1, 0)   # [[0,0,0],[0,0,1],[1,1,1]]
```

---

### 1.5 Reshaping & Stacking

| Function | Description |
|---|---|
| `X.reshape(m, n)` | change shape (total elements must match) |
| `X.reshape(-1, 1)` | flatten to column vector |
| `X.flatten()` | collapse to 1D array |
| `X.T` | transpose |
| `np.hstack([A, B])` | join arrays column-wise |
| `np.vstack([A, B])` | join arrays row-wise |
| `np.append(X, y, axis=1)` | append column(s) to matrix |

```python
x = np.array([1, 2, 3, 4, 5, 6])
x.reshape(2, 3)       # [[1,2,3],[4,5,6]]
x.reshape(-1, 1)      # column vector shape (6,1)

A = np.ones((3, 2))
B = np.zeros((3, 1))
np.hstack([A, B])     # shape (3, 3)
```

---

### 1.6 Element-wise Math & Broadcasting

| Operation | Description |
|---|---|
| `A + B`, `A - B` | element-wise add/subtract |
| `A * B`, `A / B` | element-wise multiply/divide |
| `np.square(X)` | element-wise x² |
| `np.sqrt(X)` | element-wise √x |
| `np.exp(X)` | element-wise eˣ |
| `np.log(X)` | element-wise ln(x) |
| `np.abs(X)` | element-wise |x| |
| `np.power(X, n)` | element-wise xⁿ |

---

### 1.7 Matrix Multiplication

| Operation | Description |
|---|---|
| `A @ B` | matrix multiply (preferred) |
| `np.dot(A, B)` | equivalent to `@` |
| `A * B` | element-wise (NOT matrix multiply) |

```python
A = np.array([[1, 2], [3, 4]])   # 2×2
B = np.array([[5], [6]])          # 2×1

A @ B       # [[17], [39]]  — matrix product
A * A       # [[1,4],[9,16]] — element-wise
```

---

### 1.8 Linear Algebra (`np.linalg`)

| Function | Description |
|---|---|
| `inv(A)` | inverse of square matrix |
| `det(A)` | determinant |
| `matrix_rank(A)` | rank of matrix |
| `np.linalg.solve(A, b)` | exact solution to Ax = b |
| `np.linalg.norm(x)` | Euclidean norm |

```python
from numpy.linalg import inv, matrix_rank, det

A = np.array([[2, 1], [5, 3]])
b = np.array([[8], [21]])

det(A)                      # 1.0
inv(A)                      # inverse
matrix_rank(A)              # 2  (full rank)
np.linalg.solve(A, b)       # exact solution

# Check augmented matrix for system consistency
X_ = np.append(X, y, axis=1)   # augmented [X | y]
print("rank(X) =",  matrix_rank(X))
print("rank(X~) =", matrix_rank(X_))
# If rank(X) == rank(X~) → consistent; else no solution
```

---

### 1.9 Statistics

| Function | Description |
|---|---|
| `np.mean(X, axis=0)` | column-wise mean |
| `np.mean(X, axis=1)` | row-wise mean |
| `np.std(X, axis=0)` | column-wise standard deviation |
| `np.var(X, axis=0)` | column-wise variance |
| `np.sum(X)` | sum of all elements |
| `np.min(X)` / `np.max(X)` | global min / max |
| `np.argmin(X)` / `np.argmax(X)` | index of min / max |

---

### 1.10 ML-Specific NumPy Patterns

```python
# Add bias (offset) column — w[0] becomes intercept
N = X.shape[0]
bias = np.ones((N, 1))
X_b = np.hstack((bias, X))          # shape (N, M+1)

# Standardize features (z-score)
mu    = np.mean(X, axis=0)
sigma = np.std(X,  axis=0)
X_norm = (X - mu) / sigma

# Train / test split by slicing
split = int(0.8 * N)
X_train, X_test = X_b[:split], X_b[split:]
y_train, y_test = y[:split],   y[split:]

# MSE
mse = np.mean((y_pred - y_test) ** 2)
```

---

## Part 2 — Pandas

### 2.1 Loading & Inspecting

| Call | Description |
|---|---|
| `pd.read_csv("file.csv")` | load CSV into DataFrame |
| `df.head(n)` | first n rows (default 5) |
| `df.tail(n)` | last n rows |
| `df.info()` | dtypes + non-null counts |
| `df.describe()` | summary statistics |
| `df.shape` | (rows, cols) |
| `df.columns` | column names |

```python
df = pd.read_csv("winequality-red.csv")
df.head()
df.describe()
print(df.shape)      # (1599, 12)
```

---

### 2.2 Selecting Data

| Expression | Returns |
|---|---|
| `df['col']` | single column as Series |
| `df[['col1', 'col2']]` | multiple columns as DataFrame |
| `df.iloc[i, j]` | by integer position |
| `df.iloc[0:5, 1:3]` | row/col slice by position |
| `df.loc[i, 'col']` | by row label and column name |

```python
df['quality']              # Series of quality values
df[['pH', 'quality']]      # DataFrame with 2 columns
df.iloc[0:3, :]            # first 3 rows
df.iloc[:, -1]             # last column
```

---

### 2.3 Filtering (Boolean Indexing)

```python
# Single condition
df[df['quality'] > 6]

# Multiple conditions — wrap each in ()
df[(df['quality'] > 6) & (df['alcohol'] > 10.0)]
df[(df['pH'] < 3.2) | (df['quality'] == 8)]

# Negate
df[~(df['quality'] == 3)]

# .isin()
df[df['quality'].isin([7, 8, 9])]
```

---

### 2.4 Data Cleaning

```python
df.isnull().sum()                      # count NaN per column
df.dropna(inplace=True)                # remove rows with any NaN
df.fillna(df.mean(), inplace=True)     # fill NaN with column mean

# Check zero counts (Tutorial 2 / Pima pattern)
print((df[[1, 2, 3, 4, 5]] == 0).sum())

# Treat zeros as missing
suspect_cols = [1, 2, 3, 4, 5]
df[suspect_cols] = df[suspect_cols].replace(0, np.nan)
df.dropna(inplace=True)
```

---

### 2.5 Column Operations

```python
df.drop('quality', axis=1, inplace=True)              # drop one column
df.drop(['col1', 'col2'], axis=1, inplace=True)       # drop multiple
df.rename(columns={'old': 'new'}, inplace=True)
df['log_alcohol'] = np.log(df['alcohol'])              # new computed column
df.reset_index(drop=True, inplace=True)
```

---

### 2.6 Aggregation

```python
df.groupby('quality').mean()
df['quality'].value_counts()
df['quality'].unique()
df['quality'].nunique()
```

---

### 2.7 Pandas → NumPy (exam pattern)

```python
X = np.array(df.drop('target', axis=1))          # shape (N, M)
y = np.array(df['target']).reshape(-1, 1)         # shape (N, 1)

# Using column numbers (unnamed CSV headers)
X = np.array(df.iloc[:, :-1])
y = np.array(df.iloc[:, -1]).reshape(-1, 1)
```

---

### 2.8 Visualization

```python
# Line plot (Tutorial 2 Q1: education expenditure over years)
plt.plot(df['year'], df['total_expenditure'], label='Expenditure')
plt.xlabel('Year'); plt.ylabel('Expenditure')
plt.title('Education Expenditure'); plt.show()

# Multi-series line plot (Tutorial 2 Q2: bus types over years)
val1 = df.loc[df['type'] == 'Omnibuses'].index
val2 = df.loc[df['type'] == 'Excursion buses'].index
val3 = df.loc[df['type'] == 'Private buses'].index

List1 = df.loc[val1]
List2 = df.loc[val2]
List3 = df.loc[val3]

plt.plot(List1['year'], List1['number'], label='Omnibuses')
plt.plot(List2['year'], List2['number'], label='Excursion buses')
plt.plot(List3['year'], List3['number'], label='Private buses')
plt.xlabel('Year'); plt.ylabel('Number of vehicles')
plt.title('Number of vehicles over the years')
plt.legend(); plt.show()

# Scatter matrix (Tutorial 2 Q3: Iris)
from sklearn.model_selection import train_test_split
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15),
                           marker='o', hist_kwds={'bins': 20})
plt.show()

# Scatter + fitted line
plt.scatter(X, y, color='blue', label='Original Data', marker='o')
plt.plot(X, y_pred, color='red', label='Trained Model')
plt.xlabel('X'); plt.ylabel('y')
plt.legend(); plt.show()
```

---

## Part 3 — Probability (Tutorial 3)

### 3.1 Gaussian / Normal Distribution

Use-case: resistor quality control — check if production batch falls within spec range.

```python
from scipy import stats

# X ~ N(mu, sigma) — Tutorial 3 Q5: resistor within [28Ω, 33Ω], mu=30, sigma=1.8
mu, sigma = 30, 1.8
prob = stats.norm.cdf(33, mu, sigma) - stats.norm.cdf(28, mu, sigma)
print(f'P(28 ≤ X ≤ 33) = {prob:.4f}')

# P(X > a)
prob_gt = 1 - stats.norm.cdf(a, mu, sigma)

# P(X < a)
prob_lt = stats.norm.cdf(a, mu, sigma)
```

---

### 3.2 Discrete Probability (Tutorial 3 Q4)

Use-case: dice rolling — PMF of sum of two dice.

```python
from collections import defaultdict
import numpy as np

# Build sum dictionary: (i,j) -> i+j
d = {(i, j): i + j for i in range(1, 7) for j in range(1, 7)}
print(d[(3, 4)])   # 7

# Invert: group pairs by their sum value
dinv = defaultdict(list)
for k, v in d.items():
    dinv[v].append(k)

# PMF: P(X = x) = count / 36
X_pmf = {x: len(pairs) / 36. for x, pairs in dinv.items()}
print(X_pmf)

# P(half the product of three dice > their sum)
d3 = {(i, j, k): ((i*j*k)/2 > i+j+k)
      for i in range(1, 7) for j in range(1, 7) for k in range(1, 7)}
prob = sum(d3.values()) / 216
print(f'Probability = {prob:.4f}')
```

---

## Part 4 — Core ML Tools

### 4.1 LE Solving Tool (`solveLE`)

Automatically handles even / over / under-determined linear systems.
Source: Tutorial 4–6 notebooks (Ni Qingqing @ NUS ECE).

```python
import numpy as np
from numpy.linalg import inv, matrix_rank

def det_checker(X):
    m, d = X.shape
    if m == d:   return "even"
    elif m > d:  return "over"
    else:        return "under"

def RC_checker(X, y):
    X_aug = np.append(X, y, axis=1)
    rankX  = matrix_rank(X)
    rankX_ = matrix_rank(X_aug)
    d = X.shape[1]
    if rankX == rankX_:
        RC = 1 if rankX == d else 3
    else:
        RC = 2
    return RC, rankX, rankX_

def evenSolver(X, y):
    RC, _, _ = RC_checker(X, y)
    if RC == 1:   return inv(X) @ y, "Unique solution."
    elif RC == 2: return None, "No solution."
    else:         return None, "Infinitely many solutions."

def overSolver(X, y):
    RC, _, _ = RC_checker(X, y)
    if RC == 1:   return inv(X.T @ X) @ X.T @ y, "Unique solution."
    elif RC == 3: return None, "Infinitely many solutions."
    else:         return inv(X.T @ X) @ X.T @ y, "No exact sol, least square approx."

def underSolver(X, y):
    RC, _, _ = RC_checker(X, y)
    if RC == 2:   return None, "No solution."
    else:         return X.T @ inv(X @ X.T) @ y, "No exact sol, least norm approx."

def solveLE(X, y):
    det = det_checker(X)
    if det == "even":   w, ans = evenSolver(X, y)
    elif det == "over": w, ans = overSolver(X, y)
    else:               w, ans = underSolver(X, y)
    print(ans, "\nw =", w)
    return w
```

Diagnose a system before solving:
```python
X_aug = np.append(X, y, axis=1)
print("System type :", det_checker(X))           # even / over / under
print("rank(X)  =", matrix_rank(X))
print("rank(X~) =", matrix_rank(X_aug))          # equal → consistent
w = solveLE(X, y)
print("Verify X@w:", X @ w)                      # should equal (or approx) y
```

---

### 4.2 Polynomial Transformer + Solver (`polyTx`, `solvePR`)

Source: Tutorial 6 notebook.

```python
from sklearn.preprocessing import PolynomialFeatures
from numpy.linalg import inv
import numpy as np

def polyTx(X, order):
    """Polynomial feature matrix with bias column. Shape: (N, order+1)."""
    return PolynomialFeatures(order).fit_transform(X)

def solvePR(P, y, ridge=False, lamb=0.01):
    """Solve polynomial regression. Auto primal (N>M) or dual (N<M)."""
    if ridge:
        if P.shape[0] > P.shape[1]:   # Primal
            w = inv(P.T @ P + lamb * np.eye(P.shape[1])) @ P.T @ y
        else:                          # Dual
            w = P.T @ inv(P @ P.T + lamb * np.eye(P.shape[0])) @ y
    else:
        if P.shape[0] > P.shape[1]:   # Primal
            w = inv(P.T @ P) @ P.T @ y
        else:                          # Dual
            w = P.T @ inv(P @ P.T) @ y
    return w

def solveLE_Ridge(X, y, lamb=0.01):
    """Linear regression with Ridge. X must already include bias column."""
    return solvePR(X, y, ridge=True, lamb=lamb)
```

---

## Part 5 — ML Workflows

### Workflow 1 — System Analysis (Tutorial 4)

Use-case: determine if a linear system Xw = y has a unique / approximate / no solution.

```python
import numpy as np
from numpy.linalg import inv, matrix_rank

# --- Even-determined, unique solution ---
X = np.array([[1, 1], [3, 4]])
y = np.array([[0], [1]])

X_aug = np.append(X, y, axis=1)
print("rank(X) =", matrix_rank(X), "  rank(X~) =", matrix_rank(X_aug))
# rank equal and == ncols → unique solution
w = inv(X) @ y
print("w =", w)
print("Verify X@w == y:\n", X @ w)

# --- Overdetermined (N > M): N equations, M unknowns, N > M ---
# Use LEFT pseudo-inverse:  w* = (X^T X)^{-1} X^T y
X = np.array([[1, 2], [2, 4], [1, -1]])
y = np.array([[0], [0.1], [1]])

w = inv(X.T @ X) @ X.T @ y
print("Least-squares w =", w)
print("X@w (approx y):\n", X @ w)

# --- Underdetermined (N < M): fewer equations than unknowns ---
# Use RIGHT pseudo-inverse: w* = X^T (X X^T)^{-1} y  (minimum-norm solution)
X = np.array([[1, 0, 1, 0], [1, -1, 1, -1], [1, 1, 0, 0]])
y = np.array([[1], [0], [1]])

w = X.T @ inv(X @ X.T) @ y
print("Least-norm w =", w)
print("X@w (should == y):\n", X @ w)
```

---

### Workflow 2 — Linear Regression (Tutorial 5)

Use-case: predict number of books sold from number of registered students.

```python
import numpy as np
import matplotlib.pyplot as plt

X = np.array([[36],[28],[35],[39],[30],[30],[31],[38],[36],[38],[29],[26]])
y = np.array([[31],[29],[34],[35],[29],[30],[30],[38],[34],[33],[29],[26]])

# --- With bias ---
bias = np.ones((X.shape[0], 1))
X_b  = np.hstack((bias, X))         # [1, x]
w    = solveLE(X_b, y)              # w = [w0_bias, w1_slope]

# Predict for new inputs
X_test   = np.array([[30], [5]])
X_test_b = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
y_pred   = X_test_b @ w
print("Predictions:", y_pred)

# --- Without bias ---
w_nb = solveLE(X, y)

# --- Plot both ---
plt.scatter(X, y, color='blue', label='Training Samples', marker='o')
plt.plot(X, X_b @ w,    color='red',   label='With bias')
plt.plot(X, X   @ w_nb, color='green', label='No bias')
plt.xlabel('Students'); plt.ylabel('Books sold'); plt.legend(); plt.show()
```

Multi-output regression (Tutorial 5 Q9) — multiple targets simultaneously:
```python
X = np.array([[3,-1,0],[5,1,2],[9,-1,3],[-6,7,2],[3,-2,0]])
Y = np.array([[1,-1],[-1,0],[1,2],[0,3],[1,-2]])   # two targets

bias = np.ones((X.shape[0], 1))
X_b  = np.hstack((bias, X))
W    = solveLE(X_b, Y)             # W shape (M+1, 2)

x_new   = np.array([[8, 0, 2]])
x_new_b = np.hstack((np.ones((1, 1)), x_new))
y_new   = x_new_b @ W             # shape (1, 2) — two predictions
print("y_new =", y_new)
```

---

### Workflow 3 — Polynomial Regression (Tutorial 6)

Use-case: fit a higher-order curve when data has non-linear trend.

```python
import numpy as np
import matplotlib.pyplot as plt

# Tutorial 6 Q2 dataset
X = np.array([[-10], [-8], [-3], [-1], [2], [8]])
y = np.array([[5],   [5],  [4],  [3],  [2], [2]])

order = 3
P = polyTx(X, order)          # shape (N, 4): [1, x, x², x³]
print("Polynomial matrix P:\n", P)

# Fit (auto primal/dual)
w_poly = solvePR(P, y)
print("Coefficients:", w_poly.ravel())

# Predict on new point
x_test = np.array([[9]])
P_test = polyTx(x_test, order)   # use same order
y_pred = P_test @ w_poly
print("Prediction at x=9:", y_pred)

# Plot
from matplotlib import pyplot as plt
x_plot = np.linspace(-10, 10, 200).reshape(-1, 1)
P_plot  = polyTx(x_plot, order)
y_plot  = P_plot @ w_poly

plt.scatter(X, y, color='yellow', label='Training samples')
plt.plot(x_plot, y_plot, label='Poly model')
plt.scatter(9, y_pred, color='cyan', label='Poly pred')
plt.grid(); plt.legend(); plt.show()
```

⚠ NOTICE: `PolynomialFeatures` column order may differ from hand-calculated version.

---

### Workflow 4 — Ridge Regression (Tutorial 6)

```python
import numpy as np
from numpy.linalg import inv

X = np.array([[-10], [-8], [-3], [-1], [2], [8]])
y = np.array([[5],   [5],  [4],  [3],  [2], [2]])
P = polyTx(X, 3)

# --- Via tool (auto primal/dual) ---
lamb = 0.0001  # Note: 'lambda' is a Python keyword — use 'lamb' or 'lam'
w_ridge = solvePR(P, y, ridge=True, lamb=lamb)
print("Ridge w:", np.around(w_ridge.T, decimals=1))

# --- Manually (primal, N > M) ---
reg_L = lamb * np.eye(P.shape[1])      # λI
w_primal = inv(P.T @ P + reg_L) @ P.T @ y

# --- Manually (dual, N < M) ---
# X = np.array([[1,0,1],[1,-1,1]])     # underdetermined example
# P = polyTx(X, 3)
# reg_L = lamb * np.eye(P.shape[0])
# w_dual = P.T @ inv(P @ P.T + reg_L) @ y
```

---

### Workflow 5 — Data Cleaning & Full Pipeline (Tutorial 2)

Use-case: Pima Indians Diabetes — physiologically impossible zeros = missing values.

```python
import numpy as np
import pandas as pd

df = pd.read_csv("pima-indians-diabetes.csv", header=None)
print(df.describe())

# Step 1: check which columns have suspicious zeros
print((df[[1, 2, 3, 4, 5]] == 0).sum())

# Step 2: replace zeros with NaN
df[[1, 2, 3, 4, 5]] = df[[1, 2, 3, 4, 5]].replace(0, np.nan)
print(df.isnull().sum())
df.dropna(inplace=True)
print(f"Samples after cleaning: {len(df)}")

# Step 3: extract features and target
X = np.array(df.iloc[:, :-1])
y = np.array(df.iloc[:, -1]).reshape(-1, 1)

# Step 4: add bias column
N = X.shape[0]
X_b = np.hstack((np.ones((N, 1)), X))

# Step 5: train / test split (80/20)
split = int(0.8 * N)
X_train, X_test = X_b[:split], X_b[split:]
y_train, y_test = y[:split],   y[split:]

# Step 6: fit and evaluate
w = solveLE(X_train, y_train)
y_pred = X_test @ w
mse = np.mean((y_pred - y_test) ** 2)
print(f"Test MSE: {mse:.4f}")
```

---

### Workflow 6 — Binary Classification (Tutorial 6 Q4)

Use-case: classify samples as +1 or −1 (e.g., COVID-19 positive/negative, spam/not-spam).

```python
import numpy as np

# X already biased manually (first column = 1)
X = np.array([[1, -1], [1, 0], [1, 0.5], [1, 0.3], [1, 0.8]])
y = np.array([[1], [1], [-1], [1], [-1]])

# Train linear regression classifier
w = solveLE(X, y)

# Predict: threshold raw output at 0 using np.sign()
X_test  = np.array([[1, -0.1], [1, 0.4]])
y_raw   = X_test @ w
y_class = np.sign(y_raw)        # → +1 or -1
print("Raw output:     ", y_raw.ravel())
print("Predicted class:", y_class.ravel())
```

---

### Workflow 7 — Multi-Class Classification with OneHotEncoder (Tutorial 6 Q5–Q6)

Use-case: classify iris species (or hand gestures) into 3+ categories.

```python
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

# --- Simple 3-class example (Tutorial 6 Q5) ---
# X already biased; Y contains integer class labels 1/2/3
X = np.array([[1,-1], [1,0], [1,0.5], [1,0.3], [1,0.8]])
Y = np.array([[1], [1], [2], [3], [2]])

encoder = OneHotEncoder(sparse_output=False)
Y_onehot = encoder.fit_transform(Y)     # shape (N, n_classes)
print("One-hot Y:\n", Y_onehot)

W = solveLE(X, Y_onehot)               # weight matrix, shape (M, n_classes)

X_test  = np.array([[1, -0.1], [1, 0.4]])
y_raw   = X_test @ W                   # raw scores per class
y_pred  = encoder.inverse_transform(y_raw)
print("Predicted classes:", y_pred.ravel())

# --- Iris with polynomial features (Tutorial 6 Q6) ---
iris = load_iris()
X, y = iris.data, iris.target.reshape(-1, 1)

# Polynomial transform includes bias — split BEFORE polyTx
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.26, random_state=0)

P_train = polyTx(X_train, order=2)
P_test  = polyTx(X_test,  order=2)     # transform only — do NOT refit

encoder = OneHotEncoder(sparse_output=False)
y_train_onehot = encoder.fit_transform(y_train)

W = solvePR(P_train, y_train_onehot)
y_pred = encoder.inverse_transform(P_test @ W)

accuracy = np.mean(y_pred.reshape(-1, 1) == y_test)
print(f"Poly Classification Accuracy: {accuracy:.2%}")

# --- Linear version for comparison (Tutorial 6 Q6c) ---
bias    = np.ones((X.shape[0], 1))
X_b     = np.hstack((bias, X))
X_train_b, X_test_b, y_train, y_test = train_test_split(
    X_b, y, test_size=0.26, random_state=0)

encoder = OneHotEncoder(sparse_output=False)
y_train_onehot = encoder.fit_transform(y_train)

w_linear = solveLE(X_train_b, y_train_onehot)
y_pred_linear = encoder.inverse_transform(X_test_b @ w_linear)
correct = np.sum(y_pred_linear.reshape(-1, 1) == y_test)
print(f"Linear Classification: {correct} / {len(y_test)} correct")
```

Key rule:
- `encoder.fit_transform(y_train)` — fit on training labels only
- `encoder.inverse_transform(y_pred_raw)` — decode predictions back to class labels
- `polyTx(X_test, order)` uses `.fit_transform` internally, but always use the same order

---

### Workflow 8 — EDA with Scatter Matrix (Tutorial 2 Q3)

Use-case: explore **Iris** feature relationships visually; color each point by class label before training.

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris_dataset = load_iris()

# Split first, then inspect training data
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0)

print('X_train shape:', X_train.shape)
print('X_test  shape:', X_test.shape)
print('Classes:', iris_dataset['target_names'])

# Build DataFrame from training split
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
iris_dataframe.head()
```

```python
# Scatter matrix colored by class label (y_train)
grr = pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15),
                                  marker='o', hist_kwds={'bins': 20})
plt.suptitle('Iris Scatter Matrix (colored by class)')
plt.tight_layout()
plt.show()
```

---

### Workflow 9 — Duplicate Data Comparison (Tutorial 5 Q4)

Use-case: show how **duplicate training points** bias the fitted line; compare with a de-duplicated fit.

```python
import numpy as np
import matplotlib.pyplot as plt

# Original samples (no duplicates)
X = np.array([[36],[28],[35],[39],[30],[30],[31],[38],[36],[38],[29],[26]])
y = np.array([[31],[29],[34],[35],[29],[30],[30],[38],[34],[33],[29],[26]])

# Full dataset: X=26 appears 4 times with y=20 (duplicates inflate their influence)
X_full = np.array([[36],[26],[35],[39],[26],[30],[31],[38],[36],[38],[26],[26]])
y_full = np.array([[31],[20],[34],[35],[20],[30],[30],[38],[34],[33],[20],[20]])

# Purged dataset: duplicates removed
X_purged = np.array([[36],[35],[39],[30],[31],[38],[36],[38],[26]])
y_purged = np.array([[31],[34],[35],[30],[30],[38],[34],[33],[20]])

# Fit both
bias_full   = np.ones((X_full.shape[0], 1))
X_full_b    = np.hstack((bias_full, X_full))
w_full      = solveLE(X_full_b, y_full)

bias_purged = np.ones((X_purged.shape[0], 1))
X_purged_b  = np.hstack((bias_purged, X_purged))
w_purged    = solveLE(X_purged_b, y_purged)

y_pred_full   = X_full_b   @ w_full
y_pred_purged = X_purged_b @ w_purged

# Plot — duplicates pull the red line toward the repeated cluster
plt.scatter(X, y, color='blue', label='Training Samples', marker='o')
plt.plot(X_full,   y_pred_full,   color='red',   label='LR with full set (duplicates)')
plt.plot(X_purged, y_pred_purged, color='green', label='LR with purged set')
plt.xlabel('X (number of students)')
plt.ylabel('y (number of books sold)')
plt.title('Effect of Duplicate Data on Linear Regression')
plt.legend()
plt.show()
```

---

### Workflow 10 — Multi-Feature Polynomial Regression (Tutorial 6 Q3)

Use-case: apply polynomial expansion to **multi-dimensional** input — the resulting system is often underdetermined (N < M), requiring the dual form or ridge regularization.

```python
import numpy as np
from numpy.linalg import inv

# 2D input: N=2 samples, 3 features each
X = np.array([[1, 0, 1],
              [1, -1, 1]])
y = np.array([[0], [1]])

P = polyTx(X, 3)
print('X shape:', X.shape)    # (2, 3)
print('P shape:', P.shape)    # (2, 20) — polynomial expansion creates many features!
print('NOTICE: PolynomialFeatures column order may differ from handwritten version!')
print('System type:', det_checker(P))   # → "under" (N=2 < M=20)

# (c) Dual form — minimum-norm solution (N < M, no ridge)
w_dual = P.T @ inv(P @ P.T) @ y
print('\nw_dual (transposed):', w_dual.T)
print('P @ w_dual (verify == y):', (P @ w_dual).ravel())

# (d) Primal form with ridge — alternative when dual is numerically unstable
lamb   = 0.0001               # 'lambda' is a Python keyword
reg_L  = lamb * np.eye(P.shape[1])
w_ridge = inv(P.T @ P + reg_L) @ P.T @ y
print('\nw_ridge (1dp):', np.around(w_ridge.T, decimals=1))
print('Close to dual?', np.allclose(np.around(w_dual, 1), np.around(w_ridge, 1)))
```

Key insight: with 3 input features at degree 3, `PolynomialFeatures` produces 20 columns — far more than the 2 training samples. The system is underdetermined, so use **dual form** or add ridge to stabilize primal.

---

## Quick Reference Card

| Task | One-liner |
|---|---|
| System type | `det_checker(X)` → "even" / "over" / "under" |
| Check consistency | `matrix_rank(X)` vs `matrix_rank(np.append(X,y,axis=1))` |
| Auto-solve LE | `w = solveLE(X, y)` |
| Left pseudo-inv (N > M) | `w = inv(X.T @ X) @ X.T @ y` |
| Right pseudo-inv (N < M) | `w = X.T @ inv(X @ X.T) @ y` |
| Add bias column | `np.hstack((np.ones((N,1)), X))` |
| Polynomial features | `P = polyTx(X, order)` |
| Poly regression | `w = solvePR(P, y)` |
| Poly + ridge | `w = solvePR(P, y, ridge=True, lamb=0.0001)` |
| Binary classification | `np.sign(X_test @ w)` |
| One-hot encode | `OneHotEncoder(sparse_output=False).fit_transform(y)` |
| Multi-class predict | `encoder.inverse_transform(X_test @ W)` |
| MSE | `np.mean((y_pred - y)**2)` |
| Rank of matrix | `matrix_rank(X)` |
| Standardize | `(X - X.mean(0)) / X.std(0)` |
| 1D → column vector | `x.reshape(-1, 1)` |
| Drop col → numpy | `np.array(df.drop('col', axis=1))` |
| Replace 0 with NaN | `df[cols].replace(0, np.nan)` |
| Filter by category | `df.loc[df['type'] == 'val']` |
| Gaussian P(a ≤ X ≤ b) | `stats.norm.cdf(b,mu,sig) - stats.norm.cdf(a,mu,sig)` |
| Augmented matrix | `np.append(X, y, axis=1)` |
