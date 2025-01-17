# NumPy Linear Algebra Tutorial 📐
This guide covers the essential linear algebra concepts and their implementation in Python using NumPy, with comments to aid learning. 

---

## Importing NumPy
Before we begin, ensure you have NumPy installed. Import it into your Python environment:
```python
import numpy as np
```

---

## 1. Vectors and Basic Operations

### Creating Vectors
```python
# Define a vector
v = np.array([1, 2, 3])
print("Vector:", v)
```

### Operations on Vectors
```python
# Vector addition
v1 = np.array([1, 2])
v2 = np.array([3, 4])
addition = v1 + v2
print("Vector Addition:", addition)

# Scalar multiplication
scalar_mult = 2 * v1
print("Scalar Multiplication:", scalar_mult)

# Dot product
dot_product = np.dot(v1, v2)
print("Dot Product:", dot_product)
```

---

## 2. Matrices and Operations

### Creating Matrices
```python
# Define a matrix
A = np.array([[1, 2], [3, 4]])
print("Matrix A:\n", A)
```

### Operations on Matrices
```python
# Matrix addition
B = np.array([[5, 6], [7, 8]])
mat_add = A + B
print("Matrix Addition:\n", mat_add)

# Matrix multiplication
mat_mult = np.dot(A, B)
print("Matrix Multiplication:\n", mat_mult)

# Transpose
mat_transpose = A.T
print("Matrix Transpose:\n", mat_transpose)
```

---

## 3. Determinants and Inverses

### Determinant of a Matrix
```python
from numpy.linalg import det

# Determinant
determinant = det(A)
print("Determinant of A:", determinant)
```

### Inverse of a Matrix
```python
from numpy.linalg import inv

# Inverse
inverse = inv(A)
print("Inverse of A:\n", inverse)
```

---

## 4. Eigenvalues and Eigenvectors

### Eigen Decomposition
```python
# Eigenvalues and Eigenvectors
from numpy.linalg import eig

eigenvalues, eigenvectors = eig(A)
print("Eigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors)
```

---

## 5. Singular Value Decomposition (SVD)

### SVD Example
```python
# Singular Value Decomposition
U, S, VT = np.linalg.svd(A)
print("U:\n", U)
print("Sigma:", S)
print("V Transpose:\n", VT)
```

---

## 6. Solving Systems of Linear Equations

### Solving Ax = b
```python
# Define a system: Ax = b
b = np.array([5, 11])
x = np.linalg.solve(A, b)
print("Solution x:", x)
```

---

## 7. Practical Applications

### Example: Principal Component Analysis (PCA)
```python
# Data matrix
X = np.array([[2.5, 2.4], [0.5, 0.7], [2.2, 2.9], [1.9, 2.2], [3.1, 3.0], [2.3, 2.7], [2, 1.6], [1, 1.1], [1.5, 1.6], [1.1, 0.9]])

# Mean centering
X_meaned = X - np.mean(X, axis=0)

# Covariance matrix
cov_matrix = np.cov(X_meaned, rowvar=False)

# Eigen decomposition
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

print("Eigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors)
```

---

## Conclusion
This guide provides a solid foundation in NumPy-based linear algebra. By practicing these examples and extending them to real-world problems, you can deepen your understanding and application of these concepts in fields like machine learning and data analysis.

---

Feel free to contribute or suggest improvements to this tutorial! 🚀
