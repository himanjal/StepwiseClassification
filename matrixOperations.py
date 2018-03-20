import numpy as np

def problem1 (A, B):
	return A + B

def problem2 (A, B, C):
	return np.dot(A,B) - C

def problem3 (A, B, C):
	return A*B + C.T

def problem4 (x, y):
	return np.inner(x,y)

def problem5 (A):
	return np.zeros(A.size).reshape(A.shape)


def problem6 (A):
	return np.ones(A.shape[0]).reshape(A.shape[0],1)

def problem7 (A):
	return np.linalg.inv(A)

def problem8 (A, alpha):
	return A + alpha * np.eye(A.shape[0])

def problem9 (A, i, j):
	return A[i][j]

def problem10 (A, i):
	return np.sum(A[i])

def problem11 (A, c, d):
	geaterThan = A[np.nonzero(A > c)]
	lessThan = geaterThan[np.nonzero(greaterThan < d)]
	return np.mean(lessThan)


def problem12 (A, k):
	eignvalues = np.linalg.eig(A)[1]
	colNum = A.shape - k
	return eignvalues[:,colNum:]

def problem13 (A, x):
	return np.linalg.solve(A,x)

def problem14 (A, x):
	ytranspose = np.linalg.solve(A.T, x.T)
	return ytranspose.T


