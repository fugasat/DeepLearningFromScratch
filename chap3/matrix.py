import numpy as np


# 3x2
A = np.array([[1,2],[3,4],[5,6]])

# 2x3
B = np.array([[1,2,3],[4,5,6]])

# = 3x3
AB = np.dot(A, B)
print("3x2 . 2x3")
print(AB)


# 3x1
A = np.array([[1],[2],[3]])

# 1x3
B = np.array([[4,5,6]])

# = 3x3
AB = np.dot(A, B)
print("3x1 . 1x3")
print(AB)

# = 1x1
BA = np.dot(B, A)
print("1x3 . 3x1")
print(BA)



# 2x3
A = np.array([[1,2,3],[4,5,6]])

# 3 (!)3x1ではない
#B = np.array([[1],[2],[3]])
B = np.array([1,2,3])

# = 2　(!)2x1ではない
AB = np.dot(A, B)
print("2x3 . 3x1")
print(AB)

# NNの内積
print("NNの内積")
X = np.array([1,2])
W = np.array([[1, 3, 5], [2, 4, 6]])
print(W)

Y = np.dot(X, W)
print(Y)
