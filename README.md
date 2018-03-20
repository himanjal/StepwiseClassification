#Stepwise Classification

1 Python and numpy
We practice some Matrix operations and solve the basic 14 problems using numpy N dimensional array matrices 

1. Given matrices A and B, compute and return an expression for A + B. [ 2 pts ]

2. Given matrices A, B, and C, compute and return AB − C (i.e., right-multiply matrix A by matrix B, and then subtract C). Use dot or np.dot. [ 2 pts ]

3. Given matrices A, B, and C, return AB + C>, where represents the element-wise (Hadamard) product and > represents matrix transpose. In numpy, the element-wise product is obtained simply with *. [2pts]

4. Given column vectors x and y, compute the inner product of x and y (i.e., x>y). [2 pts]

5. Given matrix A, return a matrix with the same dimensions as A but that contains all zeros. Use np.zeros. [ 2 pts ]

6. Given matrix A, return a vector with the same number of rows as A but that contains all ones. Use
np.ones. [ 2 pts ]

7. Given (invertible) matrix A, compute A−1. [ 2 pts ]

8. Given square matrix A and (scalar) α, compute A + αI, where I is the identity matrix with the same dimensions as A. Use np.eye. [ 2 pts ]

9. Given matrix A and integers i, j, return the jth column of the ith row of A, i.e., Aij . [ 2 pts ]

10. Given matrix A and integer i, return the sum of all the entries in the ith row, i.e., Pj Aij . Do not use a loop, which in Python is very slow. Instead use the np.sum function. [ 4 pts ]

11. Given matrix A and scalars c, d, compute the arithmetic mean over all entries of A that are betweenc and d (inclusive). In other words, if S = {(i, j) : c ≤ Aij ≤ d}, then compute 1 |S| P (i,j)∈S Aij . Use np.nonzero along with np.mean. [ 5 pts ]

12. Given an (n × n) matrix A and integer k, return an (n × k) matrix containing the right-eigenvectors of A corresponding to the k largest eigenvalues of A. Use np.linalg.eig to compute eigenvectors. [5 pts ]

13. Given square matrix A and column vector x, use np.linalg.solve to compute A−1x. [ 4 pts ]

14. Given square matrix A and row vector x, use np.linalg.solve to compute xA−1. Hint: AB =(B>A>)>. [ 4 pts ]


#2 Step-wise Classification

In this part of the assignment you will train a very simple smile classifier that analyzes a grayscale image x ∈ R 24×24 and outputs a prediction ˆy ∈ {0, 1} indicating whether the image is smiling (1) or not (0).
The classifier will make its decision based on very simple features of the input image consisting of binary comparisons between pixel values. Each feature is computed as I[xr1,c1 > xr2,c2 ]where ri, ci ∈ {0, 1, 2, . . . , 23} are the row and column indices, respectively, and I[·] is an indicator function whose value is 1 if the condition is true and 0 otherwise. In general, these features are not very good, but nonetheless they will enable the classifier to achieve an accuracy (fPC) much better than just guessing or just choosing the dominant class. Based on these features, you should train a smile classifier using step-wise classification for m = 5 features. Step-wise classification/regression is a greedy algorithm: at each round j, choose the jth feature (r1, c1, r2, c2) such that – when it is added to the set of j − 1 features that have already been selected – the accuracy (fPC) of the overall classifier is maximized. Once you select a feature, it stays in the set forever – it can never be removed. (Otherwise, it wouldn’t be a greedy algorithm at all.)
Your training code should perform step-wise classification by analyzing the images and maximizing the accuracy only on the training set. To estimate how well the “machine” (smile classifier) you are building would work on a set of images not used for training, you should then measure it accuracy on the test set.


Tasks:

1. Download the following data files: trainingFaces.npy,trainingLabels.npy,testingFaces.npy,testingLabels.npy.

2. Write code to train a step-wise classifier for m = 5 features of the binary comparison type described above; the greedy procedure should maximize fPC (which you will also need to implement). At each round, the code should examine every possible feature (r1, c1, r2, c2).. Make sure your code is vectorized to improve run-time performance (wall-clock time) [ 25 pts ].

3. Write code to analyze how training/testing accuracy changes as a function of number of examples n ∈ {400, 800, 1200, 1600, 2000} (implement this in a for-loop)

  (a) Run your step-wise classifier training code only on the first n examples of the training set.

  (b) Measure and record the training accuracy of the trained classifier on the n examples.

  (c) Measure and record the testing accuracy of the classifier on the (entire) test set.

Important: you must write code (a simple loop) to do this – do not just do it manually for each value of n. This is good experimental practice in general and is especially important in machine learning to ensure reproducibility of results. [ 8 pts ].

4. In a PDF document (you can use whatever tool you like – LaTeX, Word, Google Docs, etc. – but make sure you export to PDF), show the training accuracy and testing accuracy as a function of n. Please use the following simple format: n trainingAccuracy testingAccuracy n E {400, 800, 1200, 1600, 2000}. Moreover, characterize in words how the training accuracy and testing accuracy changes as a function of n, and how the two curves relate to each other. What trends do you observe? [4 pts]

5. Visualizing the learned features: It is very important in empirical machine learning work to visualize what was actually learned during training. This can be useful for debugging to make sure that your training code is working as it should, and also to make sure your training and testing sets are selected wisely. For n = 2000, visualize the m = 5 features that were learned by (a) displaying any face image from the test set; and (b) drawing a square around the specific pixel locations ((r1, c1) and (r2, c2)) that are examined by the feature. You can use the example code in the homework1 smile.py template to render the image. Insert the graphic (just one showing all 5 features) into your PDF file. [ 3 pts].

Tip on vectorization: Implement your training algorithm so that, for any particular feature (r1, c1, r2, c2), all the feature values (over all the n training images) are extracted at once using numpy – do not use a loop.
Even after vectorizing my own code, running the experiments required in this assignment took about 1 hour (on a single CPU of a MacBook 2016 laptop).

