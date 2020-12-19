## Papers Read

​	For the domain of transfer learning, I implemented a self-taught clustering, based on the paper “Self-taught Clustering”.^[1]^ In order to have more background in the area, I first read the paper “Self-taught Learning: Transfer Learning  from Unlabeled Data”.^[2]^ The paper essentially attempts to find a higher level feature representation on a set of auxiliary data that they then apply to the learning task, so the classifier does not need to learn the feature representation itself. They make the distinction betwee self-taught learning and transfer learning by specifying that transfer learning is typically used for similar domains (e.g. classifying unlabeled images of elephants and rhinos by first classifying labeled images of horses and goats), whereas self-taught learning does not need to be a similar domain, only the same preliminary data representation (e.g. images). 

​	For self-taught learning specifically, the most important aspect is being able to learn how to represent the data. In order to do so, the authors modify an optimization problem proposed earlier: 
$$
minimize_{b, a}\sum_i||x_u^{(i)} - \sum_ja_j^{(i)}b_j||_2^2 + \beta||a^{(i)}||_i s.t. \\
||b_j||_2 \le 1, \forall j\in 1, 2, ..., s
$$
where $b$ is set of basis vectors $\{b_1, b_2, ..., b_s\}$ where $b_j\in \real^n$, $a$ is the set of activations $\{a^{(1)}, a^{(2)}, ..., a^{(k)}\}$ where $a^{(i)}\in \real^s$, $\beta$ is a trade-off value, and $a_j^{(i)}$ is the activation of basis $b_j$ for the input $x_u^{(i)}$. Essentially, the objective function uses the two terms in the optimization problem above by: using the first term to be reconstructed as a linear combination of the basis $b_j$ and using the second term to make sure the activations are as sparse as possible. This sparsity is important because it will allow for more generalization when applied to the target domain. Solving this for the unlabeled training data will give the basis vectors. The authors then propose applying a similar optimization problem to a set of labeled training data, with the intention this time to learn the activations for the basis vectors:
$$
\hat a(x_l^{(i)}) = arg min_{a^{(i)}}||x_l^{(i)} - \sum_ja_j^{(i)}b_j||_2^2+\beta ||a^{(i)}||_1
$$
This ensures the data is represented as a sparse linear combination of the input $x_l^{(i)}$, and a classifier can be trained on this data to find the activation set $\hat a$. The classifier is then returned and can be applied to a different domain with unlabeled examples. 



Self-taught clustering is an instance of unsupervised learning. It uses a target data set $X$, an auxiliary data set $Y$, and a common feature space between $X$ and $Y$, $Z$. The output is the optimal clustering of the target data set. The sets $\hat X, \hat Y, \hat Z$ are used to describe the sets of clusters for each mapping (for example, $\hat x_i$ is the i^th^ cluster of data points belonging to $X$). We need a clustering function for each variable, $C_X$, $C_Y$, and $C_Z$ to map data to a specific cluster (e.g. $C_X: X \rightarrow \hat X$. In order to do this, it uses the auxiliary data set and common feature space in order to improve its clustering algorithm at each iteration. The authors use an extension of information theoretic co-clustering to aid them in their self-taught clustering algorithm. The objective function attempts to minimize the loss in mutual information between the examples and features:
$$
I(X, Z) - I(\hat X, \hat Z)
$$
where $I$ denotes the mutual information between the instances and features. Both these can be simplified further:
$$
I(X, Z) = \sum_{x\in X}\sum_{z\in Z}p(x, z)log(\frac{p(x, z)}{p(x)p(z)}) \\I(\hat X, \hat Z) = p(\hat x, \hat z) = \sum_{x\in\hat x}\sum_{z\in\hat z}p(x, z)
$$
The self-taught clustering algorithm is using co-clustering to improve the target clusters, meaning the auxiliary data $Y$ is being used to help improve the clustering of $X$. The objective function is therefore defined as 
$$
\tau = I(X, Z) - I(\hat X, \hat Z) + \lambda\left[I(Y, Z) - I(\hat Y, \hat Z)\right]
$$
The mutual information involving $Y$ and $\hat Y$ is multiplied by a trade-off parameter, $\lambda$, in order to make sure the auxiliary data does not have too much influence over the target data. Minimizing the objective function is non-trivial, since it is non-convex. The authors therefore used the Kullback-Leibler divergence (KL divergence) in order to re-write the equation and try to minimized it. 

​	In order to minimize this function, start by creating the joint probability distribution of a variable and the feature space with respect to their co-clusters:
$$
\hat p(x, z) = p(\hat x, \hat z) p\left(\frac{x}{\hat x}\right)p\left(\frac{z}{\hat z}\right)
$$
where $p(\hat x, \hat z)$ is the joint probability distribution of clusters $\hat x, \hat z$, $p\left(\frac{x}{\hat x}\right)$ is the total number of features in $x$ divided by the total number of features in the cluster $\hat x$, and $p\left(\frac{z}{\hat z}\right)$ is the total number of times the features $z$ appears divided by the total number of times each feature in $\hat z$ appears. The same equation can be written in terms of $Y$ to define the co-cluster joint probability distribution between $Y$ and $Z$:
$$
\hat q(y, z) = q(\hat y, \hat z) q\left(\frac{y}{\hat y}\right)q\left(\frac{z}{\hat z}\right)
$$
The objective function can be re-written in the form of KL divergence:
$$
\tau = I(X, Z) - I(\hat X, \hat Z) + \lambda\left[I(Y, Z) - I(\hat Y, \hat Z)\right] = \\D(p(X, Z)||\hat p(X, Z)) + \lambda D(q(Y, Z)||\hat q(Y, Z))
$$
where $D(\frac{}{}||\frac{}{})$ denotes the KL divergence between the probability distributions,
$$
D(p(x), q(x)) = \sum_xp(x)log\left(\frac{p(x)}{q(x)}\right)
$$
The KL divergence can be simplified for our objective function:
$$
\begin{align}
D(p(X, Z)||\hat p(X, Z)) &= \sum_{\hat x\in \hat X}\sum_{x\in \hat x}p(x)D(p(Z|x)||\hat p(Z|\hat x)) \\
&= \sum_{\hat z\in \hat Z}\sum_{z\in \hat z}p(z)D(p(X|z)||\hat p(X|\hat z))
\end{align}
$$
which can be applied similarly for $Y$:
$$
\begin{align}
D(p(Y, Z)||\hat p(Y, Z)) &= \sum_{\hat y\in \hat Y}\sum_{y\in \hat y}p(y)D(p(Z|y)||\hat p(Z|\hat y)) \\
&= \sum_{\hat z\in \hat Z}\sum_{z\in \hat z}p(z)D(p(Y|z)||\hat p(Y|\hat z))
\end{align}
$$
So, by minimizing $D(p(Z|x)||\hat p(Z|\hat x))$ for a single $x$, the optimization function will find a global minimum and the values of $x$ will be put into their respective clusters. So, for each data point $x$, the clustering function $C_X$ is defined as:
$$
C_X(x) = argmin_{\hat x\in \hat X}D(p(Z|x)||\hat p(Z|\hat x))
$$
and for each data point $y$:
$$
C_Y(y) = argmin_{\hat y\in \hat Y}D(p(Z|y)||\hat p(Z|\hat y))
$$
and finally for a feature $z$:
$$
C_Z(z) = argmin_{\hat z\in \hat Z}p(z)D(p(X|z)||\hat p(X|\hat z)) + \lambda q(z)D(p(Y|z)||\hat p(Y|\hat z))
$$
By finding the $\hat x$, $\hat y$, and $\hat z$ that are best for each data point $x$, $y$, and $z$ respectively, the objective function will be minimized for each iteration. Formally, the algorithm implemented can be drawn up as such:



---

Algorithm 1: Self Taught Clustering Algorithm
$$
\begin{align}
&\text{Input: A target unlabeled data set $X$, an auxiliary unlabeled data set $Y$, the feature space} \\
&\text{shared by both $X$ and $Y$, the initial clustering functions $C_X^{(0)}$, $C_Y^{(0)}$, and $C_Z^{(0)}$, and the} \\
&\text{number of iterations T} \\
&\text{Output: The clustering function $C_X^{(T)}$} \\
&\text{Procedure:}& \\
&\text{1: Initialize the joint probability distributions $p(X, Z)$ and $q(Y, Z)$ based on the data} \\
&\text{2: Initialize $\hat p^{(0)}(X, Z)$ and $\hat q^{(0)}(Y, Z)$} \\
&\text{3: for $i = 1... T$ iterations:} \\
&\text{		4: Update $C_X^{(i)}(X)$ , $C_Y^{(i)}(Y)$, and $C_Z^{(i)}(Z)$ } \\
&\text{		5: Update $\hat p^{(i)}(X, Z)$ and $\hat q^{(i)}(Y, Z)$}\\
&\text{6: end for} \\
&\text{7: Return $C_X^{(T)}$}

\end{align}
$$
​	I implemented the above algorithm for my final project. I did so using Python, with the following libraries: the log function from math, a few mathematical functions from numpy, shuffle from random, several functions from os in order to extract the data, a few data types from typing, again to help extract the data, and sys in order to read the arguments. 



## Experiments

Oof. 



## Conclusions



## Citations

[1]   W.  Dai,  Q.  Yang,  G.  Xue,  and  Y.  Yu,  “Self-taught  clustering,”  inProceedings of the 25th International Conference of Machine Learning.ACM, July 2008, pp. 200–207

[2] Raina, R., Battle, A., Lee, H., Packer, B., & Ng, A. Y.(2007). Self-taught learning: transfer learning fromunlabeled data.Proceedings of the Twenty-fourthInternational Conference on Machine Learning(pp.759–766).