import numpy as np

def J_obs_2_true(f_j, sigma=None):

    np.random.seed(37)

    f_j_true = f_j + np.random.randn(1) * np.log(f_j)

    return f_j_true


# the assumed functions
# all functions use simple linear transformations to a given cond vector. The transformations are fixed by given random seeds.
def weight_func(cond, K, seed=9):
    """To generate the weights of the Gaussian components given the conditiontial parameters. The output is derived by dot multiplying a sequence to the conditional.

    Args:
        cond (array): Conditional parameter that determines the noise function. 1D array.
        K (int): Number of Gaussian components (how many Gaussians).
        seed (int, optional): Random seed. Defaults to 9.

    Returns:
        array: The weight of each Gaussian component, add up to unity. 1D array, lenght K. 
    """
    D_cond = len(cond)
    np.random.seed(seed)
    matr = np.random.permutation(K*D_cond*10)[:D_cond].reshape(D_cond) / (K*D_cond*5)
    weights0 = np.zeros(K)
    for i in range(K):
        weights0[i] = np.dot(matr**(1-i/10), cond**(1+i/10))

    weights = weights0 / weights0.sum()

    return weights


def means_func(cond, K, D, seed=13):
    """To generate the means of the Gaussian components given the conditiontial parameters. The output is derived by multiplying a matrix to the conditional.

    Args:
        cond (array): Conditional parameter that determines the noise function. 1D array.
        K (int): Number of Gaussian components (how many Gaussians).
        D (int): Dimension of data.
        seed (int, optional): Random seed. Defaults to 13.

    Returns:
        narray: The means of each Gaussian component. KxD array.
    """
    D_cond = len(cond)
    np.random.seed(seed)
    matr = np.random.permutation(K*D*D_cond*10)[:K*D*D_cond].reshape(K, D, D_cond) / (K*D)
    matr -= matr.mean()
    means0 = np.dot(matr, cond**1.2)
    
    return means0


def covar_func(cond, K, D, seed=21):
    """To generate the covariance of the Gaussian components given the conditiontial parameters. The output is derived by multiplying a matrix to the conditional.

    Args:
        cond (array): Conditional parameter that determines the noise function. 1D array.
        K (int): Number of Gaussian components (how many Gaussians).
        D (int): Dimension of data.
        seed (int, optional): Random seed. Defaults to 21.

    Returns:
        narray: Covariance matrices of the Gaussian mixture model. KxDxD array.
    """

    D_cond = len(cond)
    
    np.random.seed(seed)
    matr1 = np.random.permutation(K*D*D_cond*10)[:K*D*D_cond].reshape(K, D, D_cond) / (K*D*D_cond*50)
    l_diag = np.dot(matr1, cond**0.5)
    
    np.random.seed(seed)
    matr2 = np.random.permutation(K*D*(D-1)//2*D_cond*10)[:K*D*(D-1)//2*D_cond].reshape(K, D*(D-1)//2, D_cond) / (K*D*(D-1)//2*D_cond*50)
    l_lower = np.dot(matr2, cond**0.5)

    d_idx = np.eye(D).astype('bool')
    l_idx = np.tril_indices(D, -1)

    matrL = np.zeros((K, D, D))

    matrL[:, d_idx] = l_diag + (0.1)**0.5 # adding a constant to make sure positive definiteness.
    matrL[:, l_idx[0], l_idx[1]] = l_lower
    
    covars0 = np.matmul(matrL, np.swapaxes(matrL, -2, -1))

    return covars0


def noise_func(cond, D, sigma_d=1., sigma_l=0.):
    """To generate the noise matrix.

    Args:
        cond (array): Conditional parameter that determines the noise function. 1D array.
        D (int): Dimension of data.
        sigma_d (float, optional): Diagonal scale. Not very formal. Defaults to 1.
        sigma_l (float, optional): Off diagonal scale. Not very formal. Defaults to 0.

    Returns:
        narray: DxD noise matrix.
    """
    
    noise = np.zeros((D,D))

    d_idx = np.eye(D).astype('bool')
    l_idx = np.tril_indices(D, -1)

    np.random.seed()
    noise[d_idx] = np.random.rand(D) * sigma_d
    np.random.seed()
    noise[l_idx] = (np.random.rand(int(D*(D-1)/2))*2-1) * sigma_l
    noise = np.matmul(noise, np.transpose(noise))

    return noise


def sample_func(weights0, means0, covars0, noise=None, N=1):
    """Random sampling from a Gaussian mixture distribution.

    Args:
        weights0 (array): The weight of each Gaussian component, length K.
        means0 (narray): The means of each Gaussian component, shape KxD.
        covars0 (narray): Covariance matrices of the Gaussian mixture model, shape KxDxD.
        noise (narray, optional): The noise matrix, shape DxD. Defaults to None.
        N (int, optional): Size of samples. Defaults to 1.

    Returns:
        (narray, array): Samples and the index of which Gausssian component the sample belongs to. The sample has shape NxD, and the index has length N.
    """

    K = len(weights0)
    D = len(means0[0])
    X = np.empty((N, D))
    np.random.seed()
    draw = np.random.choice(np.arange(K), N, p=weights0) # which gaussian each data is from
    
    if noise is None:
        noise = np.zeros_like(covars0[0, :, :])

    for j in range(N):
        # the real data
        X[j, :] = np.random.multivariate_normal(
            mean=means0[draw[j], :],
            cov=covars0[draw[j], :, :] + noise
        )
        
    # shuffle the data
    p = np.random.permutation(N)
    X_train = X[p]
    
    # shuffle the draw array
    draw = draw[p]
    
    return X_train, draw


def data_load(N, K, D, D_cond, noisy=False, seed0=9):
    """General data loader.

    Args:
        N (int): Size of dataset.
        K (int): Number of Gaussian components (how many Gaussians).
        D (int): Dimension of data.
        D_cond (int): Dimension of conditional parameter.
        noisy (bool, optional): Whether add noise or not. Defaults to False.
        seed0 (int, optional): Random seed. Defaults to 9.

    Returns:
        (narray, narray, narray, narray, narray, narray, array): Conditional parameters, shape NxD. Weights, shape NxK. Means, shape KxD. Covariance matrices, shape NxKxDxD. Noise matrices, shape NxDxD (is zeros if noisy=False). Index indicating which component the data belong to, length N.
    """
    cond = np.zeros((N, D_cond))
    weights = np.zeros((N, K))
    means   = np.zeros((N, K, D))
    covars  = np.zeros((N, K, D, D))
    noise   = np.zeros((N, D, D))
    draw    = np.zeros(N)
    data    = np.zeros((N, D))


    # conditional paramesters
    for i in range(N):
        np.random.seed()
        cond[i] = np.random.rand(D_cond)
        weights[i]    = weight_func(cond[i], K, seed=seed0)
        means[i]      = means_func(cond[i], K, D, seed=seed0+4)
        covars[i]     = covar_func(cond[i], K, D, seed=seed0+12)

        if noisy is True:
            noise[i]      = noise_func(cond[i], D, sigma_d=1., sigma_l=0.5)
            data[i], draw[i] = sample_func(weights[i], means[i], covars[i], noise=noise[i])

        elif noisy is False:
            data[i], draw[i] = sample_func(weights[i], means[i], covars[i])

    return (cond, weights, means, covars, data, noise, draw)