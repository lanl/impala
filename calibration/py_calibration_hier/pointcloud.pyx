cimport numpy as np
import numpy as np
import numpy.ma as ma
np.seterr(all = 'raise')
from libc.math cimport fabs

cdef class LocalSSq():
    cdef object ssqmat
    cdef int nobs

    @property
    def mat(self):
        return self.ssqmat

    @property
    def n(self):
        return self.nobs

    def __init__(self, np.ndarray[dtype = np.float64_t, ndim = 2] ssqmat, int nobs):
        self.ssqmat = ssqmat
        self.nobs = nobs
        return

# best result
cpdef LocalSSq localssq(
            np.ndarray[dtype = np.float_t, ndim = 2] obs,
            np.ndarray[dtype = np.float_t, ndim = 1] current,
            double distance,
            ):
    """ Finds localized unscaled cov matrix (obs-mean(obs))(obs-mean(obs))^t,
    and number of observations used to calculate said cov matrix.  retrieved
    from np.cov; returns both ssq and n."""
    d = len(current)
    try:
        assert obs.shape[0] > 1
    except AssertionError:
        return LocalSSq(np.zeros((d,d)), 0)
    cdef np.ndarray[dtype = np.float_t, ndim = 2] absdiff = np.abs(obs - current)
    cdef np.ndarray[dtype = np.float_t, ndim = 2] local = \
                            obs[np.where(np.all(absdiff < distance, axis = 1))]
    cdef int nlobs = local.shape[0]
    try:
        assert nlobs > d
    except AssertionError:
        return LocalSSq(np.zeros((d,d)), 0)
    return LocalSSq(np.cov(local.T) * (nlobs - 1), nlobs)

cpdef LocalSSq localssq1(
        np.ndarray[dtype = np.float_t, ndim = 2] obs,
        np.ndarray[dtype = np.float_t, ndim = 1] current,
        double distance,
        ):
    """ same as above, but finds local observations in a different way.

    Finds local observations as (obs-current)^2 < diff^2
    """
    cdef np.ndarray[dtype = np.float_t, ndim = 2] diff2 = \
                                (obs - current)*(obs - current)
    cdef double dist2 = distance * distance
    cdef np.ndarray[dtype = np.float_t, ndim = 2] local = \
                                obs[np.where(np.all(diff2 < dist2, axis = 1))]
    cdef int nlobs = local.shape[0]
    return LocalSSq(np.cov(local.T) * (nlobs - 1), nlobs)

cpdef LocalSSq localssq2(
            np.ndarray[dtype = np.float_t, ndim = 2] obs,
            np.ndarray[dtype = np.float_t, ndim = 1] current,
            double distance,
            ):
    """ same as above, but finds local observations in a different way.

    Finds local observations as abs(obs-current) < diff """
    cdef np.ndarray[dtype = np.float_t, ndim = 2] absdiff = np.abs(obs - current)
    cdef np.ndarray[dtype = np.float_t, ndim = 2] local = \
                            obs[np.where(np.all(absdiff < distance, axis = 1))]
    cdef int nlobs = local.shape[0]
    return LocalSSq(np.cov(local.T) * (nlobs - 1), nlobs)

cpdef LocalSSq localssq3(
        np.ndarray[dtype = np.float_t, ndim = 2] obs,
        np.ndarray[dtype = np.float_t, ndim = 1] current,
        double distance,
        ):
    """ attempts to imply numpy masking.  extremely slow. """
    cdef np.ndarray[dtype = np.float_t, ndim = 2] absdiff = np.abs(obs - current)
    cdef np.ndarray[dtype = np.float_t, ndim = 2] local = \
                ma.compress_rows(ma.masked_greater(absdiff,distance))
    cdef int nlobs = local.shape[0]
    return LocalSSq(np.cov(local.T) * (nlobs - 1), nlobs)

cpdef LocalSSq localssq4(
        np.ndarray[dtype = np.float_t, ndim = 2] obs,
        np.ndarray[dtype = np.float_t, ndim = 1] current,
        double distance,
        ):
    """ Same as localssq1, but finds local observations using infinity norm """
    d = len(current)
    cdef np.ndarray[dtype = np.float_t, ndim = 2] absdiff = np.abs(obs - current)
    cdef np.ndarray[dtype = np.float_t, ndim = 2] local = \
                                    obs[np.where(absdiff.max(axis = 1) < distance)]
    cdef int nlobs = local.shape[0]
    return LocalSSq(np.cov(local.T) * (nlobs - 1), nlobs)

cpdef np.ndarray[dtype = np.float_t, ndim = 2] localcovold(
        np.ndarray[dtype = np.float_t, ndim = 2] obs,
        np.ndarray[dtype = np.float_t, ndim = 1] current,
        double distance, double nu, double psi0,
        ):
    """
    Localized Covariance Matrix Window
    Arguments:
    -   obs:    Observations prior to current one
    -   current:    target location
    -   distance:   Covariance matrix window radius
    -   nu:     prior weighting for proposal matrix
    -   psi0:   prior diagonal value for proposal matrix
    """
    cdef int d = current.shape[0]
    cdef np.ndarray[dtype = np.float_t, ndim = 2] absdiff = np.abs(obs - current)
    cdef np.ndarray[dtype = np.float_t, ndim = 2] local = \
                            obs[np.where(np.all(absdiff < distance, axis = 1))]
    cdef int nlobs = local.shape[0]
    try:
        assert nlobs > d
    except AssertionError:
        return np.eye(d) * (psi0 / nu)
    return (np.cov(local.T) * (nlobs - 1) + np.eye(d) * psi0) / (nu + nlobs - d - 1)

cpdef np.ndarray[dtype = np.float_t, ndim = 2] localcov(
        np.ndarray[dtype = np.float_t, ndim = 2] obs,
        np.ndarray[dtype = np.float_t, ndim = 1] target,
        double distance, double nu, double psi0,
        ):
    """
    Localized Covariance Matrix window
    Arguments:
    -   obs:    Observations prior to current one
    -   current:    target location
    -   distance:   Covariance matrix window radius
    -   nu:     prior degrees of freedom for proposal matrix
    -   psi0:   prior diagonal value for proposal matrix
    """
    cdef int d = target.shape[0]  # dimension of data
    cdef int nobs = obs.shape[0]  # number of observations
    cdef int nlobs = 0            # Counter for number of local observations

    # Matrix to store local observations.  Declare as empty, fill later
    cdef np.ndarray[dtype = np.float_t, ndim = 2] local = \
            np.empty((nobs, d), dtype = np.float)
    # Vector to store sum (and then mean) of local Observations
    cdef np.ndarray[dtype = np.float_t, ndim = 1] lsm = \
            np.zeros(d, dtype = np.float)
    # Matrix to store sum of squares.
    cdef np.ndarray[dtype = np.float_t, ndim = 2] ssq

    # Iterators
    cdef int i, j

    # Loop through data by row
    for i in range(nobs):
        # For each column
        for j in range(d):
            # Check if local.  If not, break out of loop.
            if fabs(obs[i,j] - target[j]) > distance:
                break
        # If all dimensions of row are local, add observation to local Matrix
        # Iterate counter for local observations, add observation to sum
        else:
            local[nlobs] = obs[i]
            nlobs += 1
            for j in range(d):
                lsm[j] += obs[i,j]

    # Verify if number of local observations exceeds minimum
    try:
        assert nlobs > d
    except AssertionError:
        return np.eye(d) * psi0 / nu

    # Compute column-wise means
    for j in range(d):
        lsm[j] /= nlobs

    # Calculate deviations from mean for local observations
    for i in range(nlobs):
        for j in range(d):
            local[i,j] -= lsm[j]

    # Compute sum of squares matrix
    ssq = (local[:nlobs]).T.dot(local[:nlobs])

    # Return MAP estimate for inv. Wishart assuming diagonal prior
    return (ssq + np.eye(d) * psi0) / (nlobs + nu - d - 1)

# EOF
