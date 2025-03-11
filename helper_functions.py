import numpy as np

def rfftnfreq_2d(shape, spacing, dtype=np.float64):
    """Broadcastable "``sparse``" wavevectors for ``numpy.fft.rfftn``.

    Parameters
    ----------
    shape : tuple of int
        Shape of ``rfftn`` input.
    spacing : float or None, optional
        Grid spacing. None is equivalent to a 2π spacing, with a wavevector period of 1.
    dtype : dtype_like

    Returns
    -------
    kvec : list of jax.numpy.ndarray
        Wavevectors.

    """
    freq_period = 1
    if spacing is not None:
        freq_period = 2 * np.pi / spacing

    kvec = []
    for axis, s in enumerate(shape[:-1]):
        k = np.fft.fftfreq(s).astype(dtype) * freq_period
        kvec.append(k)

    k = np.fft.rfftfreq(shape[-1]).astype(dtype) * freq_period
    kvec.append(k)

    kvec = np.meshgrid(*kvec, indexing='ij', sparse=True)

    return kvec

def cic_preprocess(skewers_fin,nc):
    part = skewers_fin.reshape(1,-1,3)
    shape = [nc,nc,nc]
    nx, ny, nz = shape[0], shape[1], shape[2]
    ncc = [nx, ny, nz]


    if len(part.shape) > 3:
        part = np.reshape(part, (batch_size, -1, 3))

    part = np.expand_dims(part,2)
    floor = np.floor(part)
    connection = np.array([[[0, 0, 0], [1., 0, 0], [0., 1, 0], [0., 0, 1], [1., 1, 0],
                      [1., 0, 1], [0., 1, 1], [1., 1, 1]]])


    neighboor_coords =floor + connection
    kernel = 1. - np.abs(part - neighboor_coords)
    kernel = kernel[..., 0] * kernel[..., 1] * kernel[..., 2]

    neighboor_coords=np.array(neighboor_coords,dtype=np.int32)
    neighboor_coords = np.mod(neighboor_coords, ncc)

    na = neighboor_coords[0,:,:].reshape(-1,3)
    naa = ny**2*na[:,0]+nx*na[:,1]+na[:,2]
    return naa,kernel