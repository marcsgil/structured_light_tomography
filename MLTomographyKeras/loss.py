import keras

def dot(x, y):
    return keras.layers.Dot(axes=1)((x, y))[:, 0]


def fidelity_loss(y1, y2):
    N = y1.shape[1] // 2
    r1 = y1[:, :N]
    i1 = y1[:, N:]
    r2 = y2[:, :N]
    i2 = y2[:, N:]

    R = (r1*r2 + i1*i2).sum(axis=1)
    I = (r1*i2 - r2*i1).sum(axis=1)

    return 1 - (R**2 + I**2) / (dot(y1, y1) * dot(y2, y2))