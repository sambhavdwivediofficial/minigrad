from minigrad import Linear, Sequential, Tanh


def build_xor_model():
    return Sequential(
        Linear(2, 16, seed=1),
        Tanh(),
        Linear(16, 16, seed=2),
        Tanh(),
        Linear(16, 1, seed=3),
    )
