import numpy as np
from minigrad import Linear, Sequential, Tanh, Tensor, load_checkpoint, save_checkpoint


def build_test_model():
    return Sequential(
        Linear(2, 4, seed=11),
        Tanh(),
        Linear(4, 1, seed=12),
    )


def test_checkpoint_round_trip(tmp_path):
    model = build_test_model()
    inputs = Tensor([[0.2, -0.8], [0.7, 0.4]])
    expected_output = model(inputs).data.copy()
    expected_state = model.state_dict()
    checkpoint_path = tmp_path / "model.npz"

    save_checkpoint(model, checkpoint_path)
    restored_model = build_test_model()
    load_checkpoint(restored_model, checkpoint_path)

    for name, expected_value in expected_state.items():
        np.testing.assert_allclose(restored_model.state_dict()[name], expected_value)
    np.testing.assert_allclose(restored_model(inputs).data, expected_output)
