import numpy as np
import torch

from fathomfollow.nav.estimator import VelocityEstimator, VelocityGRU


def test_gru_output_shape() -> None:
    model = VelocityGRU(input_size=9, hidden_size=32)
    x = torch.randn(2, 20, 9)
    out = model(x)
    assert out.shape == (2, 3)


def test_estimator_handles_no_dvl() -> None:
    est = VelocityEstimator(window_size=5)
    imu = np.random.randn(5, 6).astype(np.float32)
    vel = est.estimate(imu, dvl=None)
    assert vel.shape == (3,)


def test_tiny_overfit_loss_decreases() -> None:
    model = VelocityGRU(input_size=9, hidden_size=16)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    x = torch.randn(4, 10, 9)
    y = torch.randn(4, 3)
    losses = []
    for _ in range(20):
        opt.zero_grad()
        pred = model(x)
        loss = torch.nn.functional.mse_loss(pred, y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]
