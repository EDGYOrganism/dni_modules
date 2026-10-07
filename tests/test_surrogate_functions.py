import pytest
import torch

from dni_modules import ATan, ATanSurrogate, Triangle, TriangleSurrogate


def test_atan_forward_shape():
    """Test that forward pass returns correct shape"""
    mem = torch.randn(5, 10, requires_grad=True)
    spk = ATan.apply(mem)
    assert spk.shape == mem.shape


def test_atan_forward_heaviside():
    """Test that forward pass implements Heaviside function"""
    mem = torch.tensor([[-1.0, 0.0, 0.5], [1.0, -0.5, 2.0]], requires_grad=True)
    spk = ATan.apply(mem)
    expected = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
    assert torch.allclose(spk, expected)


def test_atan_backward_exists():
    """Test that backward pass computes gradients"""
    mem = torch.randn(3, 4, requires_grad=True)
    spk = ATan.apply(mem)
    loss = spk.sum()
    loss.backward()
    assert mem.grad is not None
    assert mem.grad.shape == mem.shape


def test_atan_backward_non_zero():
    """Test that backward pass produces non-zero gradients"""
    mem = torch.tensor([[0.1, 0.5, 1.0]], requires_grad=True)
    spk = ATan.apply(mem, 10)
    loss = spk.sum()
    loss.backward()
    assert not torch.allclose(mem.grad, torch.zeros_like(mem.grad))


def test_atan_hor_scaling_parameter():
    """Test that scaling parameter affects gradients"""
    mem1 = torch.tensor([[0.5]], requires_grad=True)
    mem2 = torch.tensor([[0.5]], requires_grad=True)

    spk1 = ATan.apply(mem1, 10)
    loss1 = spk1.sum()
    loss1.backward()
    grad1 = mem1.grad.clone()

    spk2 = ATan.apply(mem2, 5)
    loss2 = spk2.sum()
    loss2.backward()
    grad2 = mem2.grad.clone()

    assert not torch.allclose(grad1, grad2)


def test_atan_default_scaling():
    """Test that default scaling parameter is used and that the gradient calculation is correct"""
    mem = torch.tensor([[0.5]], requires_grad=True)
    spk = ATan.apply(mem)
    loss = spk.sum()
    loss.backward()

    expected_grad = 1 / (1 + (torch.pi * mem).pow(2))

    assert spk.shape == torch.Size([1, 1])
    assert torch.allclose(expected_grad, mem.grad)


def test_atan_grad_calculation():
    """Test that the gradient computed by ATan is correct"""
    hor_scaling = 10
    mem = torch.tensor([[0.1, 0.2]], requires_grad=True)

    spk = ATan.apply(mem, hor_scaling)
    loss = spk.sum()
    loss.backward()

    grad = mem.grad.clone()

    expected_grad = 1 / (1 + (torch.pi * mem * hor_scaling).pow(2))

    assert torch.allclose(expected_grad, grad)


def test_atan_batch_processing():
    """Test ATan works with batch inputs"""
    batch_size = 32
    feature_size = 64
    mem = torch.randn(batch_size, feature_size, requires_grad=True)
    spk = ATan.apply(mem, 10)
    assert spk.shape == (batch_size, feature_size)
    assert spk.dtype == torch.float32


@pytest.mark.parametrize("hor_scaling", [0.5, 1.5])
def test_atan_surrogate_init(hor_scaling):
    """Test __init__ function of ATanSurrogate class"""
    atan = ATanSurrogate(hor_scaling=hor_scaling)
    assert atan.hor_scaling == hor_scaling


@pytest.mark.parametrize("hor_scaling", [0.5, 1.5])
def test_atan_surrogate_call(hor_scaling):
    """Test __call__ function of ATanSurrogate class"""
    mem = torch.tensor([[-1.0, 0.0, 0.5], [1.0, -0.5, 2.0]], requires_grad=True)
    atan = ATanSurrogate(hor_scaling=hor_scaling)
    spk = atan(mem)
    expected = ATan.apply(mem, hor_scaling)
    assert torch.allclose(spk, expected)


@pytest.mark.parametrize("hor_scaling", [0.5, 1.5])
def test_atan_surrogate_surrogate_grad(hor_scaling):
    """Test the surrogate_grad() function of ATanSurrogate class"""
    atan = ATanSurrogate(hor_scaling=hor_scaling)

    mem = torch.tensor([[0.5, 0.2], [0.8, 0.0]], requires_grad=True)
    spk = ATan.apply(mem, hor_scaling)
    loss = spk.sum()
    loss.backward()

    assert spk.shape == torch.Size([2, 2])
    assert torch.allclose(atan.surrogate_grad(mem), mem.grad)


# Triangle surrogate gradient tests
def test_triangle_forward_shape():
    """Test that forward pass returns correct shape"""
    mem = torch.randn(5, 10, requires_grad=True)
    spk = Triangle.apply(mem)
    assert spk.shape == mem.shape


def test_triangle_forward_heaviside():
    """Test that forward pass implements Heaviside function"""
    mem = torch.tensor([[-1.0, 0.0, 0.5], [1.0, -0.5, 2.0]], requires_grad=True)
    spk = Triangle.apply(mem)
    expected = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
    assert torch.allclose(spk, expected)


def test_triangle_backward_exists():
    """Test that backward pass computes gradients"""
    mem = torch.randn(3, 4, requires_grad=True)
    spk = Triangle.apply(mem)
    loss = spk.sum()
    loss.backward()
    assert mem.grad is not None
    assert mem.grad.shape == mem.shape


def test_triangle_backward_non_zero():
    """Test that backward pass produces non-zero gradients"""
    mem = torch.tensor([[0.1, 0.15, 0.2]], requires_grad=True)
    spk = Triangle.apply(mem)
    loss = spk.sum()
    loss.backward()
    assert not torch.allclose(mem.grad, torch.zeros_like(mem.grad))


@pytest.mark.parametrize("gamma", [0.2, 0.8])
def test_triangle_surrogate_grad_calculation(gamma):
    """Test that surrogate gradient is calculated correctly"""
    mem = torch.tensor([[0.1, 0.2]], requires_grad=True)
    spk = Triangle.apply(mem, gamma)
    loss = spk.sum()
    loss.backward()

    # Expected gradient: gamma * max(0, 1 - |x|)
    expected_grad = gamma * torch.maximum(torch.tensor(0.0), 1 - torch.abs(mem))

    assert torch.allclose(mem.grad, expected_grad, atol=1e-6)


def test_triangle_surrogate_grad_zero_outside_range():
    """Test that surrogate gradient is zero outside (-1, 1)"""
    mem = torch.tensor([[-2.0, -1.0, 1.0, 2.0]], requires_grad=True)
    spk = Triangle.apply(mem)
    loss = spk.sum()
    loss.backward()

    # Gradient should be zero for |x| > 1
    assert torch.allclose(mem.grad[0, 0], torch.tensor(0.0))
    assert torch.allclose(mem.grad[0, 1], torch.tensor(0.0))
    assert torch.allclose(mem.grad[0, 2], torch.tensor(0.0))
    assert torch.allclose(mem.grad[0, 3], torch.tensor(0.0))


@pytest.mark.parametrize("gamma", [0.2, 0.8])
def test_triangle_surrogate_grad_maximum_at_zero(gamma):
    """Test that surrogate gradient is maximum at x=0"""
    mem_zero = torch.tensor([[0.0]], requires_grad=True)
    mem_small = torch.tensor([[0.1]], requires_grad=True)

    spk_zero = Triangle.apply(mem_zero, gamma)
    loss_zero = spk_zero.sum()
    loss_zero.backward()
    grad_zero = mem_zero.grad.clone()

    spk_small = Triangle.apply(mem_small, gamma)
    loss_small = spk_small.sum()
    loss_small.backward()
    grad_small = mem_small.grad.clone()

    # Gradient at zero should be maximum (gamma)
    assert torch.allclose(grad_zero, torch.tensor(gamma))
    # Gradient at small value should be less than at zero
    assert grad_small < grad_zero


@pytest.mark.parametrize("gamma", [0.2, 0.8])
def test_triangle_surrogate_grad_linear_decay(gamma):
    """Test that surrogate gradient decays linearly within [0, 1]"""
    mem = torch.tensor([[0.0, 0.25, 0.5, 0.75]], requires_grad=True)
    spk = Triangle.apply(mem, gamma)
    loss = spk.sum()
    loss.backward()

    # Due to linearity, gradients should decrease uniformly
    grads = mem.grad[0].tolist()
    assert grads[0] > grads[1] > grads[2] > grads[3]


def test_triangle_gamma_parameter():
    """Test that gamma parameter affects gradient magnitude"""
    # Test with first gamma value
    gamma = 0.3
    mem1 = torch.tensor([[0.0]], requires_grad=True)
    spk1 = Triangle.apply(mem1, gamma)
    loss1 = spk1.sum()
    loss1.backward()
    grad1 = mem1.grad.clone()

    # Test with different gamma value
    gamma = 0.6
    mem2 = torch.tensor([[0.0]], requires_grad=True)
    spk2 = Triangle.apply(mem2, gamma)
    loss2 = spk2.sum()
    loss2.backward()
    grad2 = mem2.grad.clone()

    # Gradient should scale with gamma
    assert torch.allclose(grad2, 2 * grad1, atol=1e-5)


def test_triangle_batch_processing():
    """Test Triangle works with batch inputs"""
    batch_size = 32
    feature_size = 64
    mem = torch.randn(batch_size, feature_size, requires_grad=True)
    spk = Triangle.apply(mem)
    assert spk.shape == (batch_size, feature_size)
    assert spk.dtype == torch.float32


def test_triangle_symmetric_gradient():
    """Test that surrogate gradient is symmetric around zero"""
    mem_pos = torch.tensor([[0.3]], requires_grad=True)
    mem_neg = torch.tensor([[-0.3]], requires_grad=True)

    spk_pos = Triangle.apply(mem_pos)
    loss_pos = spk_pos.sum()
    loss_pos.backward()
    grad_pos = mem_pos.grad.clone()

    spk_neg = Triangle.apply(mem_neg)
    loss_neg = spk_neg.sum()
    loss_neg.backward()
    grad_neg = mem_neg.grad.clone()

    # Gradients should be equal due to symmetry
    assert torch.allclose(grad_pos, grad_neg)


def test_triangle_gradient_flow():
    """Test that gradients flow correctly through the function"""
    mem = torch.tensor([[0.1, 0.2, 0.3]], requires_grad=True)
    spk = Triangle.apply(mem)
    loss = (spk * torch.tensor([[1.0, 2.0, 3.0]])).sum()
    loss.backward()

    # Gradients should be non-zero
    assert mem.grad is not None
    assert mem.grad.abs().sum() > 0


def test_triangle_multiple_backward_passes():
    """Test that Triangle works with multiple forward/backward passes"""
    for _ in range(3):
        mem = torch.tensor([[0.1, 0.2]], requires_grad=True)
        spk = Triangle.apply(mem)
        loss = spk.sum()
        loss.backward()
        assert mem.grad is not None


@pytest.mark.parametrize("gamma", [0.2, 0.8])
def test_triangle_surrogate_init(gamma):
    """Test __init__ function of TriangleSurrogate class"""
    triangle = TriangleSurrogate(gamma=gamma)
    assert triangle.gamma == gamma


@pytest.mark.parametrize("gamma", [0.2, 0.8])
def test_triangle_surrogate_call(gamma):
    """Test __call__ function of TriangleSurrogate class"""
    mem = torch.tensor([[-1.0, 0.0, 0.5], [1.0, -0.5, 2.0]], requires_grad=True)
    triangle = TriangleSurrogate(gamma=gamma)
    spk = triangle(mem)
    expected = Triangle.apply(mem, gamma)
    assert torch.allclose(spk, expected)


@pytest.mark.parametrize("gamma", [0.2, 0.8])
def test_triangle_surrogate_surrogate_grad(gamma):
    """Test the surrogate_grad() function of TriangleSurrogate class"""
    triangle = TriangleSurrogate(gamma=gamma)

    mem = torch.tensor([[0.5, 0.2], [0.8, 0.0]], requires_grad=True)
    spk = Triangle.apply(mem, gamma)
    loss = spk.sum()
    loss.backward()

    assert spk.shape == torch.Size([2, 2])
    assert torch.allclose(triangle.surrogate_grad(mem), mem.grad)
