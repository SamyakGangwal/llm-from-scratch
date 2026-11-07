import torch
from engine import Value

TOL = 1e-6

def check(fn, *inputs):
    """Run fn on Values and on torch tensors, then compare output and grads."""
    vs = [Value(x) for x in inputs]
    out = fn(*vs)
    out.backward()

    ts = [torch.tensor([x]).double().requires_grad_() for x in inputs]
    tout = fn(*ts)
    tout.backward()

    assert abs(out.data - tout.item()) < TOL, f"forward: {out.data} vs {tout.item()}"
    for i, (v, t) in enumerate(zip(vs, ts)):
        assert abs(v.grad - t.grad.item()) < TOL, f"input {i} grad: {v.grad} vs {t.grad.item()}"


def test_add():
    check(lambda a, b: a + b, 2.0, -3.0)
    check(lambda a: a + a + 1.0, 2.0)
    check(lambda a: 1.0 + a, 2.0)


def test_mul():
    check(lambda a, b: a * b, 2.0, -3.0)
    check(lambda a: a * a, 3.0)
    check(lambda a: 2.0 * a, 3.0)


def test_sub():
    check(lambda a, b: a - b, 2.0, -3.0)
    check(lambda a: a - a * a, 3.0)
    check(lambda a: -a, 3.0)


def test_div():
    check(lambda a, b: a / b, 2.0, -3.0)
    check(lambda a: a / (a * a + 1.0), 3.0)
    check(lambda a: a / 4.0, 3.0)


def test_pow():
    check(lambda a: a**2, 3.0)
    check(lambda a: a**-1, 3.0)
    check(lambda a: a**0.5, 3.0)
    check(lambda a: a**3 + a**2, -1.5)


def test_tanh():
    check(lambda a: a.tanh(), 0.7)
    check(lambda a: (a * a).tanh() + a, -0.4)


def test_exp():
    check(lambda a: a.exp(), 0.7)
    check(lambda a: a.exp() * a, -0.4)


def test_accumulation():
    check(lambda a: a * a + a, 3.0)
    check(lambda a, b: (a + b) * (a - b) + a * b, 2.0, -3.0)


def test_mixed():
    def f(a, b):
        c = a + b
        d = a * b + b**3
        c = c + c + 1.0
        c = c + 1.0 + c + (-a)
        d = d + d * 2.0 + (b + a).tanh()
        d = d + 3.0 * d + (b - a).tanh()
        e = c - d
        f = e**2
        g = f / 2.0
        g = g + f**-1 * 10.0
        return (g * 0.01).exp() + a / b

    check(f, -4.0, 2.0)
