import random

import numpy as np
from lectrace import link, note, plot, table, text

from _util import untraced


def main():
    text("# micrograd: a neural network from scratch")
    text(
        "micrograd is a tiny autograd engine written by Andrej Karpathy. It "
        "works on single numbers instead of tensors, and the whole engine fits "
        "in about a hundred lines. Small as it is, it contains the one idea "
        "that every neural network library is built on: **backpropagation**, "
        "which works out how much each number in a calculation affected the "
        "result."
    )
    text(
        "This lecture follows Karpathy's micrograd video from *Neural Networks: "
        "Zero to Hero*, using the same code, with one change: the neuron uses "
        "ReLU instead of tanh. Step through it line by line and watch the "
        "variables on the right."
    )
    link(title="nn-zero-to-hero: micrograd lecture notebooks", url="https://github.com/karpathy/nn-zero-to-hero/tree/master/lectures/micrograd", authors=["Andrej Karpathy"], date="2022")
    link(title="micrograd", url="https://github.com/karpathy/micrograd", authors=["Andrej Karpathy"], date="2020")

    derivative_of_a_function()
    derivative_with_several_inputs()
    a_value_remembers_where_it_came_from()
    backprop_by_hand()
    local_derivatives()
    backward_does_it_for_you()
    why_gradients_add_up()
    a_neuron()
    a_network()
    loss_and_one_step_of_learning()
    training()

    text("## Where to go from here")
    text(
        "That is the whole of it. A neural network is a big mathematical "
        "expression. Its weights are some of the inputs. The loss is one number "
        "at the end that says how wrong it is. Backpropagation is the chain rule, "
        "applied one node at a time from the loss back to every weight, and "
        "gradient descent nudges each weight against its gradient. Repeat."
    )
    text(
        "PyTorch does exactly this. The difference is that its values are "
        "tensors instead of single numbers, so one node can be a whole matrix "
        "multiply, and the local derivatives are written in fast C++ and CUDA. "
        "The `_backward` closures, the topological sort, the `+=` and the "
        "zeroing of gradients are all still there, just out of sight."
    )


# ---------------------------------------------------------------------------
# 1


def f(x):
    return 3*x**2 - 4*x + 5


def derivative_of_a_function():
    text("## 1. What a derivative tells you")
    text(
        r"Start with an ordinary function, $f(x) = 3x^2 - 4x + 5$. The "
        r"derivative $f'(x)$ answers one question: if I nudge $x$ up by a tiny "
        r"amount $h$, how much does $f(x)$ move, relative to $h$?"
    )
    text(r"$$f'(x) = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}$$")
    text(
        "To see its shape, evaluate it at many points at once. `f` only uses "
        "`*`, `**`, `-` and `+`, so it works on a whole numpy array as well as "
        "on a single number."
    )
    xs = np.arange(-5, 5, 0.25)
    ys = f(xs)
    plot({
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "data": {"values": [{"x": float(x_), "f(x)": float(y_)} for x_, y_ in zip(xs, ys)]},
        "mark": "line",
        "encoding": {
            "x": {"field": "x", "type": "quantitative"},
            "y": {"field": "f(x)", "type": "quantitative"},
        },
        "width": 380,
        "height": 200,
    })
    text("We do not need the limit. A small enough `h` gives a very good estimate.")
    x = 3.0
    h = 0.0001
    slope = (f(x + h) - f(x)) / h
    text(
        r"Calculus gives the exact answer, $f'(x) = 6x - 4$, which is "
        + f"{6 * x - 4:g} at x = {x:g}. The estimate is {slope:.4f}, which is very close. "
        "A positive slope means that pushing `x` up makes `f` go up."
    )
    x = 2 / 3
    slope = (f(x + h) - f(x)) / h
    text(
        f"At x = 2/3 the estimate is {slope:.6f}, essentially zero. That is the "
        "bottom of the curve, where a small nudge in either direction barely "
        "changes `f`. Finding the bottom of a function by following its slope "
        "is exactly what training a neural network does."
    )


# ---------------------------------------------------------------------------
# 2


def derivative_with_several_inputs():
    text("## 2. Several inputs, one output")
    text(
        r"Now take $d = a \cdot b + c$. There are three inputs, so there are "
        r"three derivatives, one for each input, holding the other two fixed. "
        r"These are called **partial derivatives** and are written "
        r"$\frac{\partial d}{\partial a}$, $\frac{\partial d}{\partial b}$ and "
        r"$\frac{\partial d}{\partial c}$."
    )
    a = 2.0
    b = -3.0
    c = 10.0
    h = 0.0001
    d1 = a*b + c
    d2 = (a + h)*b + c
    slope_a = (d2 - d1) / h
    d2 = a*(b + h) + c
    slope_b = (d2 - d1) / h
    d2 = a*b + (c + h)
    slope_c = (d2 - d1) / h
    text(
        r"Each one matches what the math says. "
        r"$\frac{\partial d}{\partial a} = b$, which is "
        + f"{b:g} (estimate {slope_a:.4f}). "
        + r"$\frac{\partial d}{\partial b} = a$, which is "
        + f"{a:g} (estimate {slope_b:.4f}). "
        + r"$\frac{\partial d}{\partial c} = 1$ "
        + f"(estimate {slope_c:.4f})."
    )
    text(
        "Read the first one as: increasing `a` a little makes `d` go *down*, "
        f"by {abs(b):g} times as much, because `a` is multiplied by a negative "
        "number. That sentence is what a gradient means, for every weight in "
        "every neural network."
    )


# ---------------------------------------------------------------------------
# 3


def a_value_remembers_where_it_came_from():
    text("## 3. A number that remembers where it came from")
    text(
        "Nudging each input by hand works, but a neural network has thousands "
        "or billions of inputs. We want every derivative in one pass. To get "
        "that, each number needs to remember how it was made. That is what "
        "micrograd's `Value` class does."
    )
    text(
        "A `Value` wraps one float and keeps:\n\n"
        "- `data`, the number itself\n"
        "- `grad`, the derivative of the final output with respect to this number (0.0 until we compute it)\n"
        "- `_prev`, the Values it was computed from\n"
        "- `_op`, the operation that made it, empty for inputs\n"
        "- `_backward`, a small function that passes gradient back to `_prev`\n"
        "- `label`, a name, so we can tell nodes apart"
    )
    link(Value)
    text(
        "Here is a slightly bigger expression from the video, "
        r"$L = (a \cdot b + c) \cdot f$, built from Values. Step into the "
        "`a*b` line to see `__mul__` create a new Value whose `_prev` holds `a` and `b`."
    )
    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')  # @stepover
    c = Value(10.0, label='c')  # @stepover
    e = a*b; e.label = 'e'
    d = e + c; d.label = 'd'  # @stepover
    f = Value(-2.0, label='f')  # @stepover
    L = d * f; L.label = 'L'  # @stepover
    text(
        "Every Value knows its parents, so the whole calculation is now a graph:",
    )
    text(
        """
        a ──┐
            ├─ * ── e ──┐
        b ──┘           ├─ + ── d ──┐
        c ──────────────┘           ├─ * ── L
        f ──────────────────────────┘
        """,
        verbatim=True,
    )
    text(
        f"The forward pass gives `e` = {e.data:g}, `d` = {d.data:g} and "
        f"`L` = {L.data:g}. All the grads are still {L.grad}. Filling them in "
        "is the next step."
    )


# ---------------------------------------------------------------------------
# 4


def backprop_by_hand():
    text("## 4. Backpropagation by hand")
    text(
        r"We want $\frac{\partial L}{\partial x}$ for every node $x$. Start at "
        r"the end and walk backwards. The tool for each step is the **chain rule**: "
        r"if $L$ depends on $e$, and $e$ depends on $a$, then"
    )
    text(r"$$\frac{\partial L}{\partial a} = \frac{\partial L}{\partial e} \cdot \frac{\partial e}{\partial a}$$")
    text(
        "In words: how much `L` moves when `a` moves is how much `L` moves when "
        "`e` moves, times how much `e` moves when `a` moves. The first factor "
        "is already known by the time we get to `a`. The second is purely local: "
        "it only depends on the one operation that made `e`."
    )
    a = Value(2.0, label='a')  # @stepover
    b = Value(-3.0, label='b')  # @stepover
    c = Value(10.0, label='c')  # @stepover
    e = a*b; e.label = 'e'  # @stepover
    d = e + c; d.label = 'd'  # @stepover
    f = Value(-2.0, label='f')  # @stepover
    L = d * f; L.label = 'L'  # @stepover

    text(r"**The output.** $\frac{\partial L}{\partial L} = 1$. Nudging `L` moves `L` by the same amount.")
    L.grad = 1.0
    text(
        r"**Through the multiply** $L = d \cdot f$. The local derivatives are "
        r"$\frac{\partial L}{\partial d} = f$ and $\frac{\partial L}{\partial f} = d$, "
        "so each side gets the *other* side's value times the gradient coming in."
    )
    d.grad = f.data * L.grad
    f.grad = d.data * L.grad
    text(
        r"**Through the add** $d = e + c$. Here $\frac{\partial d}{\partial e} = 1$ "
        r"and $\frac{\partial d}{\partial c} = 1$, so an add just passes the "
        "gradient through to both inputs unchanged."
    )
    e.grad = 1.0 * d.grad
    c.grad = 1.0 * d.grad
    text(r"**Through the multiply** $e = a \cdot b$, the same rule as before.")
    a.grad = b.data * e.grad
    b.grad = a.data * e.grad
    text(
        f"So `a.grad` = {a.grad:g}, `b.grad` = {b.grad:g}, `c.grad` = {c.grad:g} "
        f"and `f.grad` = {f.grad:g}. Written out for `a`, the chain is"
    )
    text(
        r"$$\frac{\partial L}{\partial a} = \frac{\partial L}{\partial d} \cdot "
        r"\frac{\partial d}{\partial e} \cdot \frac{\partial e}{\partial a} = f \cdot 1 \cdot b"
        + f" = ({f.data:g})(1)({b.data:g}) = {a.grad:g}$$"
    )
    text("We can check that against the slow way, by nudging `a` and recomputing:")
    h = 0.0001
    L1 = L_from(a.data, b.data, c.data, f.data)  # @stepover
    L2 = L_from(a.data + h, b.data, c.data, f.data)  # @stepover
    slope_a = (L2 - L1) / h
    text(f"The estimate is {slope_a:.4f}, which agrees with {a.grad:g}.")

    text(
        "Gradients are useful because they tell us which way to push each "
        "input to change `L`. To make `L` bigger, move every input a small "
        "step in the direction of its gradient:"
    )
    step = 0.01
    a.data += step * a.grad
    b.data += step * b.grad
    c.data += step * c.grad
    f.data += step * f.grad
    L_new = L_from(a.data, b.data, c.data, f.data)  # @stepover
    text(
        f"`L` went from {L1:g} to {L_new:g}. It went up, as promised. Training "
        "a network is the same move with the sign flipped: we push the weights "
        "*against* their gradients to make the loss go *down*."
    )


def L_from(a, b, c, f):
    return (a*b + c) * f


# ---------------------------------------------------------------------------
# 5


def local_derivatives():
    text("## 5. Every operation only needs its local derivative")
    text(
        "Look back at what we did. At each node we only ever needed two "
        "things: the gradient arriving from above, and the derivative of that "
        "one operation. So each kind of operation only has to know its own "
        r"local derivative. With $out$ as the result and $g$ as `out.grad`:"
    )
    table(
        [
            {"operation": "out = a + b", "local derivative": "∂out/∂a = 1, ∂out/∂b = 1", "a.grad +=": "g"},
            {"operation": "out = a * b", "local derivative": "∂out/∂a = b, ∂out/∂b = a", "a.grad +=": "b · g"},
            {"operation": "out = a ** k", "local derivative": "∂out/∂a = k · a^(k-1)", "a.grad +=": "k · a^(k-1) · g"},
            {"operation": "out = relu(a)", "local derivative": "1 if a > 0, else 0", "a.grad +=": "(out > 0) · g"},
        ],
        caption="The local rules micrograd uses",
    )
    text(
        r"ReLU is short for *rectified linear unit*, $\mathrm{ReLU}(z) = \max(0, z)$. "
        "It passes positive numbers through and turns negative ones into 0. Its "
        "slope is 1 on the right and 0 on the left."
    )
    zs = np.arange(-5, 5, 0.2)
    plot({
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "data": {"values": [{"z": float(z_), "ReLU(z)": float(r_)} for z_, r_ in zip(zs, np.maximum(0, zs))]},
        "mark": "line",
        "encoding": {
            "x": {"field": "z", "type": "quantitative"},
            "y": {"field": "ReLU(z)", "type": "quantitative"},
        },
        "width": 380,
        "height": 160,
    })
    text(
        "In the `Value` class, each operation creates its output and then "
        "attaches a `_backward` function that applies exactly one row of this "
        "table. Here is `__mul__`:"
    )
    text(
        """
        def __mul__(self, other):
          out = Value(self.data * other.data, (self, other), '*')

          def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
          out._backward = _backward

          return out
        """,
        language="python",
    )
    text(
        "The closure remembers `self`, `other` and `out`, so later, when the "
        "gradient of `out` is known, calling `out._backward()` pushes it one "
        "step further back. The `+=` instead of `=` matters, and section 7 shows why."
    )


# ---------------------------------------------------------------------------
# 6


def backward_does_it_for_you():
    text("## 6. Letting `backward()` do it")
    text(
        "Doing this by hand only needs one rule: a node's `_backward` must run "
        "*after* its own gradient is complete, which means after every node "
        "that uses it. An ordering with that property is a **topological "
        "order**. `backward()` builds one by walking `_prev` from the output, "
        "then calls every `_backward` in reverse."
    )
    a = Value(2.0, label='a')  # @stepover
    b = Value(-3.0, label='b')  # @stepover
    c = Value(10.0, label='c')  # @stepover
    e = a*b; e.label = 'e'  # @stepover
    d = e + c; d.label = 'd'  # @stepover
    f = Value(-2.0, label='f')  # @stepover
    L = d * f; L.label = 'L'  # @stepover
    text("Step into this line. Watch `topo` fill up, then watch the grads appear one closure at a time.")
    L.backward()
    text(
        f"Same answers as by hand: `a.grad` = {a.grad:g}, `b.grad` = {b.grad:g}, "
        f"`c.grad` = {c.grad:g}, `f.grad` = {f.grad:g}. The by-hand version took "
        "a line per gradient. This one takes one call, no matter how big the "
        "graph is."
    )


# ---------------------------------------------------------------------------
# 7


def why_gradients_add_up():
    text("## 7. Why gradients add up")
    text(
        "In the video, the first version of `__add__` set gradients with `=`. "
        "It worked on every example until this one:"
    )
    text(r"$$b = a + a = 2a, \qquad \frac{\partial b}{\partial a} = 2$$")
    text("`ValueWithBug` is that first version. Watch what happens.")
    a = ValueWithBug(3.0, label='a')
    b = a + a; b.label = 'b'
    b.backward()
    text(
        f"`a.grad` is {a.grad:g}, not 2. Both lines of the add's `_backward` "
        "write to the same object, because `self` and `other` are both `a`. "
        "With `=` the second write replaces the first."
    )
    a = Value(3.0, label='a')  # @stepover
    b = a + a; b.label = 'b'  # @stepover
    b.backward()  # @stepover
    text(f"With `+=`, as in the real class, `a.grad` is {a.grad:g}.")
    text(
        "This is the **multivariable chain rule**. When a value feeds into "
        r"several places $u_1, u_2, \dots$, its gradient is the *sum* of what "
        "comes back from each of them:"
    )
    text(r"$$\frac{\partial L}{\partial a} = \sum_i \frac{\partial L}{\partial u_i} \cdot \frac{\partial u_i}{\partial a}$$")
    text(
        "In a neural network almost every value is used more than once. Each "
        "input goes to every neuron in the first layer, and each neuron's output "
        "goes to every neuron in the next. So `+=` is not a detail. Without "
        "it, nothing would learn correctly."
    )


# ---------------------------------------------------------------------------
# 8


def a_neuron():
    text("## 8. A neuron")
    text(
        "A neuron takes some inputs $x_i$, multiplies each by a weight $w_i$, "
        "adds a bias $b$, and passes the sum through an activation function. "
        "Here the activation is ReLU:"
    )
    text(r"$$n = \sum_i w_i x_i + b, \qquad o = \mathrm{ReLU}(n) = \max(0, n)$$")
    text(
        "The weights say how much each input matters, and the bias shifts the "
        "point where the neuron switches on. This is the neuron from the video, "
        "with two inputs."
    )
    x1 = Value(2.0, label='x1')  # @stepover
    x2 = Value(0.0, label='x2')  # @stepover
    w1 = Value(-3.0, label='w1')  # @stepover
    w2 = Value(1.0, label='w2')  # @stepover
    b = Value(6.8813735870195432, label='b')  # @stepover
    x1w1 = x1*w1; x1w1.label = 'x1*w1'  # @stepover
    x2w2 = x2*w2; x2w2.label = 'x2*w2'  # @stepover
    x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'  # @stepover
    n = x1w1x2w2 + b; n.label = 'n'  # @stepover
    o = n.relu(); o.label = 'o'
    text(
        f"$n = ({w1.data:g})({x1.data:g}) + ({w2.data:g})({x2.data:g}) + {b.data:.4f} = {n.data:.4f}$. "
        "That is positive, so ReLU lets it through and "
        f"$o = {o.data:.4f}$."
    )
    note(
        "The odd bias is from the video, where Karpathy picked it so that tanh(n) would "
        "come out at 0.7071. With ReLU there is no squashing, so o is just n."
    )
    o.backward()  # @stepover
    text(
        r"Through the chain rule, $\frac{\partial o}{\partial n} = 1$ because $n > 0$, "
        r"and then $\frac{\partial o}{\partial w_i} = x_i$ and "
        r"$\frac{\partial o}{\partial x_i} = w_i$. Check those against the panel: "
        f"`w1.grad` = {w1.grad:g}, which is `x1`. `x1.grad` = {x1.grad:g}, which "
        f"is `w1`. `b.grad` = {b.grad:g}."
    )
    text(
        f"And `w2.grad` = {w2.grad:g}. The weight is fine, but its input `x2` is "
        "0, and the gradient of a weight is its input. When an input is zero, "
        "changing its weight cannot change the output, so the weight gets no "
        "signal to learn from on this example."
    )
    text("### When ReLU switches off")
    text(
        r"The other side of ReLU: if $n < 0$, the output is 0 and its slope is "
        "0. Here is the same neuron with the bias set to 5 instead."
    )
    off = neuron(5.0)  # @stepover
    text(
        f"Now `n` = {off['n'].data:g}, so `o` = {off['o'].data:g}, and every "
        f"gradient is 0: `w1.grad` = {off['w1'].grad:g}, `b.grad` = {off['b'].grad:g}. "
        "The gradient arrives at `o` and stops there, because ReLU multiplies it "
        "by its slope, which is 0. A neuron stuck like this for every input "
        "never learns again. That is called a **dead ReLU**, and it comes back "
        "in section 11."
    )


def neuron(bias):
    x1 = Value(2.0, label='x1')
    x2 = Value(0.0, label='x2')
    w1 = Value(-3.0, label='w1')
    w2 = Value(1.0, label='w2')
    b = Value(bias, label='b')
    x1w1 = x1*w1; x1w1.label = 'x1*w1'
    x2w2 = x2*w2; x2w2.label = 'x2*w2'
    x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
    n = x1w1x2w2 + b; n.label = 'n'
    o = n.relu(); o.label = 'o'
    o.backward()
    return locals()


# ---------------------------------------------------------------------------
# 9


def a_network():
    text("## 9. From one neuron to a network")
    text(
        "A **layer** is several neurons that all see the same inputs. An "
        "**MLP** (multi-layer perceptron) is layers in a row, each one feeding "
        "the next. The video builds `MLP(3, [4, 4, 1])`: 3 inputs, two hidden "
        "layers of 4 neurons, and 1 output. Written as math, with $W$ the "
        "weights of a layer and $\\mathbf{b}$ its biases:"
    )
    text(
        r"$$\mathbf{h}_1 = \mathrm{ReLU}(W_1 \mathbf{x} + \mathbf{b}_1), \quad "
        r"\mathbf{h}_2 = \mathrm{ReLU}(W_2 \mathbf{h}_1 + \mathbf{b}_2), \quad "
        r"\hat{y} = W_3 \mathbf{h}_2 + \mathbf{b}_3$$"
    )
    text(
        "The last layer has no ReLU, as in micrograd's `nn.py`. The targets "
        "here go down to -1, and a ReLU can never output a negative number."
    )
    link(Neuron)
    link(MLP)
    random.seed(5)
    model = MLP(3, [4, 4, 1])  # @stepover
    params = len(model.parameters())  # @stepover
    text(
        f"That is {params} parameters: each neuron has one weight per input plus a "
        "bias, so 4·(3+1) + 4·(4+1) + 1·(4+1) = 16 + 20 + 5. They start as random "
        "numbers between -1 and 1."
    )
    x = [2.0, 3.0, -1.0]
    out = model(x)  # @stepover
    text(
        f"Feeding in x = {x} gives {out.data:.4f}. Under the hood that is one "
        "big expression graph made of the same `+`, `*` and ReLU nodes as "
        "before, just many more of them. So `backward()` works on it unchanged."
    )


# ---------------------------------------------------------------------------
# 10


def loss_and_one_step_of_learning():
    text("## 10. A loss, and one step of learning")
    text(
        "To train, we need a single number that says how wrong the network is. "
        "That number is the **loss**. With four examples and their targets, the "
        "video uses the squared error, summed over the examples:"
    )
    text(r"$$L = \sum_{j=1}^{4} (\hat{y}_j - y_j)^2$$")
    text(
        "Squaring makes every error positive and punishes big misses more than "
        "small ones. The loss is 0 only when every prediction hits its target."
    )
    random.seed(5)
    model = MLP(3, [4, 4, 1])  # @stepover
    xs = [
      [2.0, 3.0, -1.0],
      [3.0, -1.0, 0.5],
      [0.5, 1.0, 1.0],
      [1.0, 1.0, -1.0],
    ]
    ys = [1.0, -1.0, -1.0, 1.0]
    ypred = list(model(x) for x in xs)  # @stepover
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))  # @stepover
    table(
        [{"target": y, "prediction": round(p.data, 4)} for y, p in zip(ys, ypred)],
        caption="Before training",
    )
    text(f"The loss is {loss.data:.4f}. Now one step of **gradient descent**:")
    text(r"$$\theta \leftarrow \theta - \eta \, \frac{\partial L}{\partial \theta}$$")
    text(
        r"Every parameter $\theta$ moves a small step against its gradient. "
        r"$\eta$ is the **learning rate**, the size of the step."
    )
    text(
        "First, zero every gradient. `_backward` uses `+=`, so without this the "
        "new gradients would pile on top of the old ones from the last step. "
        "Forgetting this line is one of the most common bugs in training code."
    )
    for p in model.parameters(): p.grad = 0.0  # @stepover
    loss.backward()  # @stepover
    w = model.layers[0].neurons[0].w[0]
    params = len(model.parameters())  # @stepover
    text(
        f"After `backward()`, all {params} parameters have a "
        f"gradient. For example, the first weight of the first neuron has "
        f"data {w.data:.4f} and grad {w.grad:.4f}."
    )
    lr = 0.05
    for p in model.parameters(): p.data += -lr * p.grad  # @stepover
    ypred = list(model(x) for x in xs)  # @stepover
    new_loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))  # @stepover
    text(
        f"That weight is now {w.data:.4f}. Running the forward pass again, the "
        f"loss went from {loss.data:.4f} to {new_loss.data:.4f}. One step, and the "
        "network is already much less wrong. It is not always this smooth, though. "
        "The step is big enough that on the next one the loss jumps back up before it "
        "settles, which you can see at the start of the plot in section 11."
    )


# ---------------------------------------------------------------------------
# 11


def training():
    text("## 11. Training")
    text(
        "Training is that step in a loop: forward pass, zero the grads, "
        "backward pass, update. Here it runs for 30 steps."
    )
    text(
        """
        for k in range(steps):
            ypred = [model(x) for x in xs]                              # forward
            loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
            for p in model.parameters():                                # zero grads
                p.grad = 0.0
            loss.backward()                                             # backward
            for p in model.parameters():                                # update
                p.data += -lr * p.grad
        """,
        language="python",
    )
    good = untraced(train, seed=5, steps=30, lr=0.05)
    note(
        "Training runs with the tracer switched off (untraced, in _util.py). It is tens of "
        "thousands of lines, so stepping through it is not useful, and tracing it would make "
        "the build slow."
    )
    plot({
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "data": {"values": good["history"]},
        "mark": {"type": "line", "point": True},
        "encoding": {
            "x": {"field": "step", "type": "quantitative"},
            "y": {"field": "loss", "type": "quantitative", "scale": {"type": "log"}},
        },
        "width": 420,
        "height": 220,
    })
    table(
        [{"target": y, "prediction": round(p, 3)} for y, p in zip(good["ys"], good["ypred"])],
        caption="After training",
    )
    text(
        f"The loss went from {good['history'][0]['loss']:.4g} to "
        f"{good['history'][-1]['loss']:.2g}, and the predictions are close to "
        "the targets. Nobody wrote the rule that maps these inputs to these "
        f"outputs. Gradient descent found {good['params']} numbers that do it."
    )

    text("### When training goes wrong")
    text(
        "Run the exact same code from a different random start and it can fail "
        "without any error at all."
    )
    bad = untraced(train, seed=2, steps=30, lr=0.05)
    text(
        f"With seed 2 the loss starts at {bad['history'][0]['loss']:.4g} and gets "
        f"stuck at {bad['history'][-1]['loss']:.4g}. Every prediction is within "
        f"{max(abs(p) for p in bad['ypred']):.0e} of 0. The cause is the dead "
        f"ReLU from section 8: {bad['dead'][1]} of the 4 neurons in the second "
        "hidden layer output 0 for every training example. The output layer "
        "only ever sees zeros, so it can only return its bias. And since ReLU's "
        "slope is 0 there, no gradient gets back through to fix the layers below."
    )
    if any(good["dead"]):
        text(
            f"Even the run that worked has {sum(good['dead'])} of its 8 hidden "
            "neurons dead by the end. It fits the data with the ones that are left. "
            "Tiny ReLU networks are fragile like this. Bigger networks, careful "
            "initialization and variants like Leaky ReLU are the usual fixes."
        )


def train(seed, steps, lr):
    random.seed(seed)
    xs = [
      [2.0, 3.0, -1.0],
      [3.0, -1.0, 0.5],
      [0.5, 1.0, 1.0],
      [1.0, 1.0, -1.0],
    ]
    ys = [1.0, -1.0, -1.0, 1.0] # desired targets
    model = MLP(3, [4, 4, 1])
    history = []
    for k in range(steps):
        # forward pass
        ypred = [model(x) for x in xs]
        loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
        history.append({"step": k, "loss": loss.data})

        # backward pass
        for p in model.parameters():
            p.grad = 0.0
        loss.backward()

        # update
        if k < steps - 1:
            for p in model.parameters():
                p.data += -lr * p.grad

    # dead[i] counts the units in hidden layer i that output 0 on every input
    acts = []
    for x in xs:
        h, per_layer = x, []
        for layer in model.layers[:-1]:
            h = layer(h)
            per_layer.append([v.data for v in h])
        acts.append(per_layer)
    dead = [
        sum(1 for j in range(len(layer.neurons)) if all(a[i][j] == 0 for a in acts))
        for i, layer in enumerate(model.layers[:-1])
    ]
    return {
        "history": history,
        "ys": ys,
        "ypred": [y.data for y in ypred],
        "params": len(model.parameters()),
        "dead": dead,
    }


# ---------------------------------------------------------------------------
# The Value class from Karpathy's micrograd lecture notebook (nn-zero-to-hero,
# lectures/micrograd, MIT license). Two changes: tanh and exp are replaced by
# relu from micrograd/engine.py, and __lectrace__ is added, which only changes
# how the viewer displays a Value.


class Value:

  def __init__(self, data, _children=(), _op='', label=''):
    self.data = data
    self.grad = 0.0
    self._backward = lambda: None
    self._prev = set(_children)
    self._op = _op
    self.label = label

  def __repr__(self):
    return f"Value(data={self.data})"

  def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data + other.data, (self, other), '+')

    def _backward():
      self.grad += 1.0 * out.grad
      other.grad += 1.0 * out.grad
    out._backward = _backward

    return out

  def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self, other), '*')

    def _backward():
      self.grad += other.data * out.grad
      other.grad += self.data * out.grad
    out._backward = _backward

    return out

  def __pow__(self, other):
    assert isinstance(other, (int, float)), "only supporting int/float powers for now"
    out = Value(self.data**other, (self,), f'**{other}')

    def _backward():
        self.grad += other * (self.data ** (other - 1)) * out.grad
    out._backward = _backward

    return out

  def __rmul__(self, other): # other * self
    return self * other

  def __truediv__(self, other): # self / other
    return self * other**-1

  def __neg__(self): # -self
    return self * -1

  def __sub__(self, other): # self - other
    return self + (-other)

  def __radd__(self, other): # other + self
    return self + other

  def relu(self):
    out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

    def _backward():
      self.grad += (out.data > 0) * out.grad
    out._backward = _backward

    return out

  def backward(self):

    topo = []
    visited = set()
    def build_topo(v):
      if v not in visited:
        visited.add(v)
        for child in v._prev:
          build_topo(child)
        topo.append(v)
    build_topo(self)

    self.grad = 1.0
    for node in reversed(topo):
      node._backward()

  def __lectrace__(self):
    # getattr, because the viewer also shows `self` halfway through __init__
    return {
      "label": getattr(self, "label", ""),
      "data": getattr(self, "data", None),
      "grad": getattr(self, "grad", None),
      "op": getattr(self, "_op", ""),
      # Children by label rather than as Values, so the panel shows one level
      # of the graph instead of unfolding all of it. Sorted, because _prev is
      # a set and its order changes between runs.
      "children": sorted(c.label or f"{c.data:.4g}" for c in getattr(self, "_prev", ())),
    }


class ValueWithBug(Value):
  """__add__ as it was first written in the video, with = where += belongs."""

  def __add__(self, other):
    out = Value(self.data + other.data, (self, other), '+')

    def _backward():
      self.grad = 1.0 * out.grad
      other.grad = 1.0 * out.grad
    out._backward = _backward

    return out


# The MLP from the same notebook. As in micrograd/nn.py, the last layer is
# linear (nonlin=False), since a ReLU output can never go negative.


class Neuron:

  def __init__(self, nin, nonlin=True):
    self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
    self.b = Value(random.uniform(-1,1))
    self.nonlin = nonlin

  def __call__(self, x):
    # w * x + b
    act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
    return act.relu() if self.nonlin else act

  def parameters(self):
    return self.w + [self.b]

class Layer:

  def __init__(self, nin, nout, **kwargs):
    self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

  def __call__(self, x):
    outs = [n(x) for n in self.neurons]
    return outs[0] if len(outs) == 1 else outs

  def parameters(self):
    return [p for neuron in self.neurons for p in neuron.parameters()]

class MLP:

  def __init__(self, nin, nouts):
    sz = [nin] + nouts
    self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    return x

  def parameters(self):
    return [p for layer in self.layers for p in layer.parameters()]


if __name__ == "__main__":
    main()
