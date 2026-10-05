# micrograd, step by step

A lectrace lecture that walks through Andrej Karpathy's micrograd, from what
a derivative is to training a small neural network. It follows his
[micrograd video](https://github.com/karpathy/nn-zero-to-hero/tree/master/lectures/micrograd)
from *Neural Networks: Zero to Hero* and uses the same code, with one change:
the activation is ReLU instead of tanh.

You step through the code line by line, watch the values and gradients
change in the variable panel, and read the math next to the code that
computes it.

**Live lecture:** <https://praisegee.github.io/micrograd-lectrace/>

## Running it

```sh
uv sync
uv run lectrace serve 01_micrograd.py   # opens the step-through viewer
```

`uv run python 01_micrograd.py` runs it as a plain script. To build a static
site, use `uv run lectrace build 01_micrograd.py -o _site`. Without uv,
`pip install lectrace` and drop the `uv run`.

In the viewer, `→` steps forward, `Shift+→` steps over a call, and `u` steps
out of the current function. Each section is its own function, so you can
step into one, or step over it to skip it.

## Deploying

`lectrace init` set up `.github/workflows/lectrace.yml`. Every push to
`main` builds the lecture and deploys it to GitHub Pages, so merging a pull
request is all it takes to update the live site. The site title is set in
`lectrace.toml`.

## What it covers

1. **What a derivative tells you.** $f(x) = 3x^2 - 4x + 5$, estimated
   numerically and checked against $f'(x) = 6x - 4$.
2. **Several inputs, one output.** Partial derivatives of $d = ab + c$.
3. **A number that remembers where it came from.** The `Value` class, and
   the expression $L = (ab + c)f$ as a graph.
4. **Backpropagation by hand.** The chain rule, one node at a time, checked
   numerically. Then one step uphill using the gradients.
5. **Local derivatives.** Why each operation only needs to know its own
   derivative, and how `_backward` implements that.
6. **`backward()`.** Topological order, and the same gradients in one call.
7. **Why gradients add up.** The `b = a + a` bug from the video, and the
   multivariable chain rule behind `+=`.
8. **A neuron.** $o = \mathrm{ReLU}(\sum_i w_i x_i + b)$, why `w2.grad` is
   0 when `x2` is 0, and what a dead ReLU is.
9. **A network.** `Neuron`, `Layer` and `MLP(3, [4, 4, 1])`, 41 parameters.
10. **A loss and one step of learning.** Squared error, zeroing gradients,
    and one gradient descent update.
11. **Training.** 30 steps, a loss plot, and a run from a different seed
    that fails silently because a whole layer of ReLUs is dead.

Every number in the text is filled in from the run with f-strings, so if you
change an input and rebuild, the explanation changes with it.

## Files

| File | What it is |
| --- | --- |
| `01_micrograd.py` | The lecture. The bottom half is the `Value` class and MLP from the video's notebook. |
| `_util.py` | `untraced()`, which runs the training loop with the tracer switched off. |

## Changes to Karpathy's code

- `tanh` and `exp` are gone, and `relu` from micrograd's `engine.py` takes
  their place.
- `__lectrace__` is added. It only changes how the viewer shows a `Value`:
  its label, data, grad, op, and `children`, the labels of the Values in
  `_prev`. Children are shown by label rather than as full Values, so the
  panel shows one level of the graph instead of unfolding all of it. It uses
  `getattr` because the viewer also shows `self` halfway through `__init__`,
  before `self.data` exists.
- numpy is used where the video uses it: `np.arange` and a vectorized `f` to
  plot $f(x)$, and `np.maximum(0, z)` to plot ReLU where the video plots
  `np.tanh`. The video draws these with matplotlib; here they are lectrace
  plots.
- `ValueWithBug` is added for section 7. Its `__add__` uses `=` where `+=`
  belongs, the way it was first written in the video.
- The MLP's last layer is linear (`nonlin=False`), as in micrograd's
  `nn.py`. With tanh the video could use the same activation everywhere, but
  a ReLU output can never reach the -1 targets.

## Notes on lectrace

- **Training runs untraced.** `# @stepover` hides a call from the viewer,
  but the tracer still runs its hook on every line, and training is tens of
  thousands of lines. `untraced()` switches the tracer off for the call.
- **`@stepover` and list comprehensions.** Since Python 3.12, list
  comprehensions run in the enclosing frame, so a line like
  `ypred = [model(x) for x in xs]  # @stepover` fires a line event on every
  iteration, and the stepover switches on and off each time. That line
  produced over 4,000 steps. The lecture writes it as
  `list(model(x) for x in xs)`, which runs in its own frame and does not
  have the problem.
- **Editing `_util.py` while `lectrace serve` is running.** The server keeps
  helper modules loaded between rebuilds and only watches the lecture file.
  After changing a helper, restart the server and run
  `lectrace build --no-incremental` once.

## Credits

The `Value` class and MLP are from Andrej Karpathy's nn-zero-to-hero and
micrograd repositories, MIT license. lectrace is at
<https://praisegee.github.io/lectrace/>.
