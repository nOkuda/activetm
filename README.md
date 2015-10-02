# ActiveTM

`ActiveT(opic)M(odeling)` is a framework for building experiments to measure the
effects of active learning on predictive topic models.

## Prerequisites

The Python code relies on ctypes to get reasonable computation speeds for
sampling based methods.  To compile the C code, make sure you have the following
programs and libraries installed:

* `gcc`
* `make`
* `LAPACK`
* `LAPACK` development headers

Then run `make` in the `sampler` directory.

To run the Python code, you will need to install the following modules:

* [`ankura`](https://github.com/jlund3/ankura)
* `ctypes`
* `numpy`

It is possible to install these modules by running

```
pip install -r requirements.txt
```

in the `ActiveTM` directory.
