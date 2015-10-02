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

Then run `make` in the `sampler` directory. If just running `make` didn't work,
you may need to change the value of `lapacke_headers_location` in the Makefile.
The default value is for a Fedora 22 system that had the `lapack-headers`
package installed.

To run the Python code, you will need to install the
[`ankura`](https://github.com/jlund3/ankura) and `ctypes` modules.  It is
possible to do this by running

```
pip install -r requirements.txt
```

in the `ActiveTM` directory.
