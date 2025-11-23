# gdsp_delay_multitap_comb
```python
"""
gdsp_delay_linear_multitap_comb.py

MODULE NAME:
gdsp_delay_comb

DESCRIPTION:
Fully differentiable, JAX-based delay and comb filter primitives in the GDSP
core style. This module implements:

A) A basic linear-interpolated delay line (fractional delay).
B) A delay line with optional wet/dry mixing.
C) A true delay-based feedback comb filter with linear interpolation.
D) A multitap fractional delay line with linear interpolation on each tap.

All units are:
- Purely functional (no side effects, no mutation).
- JAX-jittable and differentiable end-to-end.
- Implemented with explicit state tuples.
- Using lax.scan for block processing.

INPUTS (per tick):
- x : scalar sample (jnp.ndarray scalar) to process.

PARAMS (examples, exact layout documented per unit below):
- delay_samples_target : Desired delay in samples (can be fractional).
- fbk_target           : Target feedback gain (comb).
- ffd_target           : Target feedforward gain (comb).
- wet_target           : Target wet mix amount [0, 1] (wet/dry delay).
- dry_target           : Target dry mix amount [0, 1] (wet/dry delay).
- smooth_coeff         : One-pole smoothing coefficient in (0, 1).

OUTPUTS:
- y : processed sample for that unit.

STATE VARIABLES (examples, exact layout documented per unit below):
For all units, the delay buffer is a 1D array of shape (buffer_len,).
Indices are scalar integers treated modulo buffer_len.

A) Core fractional delay state:
    state_delay = (
        buffer,          # [buffer_len] ring buffer of past samples
        write_idx,       # scalar int32 write index
        delay_smooth,    # scalar float32, smoothed delay in samples
    )

B) Wet/dry delay:
    state_delay_wet = (
        buffer,
        write_idx,
        delay_smooth,
        wet_smooth,
        dry_smooth,
    )

C) Feedback comb:
    state_comb = (
        buffer,
        write_idx,
        delay_smooth,
        fbk_smooth,
        ffd_smooth,
    )

D) Multitap fractional delay:
    state_multitap = (
        buffer,
        write_idx,
        delays_smooth,  # [num_taps] float32 smoothed delays in samples
    )

EQUATIONS / MATH:

Basic linear fractional delay:
Let N be buffer_len, and b[k] the buffer content at index k (0 <= k < N).
At time n, write_idx points to the location where x[n] will be written
after reading the delayed sample.

Given a (possibly time-varying) delay in samples d[n] >= 0:

    i[n]   = floor(d[n])
    α[n]   = d[n] - i[n]  in [0, 1)
    k0[n]  = (write_idx[n] - i[n]) mod N
    k1[n]  = (write_idx[n] - i[n] - 1) mod N

    y_delay[n] = (1 - α[n]) * b[k0[n]] + α[n] * b[k1[n]]

Then we write the current input (or some internal t[n]) into the buffer:

    b_next[write_idx[n]] = value_to_write[n]
    write_idx[n+1]       = (write_idx[n] + 1) mod N

Integer delay only:
If d[n] is integer, then α[n] = 0, so

    y_delay[n] = b[k0[n]]

This reduces to an integer delay with no interpolation coloration.

Parameter smoothing (one-pole per-sample smoothing):
For any parameter p[n] with target p_target[n] and state p_smooth[n]:

    p_smooth[n+1] = p_smooth[n] + smooth_coeff * (p_target[n] - p_smooth[n])

where smooth_coeff is in (0, 1). Small coeff = slow smoothing,
large coeff = fast smoothing.

B) Wet/dry mix delay:
Let x[n] be the dry signal and y_delay[n] the delayed signal.

    y[n] = dry[n] * x[n] + wet[n] * y_delay[n]

dry[n], wet[n] are themselves smoothed versions of their targets.

C) Linear-interpolated feedback comb filter (true delay-based comb):
We use the classic Gamma comb structure:

    H(z) = (ffd + z^{-m}) / (1 - fbk z^{-m})
    y[n] = ffd * x[n] + x[n - m] + fbk * y[n - m]

Implementation with an internal delay line that stores:

    t[n] = x[n] + fbk * y[n]

At each sample:

    oN[n]          = delayed value read from buffer (linear fractional delay)
                     (this represents y[n - m] + ... internal structure)
    t[n]           = x[n] + fbk[n] * oN[n]
    store t[n] in the delay line
    y[n]           = oN[n] + ffd[n] * t[n]

This matches the Gamma pattern:
    t = i0 + oN * fbk
    write(t)
    return oN + t * ffd

D) Multitap fractional delay:
Given K taps with delay vectors d_k[n], k = 0..K-1:

For each tap k:
    i_k[n]   = floor(d_k[n])
    α_k[n]   = d_k[n] - i_k[n]
    k0_k[n]  = (write_idx[n] - i_k[n]) mod N
    k1_k[n]  = (write_idx[n] - i_k[n] - 1) mod N

    y_k[n]   = (1 - α_k[n]) * b[k0_k[n]] + α_k[n] * b[k1_k[n]]

All taps share the same write head and buffer.

Through-zero / wrapping rules:
- Delay amount is clamped to [0, N-1-ε] for stability.
- Indices are wrapped with modulo N:
        k = (index) mod N
- write_idx increments modulo N:

        write_idx[n+1] = (write_idx[n] + 1) mod N

Nonlinearities:
- None: all filters are linear in x for fixed parameters.
- Parameter smoothing is linear.

Interpolation rules:
- Linear interpolation in the delay domain as above.

Time-varying coefficient rules:
- All coefficients (delay, fbk, ffd, wet, dry, multitap delays) are smoothed
  per-sample using the one-pole rule.

NOTES:
- For stability of the comb filter, |fbk| < 1 is recommended.
- For wet/dry mixing, using dry + wet <= 1 avoids clipping, but is not required.
- Max delay in samples must be less than buffer_len.
- All processing is implemented with lax.scan, no Python loops inside jitted code.
"""

from typing import Tuple

import jax
import jax.numpy as jnp
from jax import lax


# ---------------------------------------------------------------------------
# Utility helpers (pure JAX, jittable)
# ---------------------------------------------------------------------------


def _one_pole_smooth(current: jnp.ndarray,
                     target: jnp.ndarray,
                     smooth_coeff: jnp.ndarray) -> jnp.ndarray:
    """One-pole smoothing step for parameters.

    p_next = p + a * (target - p)
    """
    return current + smooth_coeff * (target - current)


def _wrap_index(idx: jnp.ndarray, length: int) -> jnp.ndarray:
    """Wrap a scalar index modulo length."""
    length_j = jnp.asarray(length, dtype=jnp.int32)
    idx = jnp.asarray(idx, dtype=jnp.int32)
    return jnp.mod(idx, length_j)


def _read_linear_delay(buffer: jnp.ndarray,
                       write_idx: jnp.ndarray,
                       delay_samples: jnp.ndarray) -> jnp.ndarray:
    """Read from delay buffer with linear interpolation.

    buffer: [N]
    write_idx: scalar int
    delay_samples: scalar float, >= 0
    """
    N = buffer.shape[0]
    # Clamp delay to valid range [0, N-1-eps]
    max_delay = jnp.asarray(N - 1, dtype=jnp.float32) - 1e-3
    d = jnp.clip(delay_samples, 0.0, max_delay)

    i = jnp.floor(d).astype(jnp.int32)
    frac = d - i.astype(d.dtype)

    idx0 = _wrap_index(write_idx - i, N)
    idx1 = _wrap_index(write_idx - i - 1, N)

    # dynamic_slice returns length-1 vector; take [0]
    y0 = lax.dynamic_slice(buffer, (idx0,), (1,))[0]
    y1 = lax.dynamic_slice(buffer, (idx1,), (1,))[0]

    return y0 * (1.0 - frac) + y1 * frac


def _write_buffer(buffer: jnp.ndarray,
                  write_idx: jnp.ndarray,
                  value: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Write a single sample into the ring buffer and increment write index.

    Returns (buffer_next, write_idx_next).
    """
    N = buffer.shape[0]
    # Ensure scalar shapes
    value_vec = jnp.reshape(value, (1,))
    write_idx_i = jnp.asarray(write_idx, dtype=jnp.int32)

    buffer_next = lax.dynamic_update_slice(buffer, value_vec, (write_idx_i,))
    write_idx_next = _wrap_index(write_idx_i + 1, N)
    return buffer_next, write_idx_next


def _read_linear_delay_vector(buffer: jnp.ndarray,
                              write_idx: jnp.ndarray,
                              delay_samples_vec: jnp.ndarray) -> jnp.ndarray:
    """Vectorized linear fractional delay for multitap.

    buffer: [N]
    write_idx: scalar int
    delay_samples_vec: [K] float
    returns: [K] delayed samples
    """

    def _read_one(d):
        return _read_linear_delay(buffer, write_idx, d)

    return jax.vmap(_read_one)(delay_samples_vec)


# ---------------------------------------------------------------------------
# A) Core fractional delay line (not exposed directly, but used below)
# ---------------------------------------------------------------------------


def delay_init(max_delay_samples: int) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Initialize core fractional delay state.

    Args:
        max_delay_samples: int, buffer length in samples.

    Returns:
        state_delay = (buffer, write_idx, delay_smooth)
    """
    buffer = jnp.zeros((max_delay_samples,), dtype=jnp.float32)
    write_idx = jnp.asarray(0, dtype=jnp.int32)
    delay_smooth = jnp.asarray(0.0, dtype=jnp.float32)
    return buffer, write_idx, delay_smooth


def delay_update_state(state_delay, delay_target, smooth_coeff):
    """Update smoothed delay without processing a sample.

    This can be used at block rate if desired.
    """
    buffer, write_idx, delay_smooth = state_delay
    delay_next = _one_pole_smooth(delay_smooth, delay_target, smooth_coeff)
    return buffer, write_idx, delay_next


@jax.jit
def delay_tick(x, state_delay, params):
    """Core fractional delay tick (no mixing, just delayed signal out).

    Args:
        x: scalar sample input.
        state_delay: (buffer, write_idx, delay_smooth)
        params: (delay_target, smooth_coeff)

    Returns:
        (y_delay, new_state_delay)
    """
    buffer, write_idx, delay_smooth = state_delay
    delay_target, smooth_coeff = params

    delay_smooth_next = _one_pole_smooth(delay_smooth, delay_target, smooth_coeff)

    y_delay = _read_linear_delay(buffer, write_idx, delay_smooth_next)
    buffer_next, write_idx_next = _write_buffer(buffer, write_idx, x)

    new_state = (buffer_next, write_idx_next, delay_smooth_next)
    return y_delay, new_state


def delay_process(x_series, state_delay, params):
    """Process a sequence with the core fractional delay.

    Args:
        x_series: [T] input samples
        state_delay: (buffer, write_idx, delay_smooth)
        params: (delay_target, smooth_coeff)

    Returns:
        (y_series, final_state)
    """

    def _scan_fn(carry, x_t):
        y_t, new_state = delay_tick(x_t, carry, params)
        return new_state, y_t

    final_state, y_series = lax.scan(_scan_fn, state_delay, x_series)
    return y_series, final_state


# ---------------------------------------------------------------------------
# B) Delay with wet/dry mixing
# ---------------------------------------------------------------------------


def delay_wetdry_init(max_delay_samples: int) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Initialize wet/dry delay state.

    Returns:
        state_delay_wet = (buffer, write_idx, delay_smooth, wet_smooth, dry_smooth)
    """
    buffer = jnp.zeros((max_delay_samples,), dtype=jnp.float32)
    write_idx = jnp.asarray(0, dtype=jnp.int32)
    delay_smooth = jnp.asarray(0.0, dtype=jnp.float32)
    wet_smooth = jnp.asarray(0.0, dtype=jnp.float32)
    dry_smooth = jnp.asarray(1.0, dtype=jnp.float32)
    return buffer, write_idx, delay_smooth, wet_smooth, dry_smooth


def delay_wetdry_update_state(state, delay_target, wet_target, dry_target, smooth_coeff):
    """Block-rate parameter smoothing for wet/dry delay."""
    buffer, write_idx, delay_smooth, wet_smooth, dry_smooth = state

    delay_next = _one_pole_smooth(delay_smooth, delay_target, smooth_coeff)
    wet_next = _one_pole_smooth(wet_smooth, wet_target, smooth_coeff)
    dry_next = _one_pole_smooth(dry_smooth, dry_target, smooth_coeff)

    return buffer, write_idx, delay_next, wet_next, dry_next


@jax.jit
def delay_wetdry_tick(x, state, params):
    """Wet/dry fractional delay tick.

    Args:
        x: scalar sample
        state: (buffer, write_idx, delay_smooth, wet_smooth, dry_smooth)
        params: (delay_target, wet_target, dry_target, smooth_coeff)

    Returns:
        (y, new_state)
    """
    buffer, write_idx, delay_smooth, wet_smooth, dry_smooth = state
    delay_target, wet_target, dry_target, smooth_coeff = params

    delay_smooth_next = _one_pole_smooth(delay_smooth, delay_target, smooth_coeff)
    wet_smooth_next = _one_pole_smooth(wet_smooth, wet_target, smooth_coeff)
    dry_smooth_next = _one_pole_smooth(dry_smooth, dry_target, smooth_coeff)

    y_delay = _read_linear_delay(buffer, write_idx, delay_smooth_next)
    buffer_next, write_idx_next = _write_buffer(buffer, write_idx, x)

    y = dry_smooth_next * x + wet_smooth_next * y_delay

    new_state = (
        buffer_next,
        write_idx_next,
        delay_smooth_next,
        wet_smooth_next,
        dry_smooth_next,
    )
    return y, new_state


def delay_wetdry_process(x_series, state, params):
    """Process a sequence with wet/dry delay."""

    def _scan_fn(carry, x_t):
        y_t, new_state = delay_wetdry_tick(x_t, carry, params)
        return new_state, y_t

    final_state, y_series = lax.scan(_scan_fn, state, x_series)
    return y_series, final_state


# ---------------------------------------------------------------------------
# C) Linear-interpolated feedback comb filter (true delay-based comb)
# ---------------------------------------------------------------------------


def comb_init(max_delay_samples: int) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Initialize comb filter state.

    Returns:
        state_comb = (buffer, write_idx, delay_smooth, fbk_smooth, ffd_smooth)
    """
    buffer = jnp.zeros((max_delay_samples,), dtype=jnp.float32)
    write_idx = jnp.asarray(0, dtype=jnp.int32)
    delay_smooth = jnp.asarray(1.0, dtype=jnp.float32)
    fbk_smooth = jnp.asarray(0.0, dtype=jnp.float32)
    ffd_smooth = jnp.asarray(0.0, dtype=jnp.float32)
    return buffer, write_idx, delay_smooth, fbk_smooth, ffd_smooth


def comb_update_state(state, delay_target, fbk_target, ffd_target, smooth_coeff):
    """Block-rate smoothing for comb filter parameters."""
    buffer, write_idx, delay_smooth, fbk_smooth, ffd_smooth = state

    delay_next = _one_pole_smooth(delay_smooth, delay_target, smooth_coeff)
    fbk_next = _one_pole_smooth(fbk_smooth, fbk_target, smooth_coeff)
    ffd_next = _one_pole_smooth(ffd_smooth, ffd_target, smooth_coeff)

    return buffer, write_idx, delay_next, fbk_next, ffd_next


@jax.jit
def comb_tick(x, state, params):
    """Feedback comb filter tick with fractional delay.

    Args:
        x: scalar input sample.
        state: (buffer, write_idx, delay_smooth, fbk_smooth, ffd_smooth)
        params: (delay_target, fbk_target, ffd_target, smooth_coeff)

    Returns:
        (y, new_state)
    """
    buffer, write_idx, delay_smooth, fbk_smooth, ffd_smooth = state
    delay_target, fbk_target, ffd_target, smooth_coeff = params

    delay_smooth_next = _one_pole_smooth(delay_smooth, delay_target, smooth_coeff)
    fbk_smooth_next = _one_pole_smooth(fbk_smooth, fbk_target, smooth_coeff)
    ffd_smooth_next = _one_pole_smooth(ffd_smooth, ffd_target, smooth_coeff)

    oN = _read_linear_delay(buffer, write_idx, delay_smooth_next)
    t = x + fbk_smooth_next * oN
    buffer_next, write_idx_next = _write_buffer(buffer, write_idx, t)
    y = oN + ffd_smooth_next * t

    new_state = (
        buffer_next,
        write_idx_next,
        delay_smooth_next,
        fbk_smooth_next,
        ffd_smooth_next,
    )
    return y, new_state


def comb_process(x_series, state, params):
    """Process a sequence with the feedback comb filter."""

    def _scan_fn(carry, x_t):
        y_t, new_state = comb_tick(x_t, carry, params)
        return new_state, y_t

    final_state, y_series = lax.scan(_scan_fn, state, x_series)
    return y_series, final_state


# ---------------------------------------------------------------------------
# D) Multitap fractional delay
# ---------------------------------------------------------------------------


def multitap_delay_init(max_delay_samples: int, num_taps: int):
    """Initialize multitap fractional delay state.

    Args:
        max_delay_samples: buffer length in samples.
        num_taps: number of simultaneous read taps.

    Returns:
        state_multitap = (buffer, write_idx, delays_smooth)
            buffer: [max_delay_samples]
            write_idx: scalar int32
            delays_smooth: [num_taps] float32
    """
    buffer = jnp.zeros((max_delay_samples,), dtype=jnp.float32)
    write_idx = jnp.asarray(0, dtype=jnp.int32)
    delays_smooth = jnp.zeros((num_taps,), dtype=jnp.float32)
    return buffer, write_idx, delays_smooth


def multitap_delay_update_state(state, delays_target, smooth_coeff):
    """Block-rate smoothing of multitap delays.

    Args:
        state: (buffer, write_idx, delays_smooth)
        delays_target: [num_taps] target delay samples
        smooth_coeff: scalar

    Returns:
        new_state
    """
    buffer, write_idx, delays_smooth = state
    delays_next = _one_pole_smooth(delays_smooth, delays_target, smooth_coeff)
    return buffer, write_idx, delays_next


@jax.jit
def multitap_delay_tick(x, state, params):
    """Multitap fractional delay tick.

    Args:
        x: scalar input sample.
        state: (buffer, write_idx, delays_smooth)
        params: (delays_target, smooth_coeff)
            delays_target: [num_taps]
            smooth_coeff: scalar

    Returns:
        (y_taps, new_state)
            y_taps: [num_taps] delayed samples
    """
    buffer, write_idx, delays_smooth = state
    delays_target, smooth_coeff = params

    delays_smooth_next = _one_pole_smooth(delays_smooth, delays_target, smooth_coeff)
    y_taps = _read_linear_delay_vector(buffer, write_idx, delays_smooth_next)

    buffer_next, write_idx_next = _write_buffer(buffer, write_idx, x)

    new_state = (buffer_next, write_idx_next, delays_smooth_next)
    return y_taps, new_state


def multitap_delay_process(x_series, state, params):
    """Process a sequence with multitap fractional delay.

    Args:
        x_series: [T] float
        state: (buffer, write_idx, delays_smooth)
        params: (delays_target, smooth_coeff)

    Returns:
        (y_series, final_state)
        y_series: [T, num_taps]
    """

    def _scan_fn(carry, x_t):
        y_t, new_state = multitap_delay_tick(x_t, carry, params)
        return new_state, y_t

    final_state, y_series = lax.scan(_scan_fn, state, x_series)
    return y_series, final_state


# ---------------------------------------------------------------------------
# Smoke test, plotting, and listening examples
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    try:
        import sounddevice as sd
        HAS_SD = True
    except Exception:
        HAS_SD = False

    sample_rate = 48000
    duration = 1.0
    t_np = np.arange(int(sample_rate * duration), dtype=np.float32) / sample_rate

    # Simple test signal: impulse followed by sine
    impulse = np.zeros_like(t_np)
    impulse[0] = 1.0
    sine = 0.5 * np.sin(2.0 * np.pi * 440.0 * t_np)
    x_np = impulse + sine
    x = jnp.asarray(x_np)

    # ---- Test wet/dry delay -----------------------------------------------
    max_delay_samples = int(0.5 * sample_rate)  # 0.5 seconds max
    static_delay_samples = jnp.asarray(0.25 * sample_rate, dtype=jnp.float32)  # 250 ms
    wet = jnp.asarray(0.5, dtype=jnp.float32)
    dry = jnp.asarray(1.0, dtype=jnp.float32)
    smooth = jnp.asarray(0.01, dtype=jnp.float32)

    state_wet = delay_wetdry_init(max_delay_samples)
    params_wet = (static_delay_samples, wet, dry, smooth)

    y_delay_wet, state_wet_final = delay_wetdry_process(x, state_wet, params_wet)
    y_delay_wet_np = np.array(y_delay_wet)

    # ---- Test comb filter --------------------------------------------------
    comb_delay_samples = jnp.asarray(0.02 * sample_rate, dtype=jnp.float32)  # 20 ms
    fbk = jnp.asarray(0.7, dtype=jnp.float32)
    ffd = jnp.asarray(0.0, dtype=jnp.float32)
    smooth_comb = jnp.asarray(0.01, dtype=jnp.float32)

    state_comb = comb_init(max_delay_samples)
    params_comb = (comb_delay_samples, fbk, ffd, smooth_comb)

    y_comb, state_comb_final = comb_process(x, state_comb, params_comb)
    y_comb_np = np.array(y_comb)

    # ---- Plot results ------------------------------------------------------
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.title("Wet/Dry Delay Output (first 1000 samples)")
    plt.plot(y_delay_wet_np[:1000], label="delay wet/dry")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.title("Comb Filter Output (first 1000 samples)")
    plt.plot(y_comb_np[:1000], label="comb", color="orange")
    plt.legend()

    plt.tight_layout()
    plt.show()

    # ---- Optional: Listen (if sounddevice is available) --------------------
    if HAS_SD:
        print("Playing wet/dry delay...")
        sd.play(y_delay_wet_np, sample_rate)
        sd.wait()
        print("Playing comb filter...")
        sd.play(y_comb_np, sample_rate)
        sd.wait()

    # -----------------------------------------------------------------------
    # Follow-up prompt suggestions (as comments)
    # -----------------------------------------------------------------------
    # Example follow-up prompts you might use with this module:
    #
    # 1) "Show me how to modulate the delay_wetdry delay time with a JAX LFO."
    # 2) "Extend the comb filter to support per-sample-varying fbk and ffd."
    # 3) "Add a tone control (one-pole LPF) inside the comb feedback loop."
    # 4) "Modify the multitap delay to support independent wet gains per tap."
    # 5) "Demonstrate differentiating a loss through comb_process for tuning."

```
