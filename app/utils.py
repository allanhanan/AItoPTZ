# app/utils.py

def clamp(val, minval, maxval):
    """Clamp val to the inclusive range [minval, maxval]."""
    return max(min(val, maxval), minval)

def smooth(current, target, factor):
    """Smoothly move current value towards target by the given factor (0..1)."""
    return current + (target - current) * factor

# Additional utilities can be added here as needed.

