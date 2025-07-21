#!/usr/bin/env python3
"""Fully-determined (p = T) version of the signal-range evolution test.

Re-uses the logic from varied_signal_range_test but sets p = T (fully determined)
so that projection loss is zero.  We expect a much wider stability band.
"""

from varied_signal_range_test import analyze_signal_evolution

if __name__ == "__main__":
    # For k=10  ⇒  T = 55  ⇒  choose p = 55 (fully determined)
    analyze_signal_evolution(k=10, p=55, cycles=15)
