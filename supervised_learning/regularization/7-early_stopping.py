#!/usr/bin/env python3
"""
module early_stopping
"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    determines if you should stop gradient descent early
    """
    if opt_cost - cost > threshold:
        count = 0
    else:
        count += 1
    return count >= patience, count
