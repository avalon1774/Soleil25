
import logging
import numpy as np
import matplotlib.pyplot as plt


def test_plot():
    fig, ax = plt.subplots()
    ax.plot([1,2,3], [1,2,3])
    print('test')
    plt.show()# Clean up
    return fig

# Test 1: Direct plotting
plt.figure()
plt.plot([1,2,3], [1,2,3])
plt.show()
print("Direct plot done")

# Test 2: Function plotting
fig = test_plot()
print("Function plot done")