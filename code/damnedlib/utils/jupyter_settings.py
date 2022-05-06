import matplotlib.pyplot as plt
import numpy as np

# use standard white background blue line theme. A bit less pretty in the dark notebooks
# but definetively better for saving and reusing the figures
plt.style.use("default")
# Use tex for rendering font
plt.rc("font", family="serif")
plt.rc("text", usetex=True)
# Unified figure size and resolution.
plt.rc("figure", figsize=(8, 4.5), max_open_warning=0, dpi=200, autolayout=True)
# Misc. Settings
plt.rc("axes", axisbelow=True, grid=True, xmargin=0)
plt.rc("grid", linestyle="solid")

# make sure that we got a fixed random seed per default.
np.random.seed(42)
