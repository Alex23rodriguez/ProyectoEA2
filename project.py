# %%
from linear_model import LM
from matplotlib import pyplot
import linear_model
import pandas as pd
from importlib import reload
reload(linear_model)

# %%
df = pd.read_csv('airfoil_self_noise.dat', sep='\t')
df.columns = ['freq', 'angle', 'chord_length',
              'free_stream_vel', 'displacement_thickness', 'decibels']
lm = LM(df, 'decibels')

# %%
r = lm.get_residuals()
# %%
pyplot.boxplot(r)
# %%
lm.residual_summary()
# %%
lm.anova()
# %%
lm.summary()

# %%
lm.inicial()
# %%
