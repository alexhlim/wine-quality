import pandas as pd
import seaborn as sns
import warnings

red_wine = pd.read_csv('winequality-red.csv', sep = ';')
# rename variables: replace space with underscores (makes it easier to reference)
red_wine.columns = [c.lower().replace(' ', '_') for c in red_wine.columns]

# colors I will be using in this notebook
OrRd = sns.color_palette("OrRd_r", n_colors=len(red_wine.columns))
dark_red = OrRd[0]
light_red = OrRd[8]

# ignore warnings
warnings.filterwarnings(action = 'ignore')




