import matplotlib.pyplot as plt
import seaborn as sns

sns.scatterplot('population', 'Area', data=df, hue='continent')

plt.show()

# Get Unique continents
color_labels = df['continent'].unique()

# List of colors in the color palettes
rgb_values = sns.color_palette("Set2", 4)

# Map continents to the colors
color_map = dict(zip(color_labels, rgb_values))

# Finally use the mapped values
plt.scatter(df['population'], df['Area'], c=df['continent'].map(color_map)
