# This gives an example on rugularisation:

# source: https://medium.com/@kiprono_65591/regularization-a-technique-used-to-prevent-over-fitting-886d5b361700
#
# Load necessary libraries
from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
# Control matplotlib fonts
font = {'weight' : 'normal',
        'size'   : 14}
matplotlib.rc('font', **font)
# List of colors for plotting different regression lines
colors = ["blue","cyan","black","olive","pink"]
# Different alpha values for penalty terms
alphas = [0,2,5,50]
#set the figure size
plt.figure(figsize=(10,8))
# Loop through different alpha values
for col,alpha in zip(colors,alphas):
    # define the ridge regression model with paramteter
    reg = linear_model.Ridge(alpha=alpha)
    # Fit the model on training data
    reg.fit([[1],[2], [4]], [1,2,3.75])
    x = np.linspace(-5,5,100)
    # extracting the coefficient/slope,a and intercept,b.
    a , b = round(reg.coef_[0],1), round(reg.intercept_,1)
    # the straight line
    y = a*x+b
    # Plotting reegression lines at different alpha values
    plt.ylim(0,5)
    plt.xlim(0,7)
    plt.plot(x, y, '-r', label='y={}x+{}, alpha={}'.format(a,b,alpha),linewidth=3,color=col)
    plt.plot([1,2,4],[1,2,3.75],"ro",color="red",mew=8)
    plt.plot([3,5,6],[2.5,3.45,4.2],"ro",color="green",mew=8)
    plt.xlabel('x', color='#1C2833')
    plt.ylabel('y', color='#1C2833')
    plt.legend(loc='lower right')
    plt.grid(True)
plt.show()
