import numpy as np
import matplotlib.pyplot as plt


def create_custom_weight_matrix(n, d):

    shape = (200, 200)

    origin = (100, 100)
    m = n/d
    k = n-n*np.log(d)
    print(n,m,k)

    # Create a coordinate grid

    Y, X = np.ogrid[:shape[0], :shape[1]]

    

    # Compute the distance squared from the origin

    distance_squared = (X - origin[1])**2 + (Y - origin[0])**2

    distance = np.sqrt(distance_squared)

    

    # Calculate weights based on the distance

    weights = (0.1 * distance)**n * np.exp(-m * (0.1*distance) + k)

    

    return weights



# Parameters


# Creating the weight matrix

weight_matrix = create_custom_weight_matrix(0.5, 2)



# Visualizing the weight matrix

plt.imshow(weight_matrix, cmap='hot', interpolation='nearest')

plt.colorbar()

plt.title('Custom Weight Matrix Visualization')

plt.show()