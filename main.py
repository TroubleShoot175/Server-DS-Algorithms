# Homework 11 - Exercise 01

# Dependency Libraries
import pandas as pd
import math as m
import random as r
import numpy as np
from matplotlib import pyplot as plt


def calc_distance(p1, p2):
    '''
    Used to Calculate the distance between two points
    '''
    dist = m.sqrt(((p2[0] - p1[0]) ** 2) + ((p2[1] - p1[1]) ** 2))
    return dist


def rider_kmeans(dataframe: pd.DataFrame, k: int):
    """
    Custom K-Means Implementation
    """

    # Make a copy of the Passed DataFrame and assign its value to the working dataframe (wdf)
    wdf = dataframe

    # Get the Mins and Maxes To Determine Range for Guessed Centroids
    xmax, ymax = dataframe.max(axis=0)
    xmin, ymin = dataframe.min(axis=0)

    # Add Column "nK" or nearest centroid to working dataframe (wdf)
    wdf["nK"] = list(0 for _ in range(len(dataframe.index)))

    # Create Centroid DataFrame (cdf) using passed K
    cdf = pd.DataFrame(
        zip(list(r.randint(xmin, xmax) for _ in range(k)), list(r.randint(ymin, ymax) for _ in range(k))),
        columns=["X", "Y"])

    # Start of K-Means Algorithm
    while True:
        # List for the centroid nearest to that centroid
        # cnearestc = []

        # I removed this portion because I realized that my centroid math is done based on the position of the centroid and not the distances to or from the centroids
        # # For each of the centroids calculate the distances to the other centroids leaving out the one that is itself
        # for i in cdf.index:
        #     templist = []
        #     for j in cdf.index:
        #         if i != j:
        #             # Call calc_distance or Euclidean Distance Equation
        #             dist = calc_distance((cdf["X"][i], cdf["Y"][i]), (cdf["X"][j], cdf["Y"][j]))
        #             # print(f'Centroid {i} to Centroid {j} distance is: {dist}')
        #             # Append results to temporary list
        #             templist.append((j, dist))
        #     # print(f"Centroid {i}'s closest centroid is {min(templist, key=lambda t: t[1])[0]} and the distance is {min(templist, key=lambda t : t[1])[1]}")
        #     # Find the smallest distnace and append that to the centroids nearset centroid will be read 0, 1, 2
        #     cnearestc.append(min(templist, key=lambda t: t[1])[0])
        #
        # # Create the Nearest Centroid column for the centroid dataframe
        # cdf["nK"] = cnearestc

        # Temporary List for the point's nearest centroid
        pnearestc = []

        # For each of the points in the working data frame calculate the distance to each of the four centroids
        for p in wdf.index:
            templist = []
            for c in cdf.index:
                # Check to make sure that if a point has the same value as a centroid that distance is zero
                if (wdf["X"][p], wdf["Y"][p]) != (cdf["X"][c], cdf["Y"][c]):
                    # print(f'Point Coord: {wdf["X"][p], wdf["Y"][p]} | Centroid Coord: {cdf["X"][c], cdf["Y"][c]}')
                    dist = calc_distance((wdf["X"][p], wdf["Y"][p]), (cdf["X"][c], cdf["Y"][c]))
                    templist.append((c, dist))
            # print(f"Point {p}'s closest centroid is {min(templist, key=lambda t: t[1])[0]} and the distance is {min(templist, key=lambda t: t[1])[1]}")
            # Find the smallest distnace and append that to the centroids nearset centroid will be read 0, 1, 2
            pnearestc.append(min(templist, key=lambda t: t[1])[0])

        # Add list to column on working dataframe (wdf)
        wdf["nK"] = pnearestc
        previousCentroids = []

        # Set aside the current centroids to be checked against the new centroids
        for c in cdf.index:
            previousCentroids.append((cdf.at[c, "X"], cdf.at[c, "Y"]))

        # Calculate new centroids
        for c in cdf.index:
            cnt = 0
            xval = []
            yval = []
            for p in wdf.index:
                if wdf["nK"][p] == c:
                    cnt += 1
                    xval.append(wdf["X"][p])
                    yval.append(wdf["Y"][p])
            if cnt != 0:
                xavg = sum(xval) / cnt
                yavg = sum(yval) / cnt
                cdf.at[c, "X"] = xavg
                cdf.at[c, "Y"] = yavg
            else:
                continue

        # Check to see if the new centroids are equal to the old centroids
        cnt = 0
        for uC in cdf.index:
            if (cdf.at[uC, "X"], cdf.at[uC, "Y"]) == previousCentroids[uC]:
                # print(cnt)
                cnt += 1

        # If the count of the number of match centroids is equal to K then break out of the loop
        if cnt == k:
            break

    # Return centroid dataframe (cdf) and the column of the working dataframe (wdf) as a numpy array
    return cdf.to_numpy(), wdf["nK"].to_numpy()

# Read CSV For Data
df = pd.read_csv("Mall_Customers.csv")

# Rename Columns
df.columns = ['Customer', 'Genre', 'Age', 'Income', 'Spending']

# Make Arrays for k-means clustering
x = np.array(df[["Age"]])
y = np.array(df[["Spending"]])

# Make DataFrame From Two Arrays
df = pd.DataFrame(list(zip(x, y)), columns=["X", "Y"])

# Call My Implementation
result, labels = rider_kmeans(df, 3)

# My Centroids and the built-in KMeans
print("My Centroids")
print(result)

# Plot On Chart For Client

