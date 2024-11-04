
"""
Created on Mon Apr 22 15:23:44 2024


"""
#%% IMPORT STATEMENTS & DATA LOADING
# Import statements 

import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.pyplot as plt
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
from sklearn.decomposition import PCA 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from scipy.special import expit 


# Load the data 

df = pd.read_csv('spotify52kData.csv')

# Check for missing values 

missing_data = df.isnull().sum()
total_missing = missing_data.sum()

if total_missing == 0:
    print("No missing values")
    
# Remove duplicates based on having the same track name and album name 

# Calculate average popularity for each duplicate group based on album name and track name
df['popularity'] = df.groupby(['album_name', 'track_name'])['popularity'].transform('mean')

# Remove duplicates, keeping the first instance
df_unique = df.drop_duplicates(subset=['album_name', 'track_name'], keep='first')



#%% QUESTION 1 
print()
print("Question 1: ")
print("plotting histograms....")
# Select the 10 song features
features = ['duration', 'danceability', 'energy', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

# Create a 2x5 grid of histograms for each feature
fig, axes = plt.subplots(2, 5, figsize=(15, 8))
axes = axes.flatten()

# plot the histogram of the  distribution for each feature 
for i, feature in enumerate(features):
    axes[i].hist(df_unique[feature], bins=30, color='skyblue', edgecolor='black')
    axes[i].set_title(feature)
plt.tight_layout()
plt.show()

#%% QUESTION 2 
print()
print("Question 2: ")

# Scatter plot of song length (duration) vs. popularity
plt.figure(figsize=(8, 6))

plt.scatter(df_unique['duration'], df_unique['popularity'], color='skyblue', alpha=0.6)
plt.title('Song Length vs. Popularity')
plt.xlabel('Duration (milliseconds)')
plt.ylabel('Popularity')

plt.grid(True)
plt.show()


duration = df_unique['duration']
popularity = df_unique['popularity']

# Calculate Pearson correlation coefficient
correlation_coefficient = np.corrcoef(duration, popularity)[0, 1]
# Calculate Spearman rank coefficient 
correlation, p_value = spearmanr(duration, popularity)

print("Spearman's correlation coefficient:", correlation)
print("Pearson correlation coefficient:", correlation_coefficient)

#%%QUESTION 3 
print()
print("Question 3: ")


# Create an empty list to store popularity ratings for explicit songs
popularity_ratings_explicit = []

# Filter explicit only into popularity_ratings
explicit_songs = df_unique[df_unique['explicit'] == True]

# Iterate through each explicit song and append its popularity rating to the list
for index, row in explicit_songs.iterrows():
    popularity_ratings_explicit.append(row['popularity'])

# Turn this list into a df 
df_explicit = pd.DataFrame(popularity_ratings_explicit, columns=['popularity'])


# Create an empty list to store popularity ratings for clean songs

popularity_ratings_non_explicit = []

# Filter explicit songs based on a condition (you may adjust this condition as needed)
non_explicit = df_unique[df_unique['explicit'] == False]

# Iterate through each clean song and append its popularity rating to the list
for index, row in non_explicit.iterrows():
    popularity_ratings_non_explicit.append(row['popularity'])
    
# Turn this list into a df 
df_non_explicit = pd.DataFrame(popularity_ratings_non_explicit, columns=['popularity'])

# Create two separate plots for explicit and non-explicit popularity scores
fig, axes = plt.subplots(2, 1, figsize=(8, 8))

# Plot histogram for explicit popularity scores
axes[0].hist(df_explicit['popularity'], bins=30, alpha=0.7, color='purple')
axes[0].set_title('Popularity Distribution for Explicit Songs')
axes[0].set_xlabel('Popularity')
axes[0].set_ylabel('Frequency')

# Plot histogram for non-explicit popularity scores
axes[1].hist(df_non_explicit['popularity'], bins=30, alpha=0.7, color='violet')
axes[1].set_title('Popularity Distribution for Non-Explicit Songs')
axes[1].set_xlabel('Popularity')
axes[1].set_ylabel('Frequency')
print("plotting histograms...")

plt.tight_layout()

plt.show()

# do mann whitney (one-sided)
u1, p1 = stats.mannwhitneyu(popularity_ratings_explicit, popularity_ratings_non_explicit, alternative='greater') # For right-tailed test

print("One-sided p-value:",p1) # one-sided 



#%% QUESTION 4


print()
print("Question 4:")

# Create an empty list to store popularity ratings for explicit songs
popularity_ratings_major = []

# Filter explicit only into popularity_ratings
major_songs = df_unique[df_unique['mode'] == 1]

# Iterate through each explicit song and append its popularity rating to the list
for index, row in major_songs.iterrows():
    popularity_ratings_major.append(row['popularity'])

# Turn this list into a df 
df_major = pd.DataFrame(popularity_ratings_major, columns=['popularity'])


# Create an empty list to store popularity ratings for clean songs

popularity_ratings_minor = []

# Filter explicit songs based on a condition (you may adjust this condition as needed)
minor = df_unique[df_unique['mode'] == 0]

# Iterate through each clean song and append its popularity rating to the list
for index, row in minor.iterrows():
    popularity_ratings_minor.append(row['popularity'])
    
# Turn this list into a df 
df_minor = pd.DataFrame(popularity_ratings_minor, columns=['popularity'])

# Create two separate plots for explicit and non-explicit popularity scores
fig, axes = plt.subplots(2, 1, figsize=(8, 8))

# Plot histogram for explicit popularity scores
axes[0].hist(df_major['popularity'], bins=30, alpha=0.7, color='hotpink')
axes[0].set_title('Popularity Distribution for Major Key Songs')
axes[0].set_xlabel('Popularity')
axes[0].set_ylabel('Frequency')

# Plot histogram for non-explicit popularity scores
axes[1].hist(df_minor['popularity'], bins=30, alpha=0.7, color='purple')
axes[1].set_title('Popularity Distribution for Minor Key Songs')
axes[1].set_xlabel('Popularity')
axes[1].set_ylabel('Frequency')

print("plotting histograms...")

plt.tight_layout()

plt.show()

# do mann whitney (one-sided)
u1, p1 = stats.mannwhitneyu(popularity_ratings_major, popularity_ratings_minor, alternative='greater') # For right-tailed test

print("One-sided p-value:",p1) # one-sided 




#%% QUESTION 5
print()
print("Question 5: ")


slope, intercept = np.polyfit(df_unique['energy'], df_unique['loudness'], 1)


# Scatter plot of energy vs. loudness
plt.figure(figsize=(8, 6))
plt.scatter(df_unique['energy'], df_unique['loudness'], color='forestgreen', alpha=0.6)
plt.title('Energy vs. Loudness')
plt.xlabel('Energy')
plt.ylabel('Loudness')
plt.plot(df_unique['energy'], slope * df_unique['energy'] + intercept, color='red', label='Best Fit Line')

plt.grid(True)
plt.show()

# Calculate Pearson correlation coefficient
correlation_coefficient = np.corrcoef(df_unique['energy'], df_unique['loudness'])[0, 1]

print("r =", correlation_coefficient)

# Now do the same with Spearmans rank correlation rho
rho = stats.spearmanr(df_unique['energy'], df_unique['loudness']) 
print('rho',rho.correlation) 

#%% QUESTION 6



print()
print("Question 6: ")
features = df_unique[features]
zScoreFeatures = stats.zscore(features)

min_rmse = float('inf')
max_r_squared = float('-inf')
best_feature_rmse = None
best_feature_r_squared = None

regression_models = {}
r_squared_values = {}


X = zScoreFeatures.values
y = df_unique['popularity'].values

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=izzy_nnumber)

# Create subplots for all 10 features

for i, feature in enumerate(features):
    # Get the index of the feature in the DataFrame columns
    feature_index = features.columns.get_loc(feature)
    
    # Extract the feature column from X_train and X_test
    x_train_feature = x_train[:, feature_index].reshape(-1, 1)
    x_test_feature = x_test[:, feature_index].reshape(-1, 1)
    
    # Initialize and fit a linear regression model
    regression_model = LinearRegression()
    regression_model.fit(x_train_feature, y_train)
    
    # Store the regression model in the dictionary
    regression_models[feature] = regression_model
    # Calculate the R-squared value
    r_squared = regression_model.score(x_test_feature, y_test)
    y_pred = regression_model.predict(x_test_feature)

  
    # Store the R-squared value in the dictionary
    r_squared_values[feature] = r_squared
  
    # Calculate Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
 
   # Print the R-squared value and RMSE for the current feature
    print(f"R-squared, RMSE value for {feature}: {r_squared}, {rmse}")
    if feature == 'instrumentalness':
        plt.scatter(x_test_feature, y_test, color='pink', label='Actual')
        plt.plot(x_test_feature, y_pred, color='blue', label='Predicted')
        plt.xlabel('instrumentalness')
        plt.ylabel('Popularity')
        plt.title(f'Linear Regression for instrumentalness \nR-squared: {r_squared:.2f}, RMSE: {rmse:.2f}')
        plt.legend()
    # get the smallest RMSE and the biggest R^2
    if rmse < min_rmse:
        min_rmse = rmse
        best_feature_rmse = feature
    if r_squared > max_r_squared:
        max_r_squared = r_squared
        best_feature_r_squared = feature

# Show plot
plt.grid(True)
plt.show()
print()
print("plotting linear regression for instrumentalness...")

        
# Print the minimum RMSE and maximum R-squared
print(f"\nMinimum RMSE: {min_rmse:.2f}, Best Feature: {best_feature_rmse}")
print(f"Maximum R-squared: {max_r_squared:.2f}, Best Feature: {best_feature_r_squared}")



#%% QUESTION 7
print("\nQuestion 7: ")
features = ['duration', 'danceability', 'energy', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
X = df_unique[features]
y = df_unique['popularity']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=izzy_nnumber, test_size=0.25)

# Fit the multiple regression model on the training data
multiple_regression = smf.ols(formula='popularity ~ danceability + energy + loudness + speechiness + acousticness + instrumentalness + liveness + valence + tempo', data=pd.concat([x_train, y_train], axis=1))
results = multiple_regression.fit()
print('Printing model summary:')
print(results.summary())

# Make predictions on the testing data
y_pred = results.predict(x_train)

# Calculate R-squared value
r_squared = r2_score(y_train, y_pred)
print()
print("R-squared value on training set:", r_squared)



print()
#%% QUESTION 8 
print("Question 8: ")
data = df_unique.to_numpy()
data = np.array(data)



features = ['duration', 'danceability', 'energy', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

features_df =df_unique[features]

y = df_unique['popularity']


# 1. Z-score the data 
zscoredData = stats.zscore(features_df) 

print("Plotting scree plot, correlation matrix, and loadng plots...")

# correlation matrix 
corr_matrix = zscoredData.corr()


# plot corr matrix 
plt.imshow(corr_matrix)
plt.xticks(range(1, len(features)+1),features,rotation = 'vertical')
plt.xlabel("Feature")
plt.yticks(range(1, len(features)+1),features,rotation = 'horizontal')
plt.ylabel("Feature")
plt.colorbar()
plt.show()
# 2. Initialize PCA object and fit to our data:
pca = PCA().fit(zscoredData) 


# Eigenvalues: Single vector of eigenvalues in decreasing order of magnitude
eigVals = pca.explained_variance_

# Loadings -- what are the coordinates of the eigenvectors in terms of the 10 factors? 
loadings = pca.components_


# 3c. Rotated Data
rotatedData = pca.fit_transform(zscoredData)


# 4.  eigenvalues in terms of  variance explained:
varExplained = eigVals/sum(eigVals)*100


# Now let's display this for each factor:
print("Variance explained by each factor: ")
for ii in range(len(varExplained)):
    print(varExplained[ii].round(3))
sum = 0
for x in range(3):
    sum+=varExplained[x]
print('sum',sum)
    
# What a scree plot is: A bar graph of the sorted Eigenvalues
numComponents = len(features)
x = np.linspace(1,numComponents,numComponents)
plt.title("Scree Plot")

plt.bar(x, eigVals, color='pink')
plt.plot([0,numComponents],[1,1],color='orange') # Orange Kaiser criterion line
plt.xlabel('Principal component')
plt.ylabel('Eigenvalue')
plt.show()

# 1) Kaiser criterion: Keep all factors with an eigenvalue > 1
kaiserThreshold = 1
print('Number of factors selected by Kaiser criterion:', np.count_nonzero(eigVals > kaiserThreshold))


whichPrincipalComponent = 1 
plt.bar(x,loadings[whichPrincipalComponent,:]*-1,color = 'orchid') 
plt.title("Loading Plot for Principal Component 1")

plt.xlabel('Song Feature')
plt.xticks(range(1, len(features)+1),features,rotation = 'vertical')
plt.ylabel('Loading')

plt.show()



whichPrincipalComponent = 2 
plt.bar(x,loadings[whichPrincipalComponent,:]*-1, color = 'mediumvioletred') 
plt.title("Loading Plot for Principal Component 2")

plt.xlabel('Song Feature')
plt.xticks(range(1, len(features)+1),features,rotation = 'vertical')
plt.ylabel('Loading')

plt.show() 

whichPrincipalComponent = 3 
plt.bar(x,loadings[whichPrincipalComponent,:]*-1,color = 'darkviolet') 
plt.title("Loading Plot for Principal Component 3")
plt.xlabel('Song Feature')
plt.xticks(range(1, len(features)+1),features,rotation = 'vertical')
plt.ylabel('Loading')

plt.show() 



#%% QUESTION 9 


print()
print("Number 9: ")

# For VALENCE ONLY

# Extracting features and target variable
y = df_unique['mode'].values
X = df_unique[['valence']].values

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=izzy_nnumber)

# Fitting logistic regression model
model_valence = LogisticRegression().fit(X_train, y_train)
y_prob= model_valence.predict_proba(X_test)[:, 1]  
auc_roc_valence = roc_auc_score(y_test, y_prob)
print("Valence: ")
print("AUROC:",auc_roc_valence)
y_pred_binary = (y_prob >= 0.5).astype(int)
accuracy_valence = accuracy_score(y_test, y_pred_binary)
print("Accuracy:",accuracy_valence)



x1 = np.linspace(0,1)
y1 = x1 * model_valence.coef_ + model_valence.intercept_
sigmoid = expit(y1)
# Plot:
plt.plot(x1,sigmoid.ravel(),color='green',linewidth=3) # the ravel function returns a flattened array
plt.scatter(df_unique['valence'],df_unique['mode'],color='pink')
plt.hlines(0.5,0,1,colors='gray',linestyles='dotted')
plt.xlabel('Valence')
plt.xlim([-0.1,1.1])
plt.ylabel('Key (Major = 1, Minor = 0)')
plt.title('Logistic Regression Curve for predicting major/minor key from valence)')

plt.yticks(np.array([0,1]))
plt.show()

print()

# Now for all other features

features_no_valence = ['duration', 'danceability', 'energy', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'tempo']
features_no_valence_pd = df_unique.loc[:, features_no_valence]


fig, axes = plt.subplots(3, 3, figsize=(12, 12))

for i, ax in enumerate(axes.ravel()):



    x = features_no_valence_pd[features_no_valence[i]].to_numpy().reshape(-1, 1)
    y = df_unique['mode'].values
    # Splitting the data into training and testing sets for each feature
    X_train2, X_test2, y_train2, y_test2 = train_test_split(x, y, test_size=0.2, random_state=izzy_nnumber)
    model_feature = LogisticRegression()
    model_feature.fit(X_train2, y_train2)
    y_prob2 = model_feature.predict_proba(X_test2)[:, 1]  
    auc_roc_feature = roc_auc_score(y_test2, y_prob2)
    y_pred_binary = (y_prob2 >= 0.5).astype(int)
    accuracy_feature = accuracy_score(y_test2, y_pred_binary)

  

    print(f"{features_no_valence[i]}: ")
    print("AUROC:", auc_roc_feature)
    print("Accuracy:", accuracy_feature)

    print()

    x1 = np.linspace(x.min(), x.max()).reshape(-1, 1)
    y1 = x1 * model_feature.coef_ + model_feature.intercept_
    sigmoid = expit(y1)
    

    ax.plot(x1, sigmoid.ravel(), color='green', linewidth=3)
    ax.scatter(features_no_valence_pd[features_no_valence[i]], df_unique['mode'], color='pink')
    ax.hlines(0.5, x.min(), x.max(), colors='gray', linestyles='dotted')
    ax.set_xlabel(features_no_valence[i])
    ax.set_xlim([x.min(), x.max()])
    ax.set_ylabel('Key (Major = 1, Minor = 0)')
    ax.set_yticks(np.array([0, 1]))
    ax.set_title(f'Logistic Regression Curve for Predicting\n Major or Minor Key\n({features_no_valence[i]})')
    


    
plt.tight_layout()
plt.show()



#%% QUESTION 10
# Loop through the 'genre' column and update values
for index, row in df_unique.iterrows():
    if row['track_genre'] == 'classical':
        df_unique.at[index, 'track_genre'] = 1
    else:
        df_unique.at[index, 'track_genre'] = 0
        


# Extracting features and target variable
y = df_unique['track_genre'].values
X = df_unique[['duration']].values
# Convert y to integer type
y = y.astype(int)


# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=izzy_nnumber)

# Fitting logistic regression model
model_valence = LogisticRegression().fit(X_train, y_train)
y_pred_valence = model_valence.predict_proba(X_test)[:, 1]
auc_roc_valence = roc_auc_score(y_test, y_pred_valence)
y_pred_binary = (y_pred_valence >= 0.5).astype(int)
accuracy_valence = accuracy_score(y_test, y_pred_binary)



# Calculate accuracy and AUROC for valence
print("Accuracy for duration:", accuracy_valence)
print("AUROC for duration:", auc_roc_valence)


features_df =df_unique[features]
# Do it for principal components 
rotated_data3 = rotatedData[:, :3]
rotatedFeatures = (pd.DataFrame(rotated_data3))
X2 = rotatedFeatures


# Splitting the data into training and testing sets
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y, test_size=0.2, random_state=izzy_nnumber)

# Fitting logistic regression model
model_principals = LogisticRegression().fit(X_train2, y_train2)
y_pred_principals = model_principals.predict_proba(X_test2)[:, 1]
auc_roc_principal = roc_auc_score(y_test2, y_pred_principals)
y_pred_binary = (y_pred_principals >= 0.5).astype(int)
accuracy_principal = accuracy_score(y_test2, y_pred_binary)

# Calculate accuracy and AUROC for principal components 
print("Accuracy for Principal Components:", accuracy_principal)
print("AUROC for Principal Components:", auc_roc_principal)


