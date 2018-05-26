#
# Import the necessary modules and libraries
import pandas as pd
from scipy.stats import pearsonr
# function definition to run pearson test
def Run():
    # read data and assign x to features, y to target
    data = pd.read_csv('student-maths.csv')
    X = data.iloc[:, 0:6]
    y = data.iloc[:, 6]
    # list of headings for columns/rows of matrix
    column_headers = ['Age', 'Studytime', 'Failures', 'Absences', 'G1', 'G2']
    # init empty list for all features to assign correlation to
    age = []
    studytime = []
    failures = []
    absences = []
    g1 = []
    g2 = []
    row = []
    # list to iterate through to calc pearsonr
    list = [0,1,2,3,4,5]
    # for loop within for loop to check each feature against each other
    # appends resulting correlation coefficient to the specified list
    for i in list:
        for j in list:
            if (i==0):
                pearson1, p_value = pearsonr(data.iloc[:, [i]],data.iloc[:, [j]])
                age.append("%0.2f"%pearson1)
            elif (i==1):
                pearson1, p_value = pearsonr(data.iloc[:, [i]],data.iloc[:, [j]])
                studytime.append("%0.2f"%pearson1)
            elif (i==2):
                pearson1, p_value = pearsonr(data.iloc[:, [i]],data.iloc[:, [j]])
                failures.append("%0.2f"%pearson1)
            elif (i==3):
                pearson1, p_value = pearsonr(data.iloc[:, [i]],data.iloc[:, [j]])
                absences.append("%0.2f"%pearson1)
            elif (i==4):
                pearson1, p_value = pearsonr(data.iloc[:, [i]],data.iloc[:, [j]])
                g1.append("%0.2f"%pearson1)
            elif (i==5):
                pearson1, p_value = pearsonr(data.iloc[:, [i]],data.iloc[:, [j]])
                g2.append("%0.2f"%pearson1)
    # for loop to append all of the rows together to be inserted into df
    for i in zip(age,studytime,failures,absences,g1,g2):
        row.append(i)
    # insert data into df and print, outputting a table-like structure displaying
    # correlations between features.
    y = pd.DataFrame(row, column_headers, column_headers)   
    print("\nPearsons Correlation Coefficient Between Features\n")
    print(y)
     
