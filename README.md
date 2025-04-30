Austin Alcancia and Harsha Rauniyar

tree.py contains code of the decision-tree coded from scratch using ID3 algorithm.
Run program as follows from the command line:

python3 tree.py <csv_filename> <training_set_percentage> <random_seed> <Either True or False indicating whether we should handle numeric attributes as
numeric (if False, then we treat them as categorical values)>


For example,
python3 tree.py opticalDigit.csv 0.75 12345 True

Research Questions

Please use your program to answer these questions and record your answers in a README file:

1) Pick a single random seed and a single training set percentage (document both in your
README) and run your program on each of the four data sets. You should pass in True
as the final parameter to your program to treat all numeric attributes as numeric.
a. What is the accuracy you observed on each data set?
b. Calculate a 95% confidence interval for the accuracy on each data set.

2) Create an image of the tree that your program learned in Question 1 for the monks1.csv
data set (you can draw by hand and scan your image into a PDF, or you can use a
drawing program to create an image file). Make sure to upload your image to GitHub.
a. What are the rules learned by the algorithm?
b. How do these rules compare to the true rules in the data set (described on page 1
of the assignment)?

3) Using the same seed and training set percentage from Q1, rerun your program on the
opticalDigit.csv data set and pass in False for the final parameter so that your algorithm
treats each attribute as categorical values (instead of numeric):
a. What is the accuracy you observed?
b. Calculate a 95% confidence interval around that accuracy
c. Compare the confidence intervals from your answer to Q1b and Q3b. What do
you observe? What does this imply?

4) Choose 9 new seeds (document in your README). Rerun your program on
opticalDigit.csv using these 9 new seeds using both True and False as the final parameter
to the program.
a. Calculate the average accuracy across the 10 seeds when you treated the attributes
as (1) numeric and (2) categorical
b. Did you observe the same trends as in Q3c? That is, if one approach achieved a
statistically significantly higher accuracy in Q3c, did the same approach achieve a
higher accuracy when averaged over 10 seeds? If they were not statistically
significantly different in Q3c, are the averages very close?
c. Did these averages fall in your confidence intervals calculated in Q1b and Q3b?


Answer to research questions

1)
With training set percentage of 0.75 and a seed of 3452456, our program achieved the following accuracies.

a)
monks1.csv = 0.9629
penguins.csv = 0.9418 (is_numeric=True)
occupancy.csv= 0.9905 (is_numeric=True)
opticalDigit.csv=0.8911(is_numeric=True)

b)The 95 percent confidence intervals for the accuracy on each set are as follows

monks1.csv

0.9629 +- (1.96 * sqrt(0.9629(1-0.9629)/108)
CI=[0.9273, 0.9985]

penguins.csv

0.9418 +- (1.96 * sqrt(0.9418(1-0.9418)/86)
CI=[0.8923, 0.9913]

occupancy.csv

0.9905 +- (1.96 * sqrt(0.9905(1-0.9905)/5140)
CI=[0.98784, 0.993151]

opticalDigit.csv

0.8911 +- (1.96 * sqrt(0.8911(1-0.8911)/1405)
CI=[0.8748, 0.9074]

2)

a) (Drawing PDF added to github)
The rules learned by the tree are
- If jacket_color is Red then YES
- If jacket_color is Blue AND holding is flag AND body_shape=head_shape then YES else NO
- If jacket_color is Blue AND holding is balloon AND has_tie is NO AND body_shape=head_shape then YES else NO
- If jacket_color is Blue AND holding is balloon AND has_tie is YES AND body_shape=head_shape then YES else NO
- If jacket_color is Blue AND holding is sword AND head_shape=body_shape then YES else NO
- If jacket_color is Green AND head_shape=body_shape then YES else NO
- If jacket_color is Yellow AND holding is flag AND head_shape=body_shape then YES else NO
- If jacket_color is Yellow AND holding is balloon AND body_shape=head_shape then YES else NO
- If jacket_color is Yellow AND holding is sword AND head_shape is square AND body_shape is square then YES else NO
- If jacket_color is Yellow AND holding is sword AND head_shape is round AND body_shape is round then YES else NO
- If jacket_color is Yellow AND holding is sword AND head_shape is octagon the NO

b) The rules for the dataset for the monks1.csv is that if the jacket_color is red then it is a monk. Otherwise if head_shape is equal to body shape, then it is also a monk.

From the rules above there are certain redundant attributes in the decision tree. Holding, and Has Tie are included in the decision tree when these attributes are not even relevant for determining a monk. This unnecessarily increases the height and complexity of the decision tree. Any branch chosen for these redundant attributes will lead to the same result so it is redundant.For example, if predict a certain instance, and we're at the path jacket_color=blue,holding=balloon, then having no_tie, or having a tie will lead to the same result no matter what the head_shape or body_shape was.

The tree, however, seems to learn the rule of the dataset with some redundancies. If we have jacket_color=red, the tree goes to yes indicating that instance is a monk. For all the other paths it also learns that if body_shape is equal to head_shape then the instance must be a monk. However, as stated before these paths have redundant attribute(s) that don't matter.

There is a case where if jacket_color=yellow, holding=sword, and head_shape=octagon, then the tree learns a NO. However, this isn't correct, as the tree hasn't looked at if the body_shape is equal to the head_shape to determine that it is not a monk. This is the only place that leads to the incorrect prediction in the tree for an instance.

3.
a) 
0.5886 accuracy

b) 
0.5886 +- (1.96 * sqrt(0.5886(1-0.5886)/1405)
[0.5629, 0.6143] Confidence Interval

c)
When comparing the results from q1b and q3b, we observe that the accuracy is much higher when using numerical attributes. This implies that handling numerical attributes by finding the best threshold and gain creates a tree that leads to the most accurate predictions. 

4. 
a) seeds = [329, 182, 94, 1, 23, 90, 823, 781, 4938]

0.5730 average for Categorical
0.8994 average for Numerical

b) Yes, we observed the same relationship when averaged over 10 seeds. The average for numerical accuracy was much higher with an average of 0.8994 accuracy over 10 different seeds while the accuracy for categorical accuracy was only 0.573, which is only a little over half predictions correct.

c) Yes, the averages for both categorical (0.5730) and numerical (0.8994) both fell within the confidence intervals from q1b and q3b.

Additional Readme Questions

3) We enjoyed implementing the ID3 algorithm, however, we faced a lot of tricky bugs which were hard to fix.  We were mostly confused about when to remove attributes from the attributes list, and how to remove it. It was a good review of how shallow and deep copy worked

4) 8-10 hrs

5) We have adhered to the Honor Code in this assignment
