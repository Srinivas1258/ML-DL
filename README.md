# ML-DL

Cousera - Launching into machine learning .

Supervised learning -label data (both input and output)
ex : tabular data ,excels ,sqls .


unsupervised learning  -unlabeled data
ex- text ,speech ,images 

In supervised learning - mainly two tasks 
1.classification (category |discrete )
ex : cat and dog image classification.
2.regression (predicting continuos values)
ex :predicting house prices ,regression model such as predicting the number of fraudulent transactions, fraudulent transaction amounts, etc.

In ML , tabular data  
each row is -example ,samples

each column is -feature ,values

output /predicting value -label


 In regression problems, we want to minimize the error between our predicted continuous value, and the label's continuous value, usually using mean squared error
 
 In classification we use cross entropy ,
 
 Ml is all about experimentation.
 
 A linear decision boundary will form a line or a hyperplane in higher dimensions, with each class on either side.
 
 linear regression : prediction is weighted sum of inputs.
 
 Y=W0x0+w1x1+w2x2+....    w - wieghts ,x -features
 y^ =XW
 
 In linear regression the error is mean square error (l2 norm)
 
 Error = |Y -Y^|^2       Y -actual value 
                         Y^ -predicted value 
                         
 LR doesn't perform well in large datasets .
 
 
 Gradient descent : finding optimal weights |global minimum  
 
 Learning rate - hyperparameter that helps determine gradient descent's step size, along the hypersurface, to hopefully speed up convergence .

Perceptron : Computational model of neuron 
simple neural network with 1 hidden layer.

What component of a biological neuron is analogous to the input portion of a perceptron? -answer dendrites 
axon -output 
nucleus,myelin sheath -hidden layers

Neural networks : these are combine layers of perceptrons ,make them more powerful but also harder to train effectively .

If I wanted my outputs to be in the form of probabilities which activation function should I use in the final layer?
 ans : Sigmoid(range is b/w 0 and 1 )is exactly for probability .
 
 Decision Trees : it will built peice-wise linear decision boundaries are easy to train , and are easy for humans to interpret.
 
 In a decision classification tree, what does each decision or node consist of? ans :linear classifier for one feature .
 
 Non linear kernel methods or SVM's
 Svms maximize the margin b/w two support vectors .
 
 We’ve seen how SVMs use kernels to map the inputs to a higher dimensional feature space. What thing in neural networks also can map to a higher dimensional vector space?
  Ans: More neurons per layer
  
  Random forests : combination of Decision trees.
  Strong learner from many weak learners .
  
  ML models are mathematical functions with parameters and hyperparameters .
  parameters : changed during training.(weights and biases)
  hyperparameters :set before training .(learning rate ,alpha ,)
  
  linear models Y =wx+b    w -weight b =bias
  
  Loss functions : By calculating errors 
  Error =actual value - predicted value 
  1. mean square error (l1 norm) -regression
  2.root mse (l2 norm) -regresssion
  
  cross entropy (log loss)
  
  
  Gradien Descent : find optimal wieghts and biases.
  1. which direction should i go ?
  2.How large or small step ?
  Ans : Loss function slope provides direction and steps size.
  
  In ml model if we run the same code again to reduce the loss -foolishness
  U have to update the code to reduce the loss(change hyper parameters)
  
  
  As learning rate increases ,weights increases.(set the learning rate that is required (not too high or not low)
  
   batch size will have a simple effect on the rate of convergence. As with learning rate, the optimal batch size is problem dependent      and can be found using hyperparameter tuning
   
   More hidden layer means more hierarchies features .(more complex the model is)
   
   Performance metrics :
   1. Confusion matrix to assess the classification model performance.
   
   TP =true positive -label +ve prediction +ve (Available parking space exists ,model predicts space is available)
   FN =False negative -label +ve prediction -ve (Available parking space exists ,model predicts space isn't available)
   FP =False positive - label -ve prediction +Ve (Available parking space isn't exists ,model predicts space is available)
   TN =True negative -label -ve prediction -ve   (Available parking space isn't exists ,model predicts space isn't available)
   
   Precision(prediction +ve) = TP / (TP + FP)
   
   Recall or sensitivity(label +ve) =TP /(TP+FN)
   
   Specificity = TN/(TN +FN)
   
   F1 score = (1/precision) +(1/recall)
   
   Accuracy = (TP +TN )/(TP+TN+FP+FN)
   
   Optimization finds best ML model parameters .
   
   Generalization : The ability of the learned model to fit unseen instance.
  
  
 Introduction to tensorflow :
 
 TensorFlow is an open source, high performance, library for numerical computation that uses directed graphs.
 
 A tensor is N dimensional array of data .
 
DAG -Directed Acyclic Graph.

Tensorflow API hierarchy -tf.estimator is high level API

the python API let's u build and  run directed graphs.

Graphs can be processed ,compiled ,remotely executed and assigned to devices.

A TensorFlow DAG consists of tensors and operations on those tensors.

A variable is a tensor whose value is initialized and then the value gets changed as a program runs.

In tensor flow programming errors occurs mainly due to (when adding two tensors they have exact same shape)

1)shape(tf.reshape) -to check shape of data and convert it in to required shape
2)data types

to solve errors  1)read error message 
2)fake data execution(giving some data like testing)
3)use print statements for diagnosing
4)Isolate the method in question(using functions)


Feature Engineering

Features (columns of data) or getting useful info from data is called features.

Good features :

1.)Should be related to Objective(predicted value)-Why these features should affect the outcome?

tip: You can not train with current data and predict the past data (not a good model)

2)Feature shoud be numeric/categorical

3)Have enough examples(like cat and dog images)

4)Bring human insight into problem

5)meaningful magnitude(like no need emp id -13832)


Apache Beam vs cloud Data flow

Apche Beam :Beam is way to write elastic data pipelines. its supports both batch and stream processing.

Beam -Batch +stream
Pipelines -sequence of steps.

Steps include :
1.Input  2.Read 3.transform 4.Group 5.Filter 6.Write  7.output

MapReduce :
1.splits the data 
2.maps the data
3.reduce and shuffle the data

Feature cross :It provides a way to combine features in linear model and help linear model to work in non-linear problems.

Ex: IN XOR Gate -(we will get decision boundary using x3=x1*x2)(non-linear models)

Ex: (combination of hour and day)will give new feature cross.

It memorize the input space.

Memorization works only on large datasets.

Goal of ML is generalization.

Feature cross will leads to sparsity. 

Sparsity :it condition is when not having enough samples.



1.Early stopping  : When test data curve not going well with train data curve(means test error is increasing)

Simpler models are always better.

2. Regularization : Our main goal is to minimize error(loss value).(test data especially)-only makes weights small.(not zero)

Regularization helps to generalize the model better.

L2 norm :The L2 norm is calculated as the square root of sum of the squared values of all vector components

|W|^2 =(w0^2+w1^2+...)

L1 norm : L1 measures absolute value of a plus absolute value of b
|W|=|w0|+|w1|+....

loss =Loss(W,D) +lambda(W)                 W- regularization -l1
                                      |W|^2 -L2 regularization
                                      
   Learing rate -controls the size of step in weight space. Default learning rate -0.01 or 1/sqrt(num features)(medium size)
   
   Batch size -controls the no.of samples.(medium size)

Hyperparameters -set before training.
parameters -changed during the training the model.

Elastice nets -these are combinations of l1 norm and l2 norm.

With (l1)regularization all useless features are set to 0
With l2 reularization all features are set to small values.

Logistic Regression :transform linear regression by a sigmoid activation function.

y^ = 1/(1+e^-x)      x=w^t X+b   - output is in probability b/w 0 to 1.

Use the ROC curve to choose the decision threshold based on decision criteria.

AUC helps you choose between models when you don't know what your system threshold is going to be ultimately used.

Introduction ANN:

Why is it important adding non-linear activation functions to neural networks?

Ans :Stops the layers from collapsing back into just a linear model

Fav non-linear activtion model -RELU


Neural networks can be arbitrarily complex. To increase hidden dimensions, I can add _______. To increase function composition, I can add _______. If I have multiple labels per example, I can add _______.


Ans: Neurons, layers, outputs

Common failures of Gradient Descent :

1.Vanshing Gradients. -problem
Each additional layer can successively reduce signal vs noise.  -insights
(use relu instead of sig/tanh)  -solution

2.Exploding Gradients  -Problem
Learning Rate are important there  -insight
Batch normaliztion can help -solution.

3.Relu layers can die -problem
Monitor Fraction of Zero weights in tensorboard.  -Insights
Lower your learning rates  -solution.

Which of these is good advice if my model is experiencing exploding gradients?

Ans:all
Lower the learning rate
Add weight regularization
Add gradient clipping
Add batch normalization

Dropout - Drop some layers randomly(probability(15-20%)
-Used in training only.
It simulates ensembles learning.
The more you drop out stronger the regularization.

Inference -test

Sigmoid -only for binary classification.
Softmax -multi classification.(it detects only single label at time)(like  cat and dog in same image)

For our classification output, if we have both mutually exclusive labels and probabilities, we should use ______. If the labels are mutually exclusive, but the probabilities aren’t, we should use ______. If our labels aren’t mutually exclusive, we should use ______.

I. tf.nn.sigmoid_cross_entropy_with_logits

II. tf.nn.sparse_softmax_cross_entropy_with_logits

III. tf.nn.softmax_cross_entropy_with_logits_v2

Ans -3,2,1

Use of Embedding :
1.Manage Sparse data

2.Reduce dimensionality Reduction

3.Increase the model generalization.

4.Cluster Observations


Create Reusable Embedding.

Create Embedding column from feature cross.

Never try your model with full dataset ,first try with samples data set , then full data set.
