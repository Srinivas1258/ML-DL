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


Apache Beam :

