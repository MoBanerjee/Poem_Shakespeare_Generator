# Shakespearean-style Poetry Generator
This machine learning project is designed to learn from shakespearean-style poetry that has been fed into it. In return, it generates shakespearean style texts when provided with some starting text. 

## Dataset Used
The text dataset used in this project was sourced from https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt

## Libraries Imported
1. Random
2. Numpy
3. Tensorflow
4. Keras
5. urllib
6. ssl

## Data Preparation 
* As the raw dataset was extensive, only a part of it was extracted to be used as training dataset for the model.
* The entire text was converted into lowercase for maintaining consistency.
* All unique characters present in the extracted text were filtered and arranged alphabetically in a characters array.
* Two dictionaries were setup, one containing character to index mapping and other in a vice-versa format. This was required as the model cannot understand raw string data. It can only operate on numerical data and therefore each character had to be mapped to its corresponding numerical encoding(index) using the enumerate function.
* A sequence length variable was set with a value of 40. This is the length of the initial text to be fed into the model depending on which it will predict the next characters.
* A step size variable of value 3 was also declared so that during training, each successive sequence of 40 characters starts at a position which is 3 characters ahead of the previous one.
* A 2-level nested for loop was run in order to fill up two arrays-: sen and nextchar. The sen array contains all the sequences spanning the extracted text. The nextchar array contains the corresponding next character of each sequence listed in the sen array.
* Following this, 2 numpy arrays were setup and populated. The x array contains the inputs of the training dataset. Its shape tuple has three parameters which can be visualised as 3 coordinate axes in space-: one of length equal to size of sen array(i.e total number of sequences), second of length equal to sequence length(40) and the third of length equal to size of characters array. The datatype of the array contents was set to boolean. So for instance, if a certain character 'x' was present in the 5th index of the 3rd sequence, the value at (3,5,numerical encoding of 'x') will be true(1).
* Similarly, the y array contains the output of the training dataset. Its shape tuple has two parameters-: one of length equal to size of sen array and the other of length equal to size of characters array. So if a certain sequence has a certain next character, that corresponding position will be a 1.

## Model Used
* A special type of Recurrent Neural Network (RNN) called LSTM (Long-Short-Term-Memory) was used in this project.
* A new sequential model was set up using Sequential class provided by Keras library. It is a linear stack of multiple layers.
* The first layer added was LSTM, comprising of 128 neurons. It is the memory unit of the system. As it has a short term memory, it comes handy in predicting the next character based on recent inputs instead of relatively older iterations.
* The second layer added was Dense layer (the hidden layers). It is the fully connected network portion of the system.It comes handy in understanding the patterns and relationships embedded in the training dataset. It has its number of neurons equal to the number of unique characters.
* The third layer is for the output called the Activation layer. It encloses an activation function called softmax which is commonly used in multi-class classification problems. It normalizes the output by representing each possible class with a certain probability, all of which sum up to 1.
* During compiling, a loss function has been set up called categorical_crossentropy to calculate the loss value, i.e difference between predicted and actual output. An optimizer function has also been set up using the imported RMSprop which adjusts the model's parameters (weights and biases) to minimize the loss value. The learning rate of the optimizer is 0.01
* The training dataset is fitted into the model using a batch-size of 256 and an epoch of 4. Batch-size refers to the sample size which is passed into the model at a time before its parameters are optimized. An epoch refers to one complete cycle of the entire training dataset being read by the model. The number of epochs is a hyperparameter which needs to be decided based on various factors like size of dataset, problem complexity and so on. Too few epochs can result in underfitting, i.e the model cannot gauge the patterns in the training data sufficiently. Similarly, excess epochs can cause overfitting, i.e the model is excessively well-trained on the training dataset provided and cannot generalize to any new test dataset.
* As the training of the model is time-consuming, the model is saved into the local directory as poet.model so that it can loaded later whenever needed instead of having to train it repeatedly.

## Output Generation
* As the predictions are returned as an array of probabilities of different classes of characters ( in their numerical form), two new functions-: sample and textgen are defined to convert the output into text format.
* The sample function first normalizes the probabilities based on temperature. The temperature parameter influences whether the final output will be random or more deterministic. A higher value causes more randomisation while a lower value yields a more predictable result. The function returns the numerical encoding of the character with the highest probability.
* The textgen function converts the numerical encoding back to text and returns a string called generated with the resultant text output. The input is randomised using the random function, i.e the input text starts randomly from any location of the initially extracted text and spans over the next 40 characters. Thus, this serves as a test data for the model. It also takes in a length parameter that decides how many new characters we wish the model to predict.
* Results are printed out with different temperature values ranging from 0.2 (a low value) to 1 (a fairly high value)

## Conclusion
* The text for both temperature values of 0.2 and 1 are very awry and largely doesn't make sense. 
* But the moderate values of 0.6 and 0.8 have fared much better with more interpretable text predictions. 
* Thus, it is not advisable to go for very high or very low temperature values.
* The loss function in the final epoch was 1.988

## Challenges Faced
* This is my maiden Machine Learning Project. So the learning curve was quite steep and I had to quickly grasp multiple new concepts.
* I faced problems in handling the installation of several package dependancies required for the various libraries imported
* I had to turn off ssl verification in order to download the text file from the link as otherwise the digital certificate was not getting approved.
* I had to experiment with several temperature values in order to get a output that largely makes sense.

## Future Efforts
* I have attempted to embed the python input function in the textgen function to generate new text based on initial text inputted by users themselves. Although the currently uploaded code in the python notebook still takes its starting text from the dataset, the attempt was successful.
* I also intend to try out this model with other datasets and also with self-made text files or text sources like boks, social media chats and so on.
* I also intend to deploy this functionality in an app.

## References
1. https://www.youtube.com/watch?v=QM5XDc4NQJo&list=PLBVQ4krSSN7Urd4YGbCvzLLm76ZJXrjAa&index=5&t=147s
2. https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt
