# Object-Detection-by-Audio-Classification-using-Multi-Layer-Perceptron

Discrimination of Reflected Sound Signals using Multilayer Perceptron
Project Run Options :
There are two options to run the project.

Option 1 : Training the model and subsequently testing with new data

Option 2: Directly perform testing by loading trained model from a .pkl file and subsequently testing with new data

Run Guide
Option 1 : 
Step1 : Run the python file traintest.py. It is advisable to run the code through an IDE like PyCharm in Conda environment. Otherwise, if it is executed in a base machine without any IDE then the necessary python libraries need to be installed in the host environment.

![Picture1](https://user-images.githubusercontent.com/57104937/149928498-64d054ea-503a-42c5-85b2-5ef2e7e5f3b7.png)

In the GUI Browse Folder field, provide the directory where the training excel files are located.  All the excel files in the directory and subdirectory would be fetched and data would be read from there.
Provide the Signal Start and End Column. As per the excel file provided to us, the entire range of an individual signal has been from Column 7 to Column 3406. 
Train the Model through the GUI Button and the Accuracy would be displayed as above. The model validation metrics and relevant plots of Confusion Matrix and ROC Curve can be viewed in the other tab – Output Measures
<img width="451" alt="Picture2" src="https://user-images.githubusercontent.com/57104937/149928628-aef8b635-7121-4859-87b7-f42b93918b28.png">

Step 3:  Now, as the model is trained and validated, it can be tested with new data.

![Picture3](https://user-images.githubusercontent.com/57104937/149928775-99c1b956-b954-4cac-95a4-d5b3d36e404c.png)

Browse the excel file to be tested through the GUI tab. Provide the Save file location. Make sure the signal length (Signal Start Column and Signal End Column) matches exactly the same dimension as it was provided during training. In this case it is 7  and 3406 respectively. Otherwise, we would get dimension mismatch error in testing.
New Output will be generated with the filename Output.xlsx. In the output file, newly created labels corresponding to individual signal would be observed at the last column as below.

![Picture4](https://user-images.githubusercontent.com/57104937/149928878-0ab08c35-b50f-469f-a596-a7ee1916b38f.png)
