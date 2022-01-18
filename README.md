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

This label corresponding to each row represents the classified Object Number with respect to that particular signal/Row.
Similarly, if we provide the particular signal information (signal row number), the GUI tab will display which object it belongs to as depicted in the snapshot.

Option 2 : 
Step 1 : Run the python file u_onlytest.py. It is advisable to run the code through an IDE like PyCharm in Conda environment. Otherwise, if it is executed in a base machine without any IDE then the necessary python libraries need to be installed in the host environment.
Step 2 : 


![Picture5](https://user-images.githubusercontent.com/57104937/149929005-b6d95953-f164-439e-90f1-c2efde9aaacb.png)

Use the signal start column and signal end column as 7 and 3406. The model has been trained and stored with this dimension. If the testing dimension does not exactly match then dimension mismatch error will be reproduced. If the dimension in the test file has to be different then it is advisable to try Option 1 and train the model with the required dimension.
Load the file trained_model.pkl in the browse model tab. Provide Path for storing the Output file.
Step 3: Test the model with the GUI button. A file named Output.xlsx will be generated. In the output file, newly created labels corresponding to individual signal would be observed at the last column as below.

![Picture6](https://user-images.githubusercontent.com/57104937/149929113-99f44bb1-2f68-45a9-9db5-9b784dd351a6.png)

This label corresponding to each row represents the classified Object Number with respect to that particular signal/Row. Similarly, in the GUI tab Output Measures, if we provide the particular signal information (signal row number), the GUI tab will display which object it belongs to as depicted in the snapshot.



![Picture7](https://user-images.githubusercontent.com/57104937/149929191-728a1bee-c5b5-41e4-8184-78d665620006.png)
