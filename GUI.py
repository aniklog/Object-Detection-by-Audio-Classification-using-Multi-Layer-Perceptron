import os
import openpyxl
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import *
from scipy import interpolate
from itertools import cycle
import matplotlib.pyplot as plt
from tkinter import ttk,messagebox,filedialog
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,f1_score
from sklearn.metrics import plot_confusion_matrix,roc_curve,auc
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

# start tkinter app code
root = tk.Tk()
root.title("Discrimination of Reflected Sound Signal(MLP Classifier)")
tabControl = ttk.Notebook(root)

tab1 = ttk.Frame(tabControl)
tab2 = ttk.Frame(tabControl)

tabControl.add(tab1, text ='Test and Train')
tabControl.add(tab2, text ='Output Measures')
tabControl.pack(expand = 1, fill ="both")

# setting canvas size and grid
canvas = tk.Canvas(tab1, width=1200, height=1200)
canvas.grid(rowspan=20, columnspan=20)

#For Training
tk.Label(tab1, text="Training the Model", font='customFont1', fg="black", bg="sky blue",
         width=15).grid(columnspan=1, row=0, column=0)

def browse_folder() :
    global folder_path
    filename = filedialog.askdirectory()
    folder_path.set(filename)

folder_path = tk.StringVar()
tk.Label(tab1, text="Path for Training the Model : ", font='customFont1', fg="black",
         width=20).grid(columnspan=1, row=1, column=0)
training_path = tk.Entry(tab1, width=60, textvariable=folder_path).grid(row=1, column=1)

browseBtn = tk.Button(tab1, text="Browse Folder", command=lambda:browse_folder(),
                      font='customFont1', bg="azure", fg="black", height=2, width=15)
browseBtn.grid(row=1, column=2)

start_column = tk.IntVar()
tk.Label(tab1, text="Signal Starts at Column : ", font='customFont1', fg="black",
         width=20).grid(columnspan=1, row=2, column=0)
signal_start = tk.Entry(tab1, width=5, textvariable=start_column).grid(row=2, column=1, sticky="W")

end_column = tk.IntVar()
tk.Label(tab1, text="Signal Ends at Column : ", font='customFont1', fg="black",
         width=20).place(x=500, y=170)
signal_end = tk.Entry(tab1, width=5, textvariable=end_column).place(x=700, y=170)

tk.Label(tab1, text="Accuracy obtained after validating the model with 20% of training data : ",
         font='customFont1', fg="black", width=40, wraplength=250).grid(columnspan=1, row=4, column=0)
string_variable = tk.StringVar()
tk.Entry(tab1, textvariable = string_variable, width=5).grid(columnspan=1, row=4, column=1)

def main():
    global training_set1, training_set2, training_set3, validation_set1, validation_set2, validation_set3
    global folder_path, start_column, end_column, classifier, string_variable
    global tp, tn, fp, fn, fdr, npv, tpr, tnr, roc, f1
    
    try :
        folder_path = folder_path.get()
    except :
        tk.messagebox.showwarning("Folder Path empty","Please specify folder path!")
    
    try :
        start_column = start_column.get()
    except :
        tk.messagebox.showwarning("Starting Column empty", "Please specify starting column!")
        
    try :
        end_column = end_column.get()
    except :
        tk.messagebox.showwarning("Ending Column empty", "Please specify ending column!")
    
    signal_width = range(start_column, end_column)

    object1_data = pd.DataFrame()
    object2_data = pd.DataFrame()
    object3_data = pd.DataFrame()  
    training_set = pd.DataFrame()
    training_set1 = pd.DataFrame() 
    training_set2 = pd.DataFrame() 
    training_set3 = pd.DataFrame()   
    validation_set = pd.DataFrame()
    validation_set1 = pd.DataFrame() 
    validation_set2 = pd.DataFrame() 
    validation_set3 = pd.DataFrame()   
    for subdir, dirs, files in os.walk(folder_path):
        for dir in dirs:
            subdir_path = os.path.join(subdir, dir)
            for subdir1, dirs1, files1 in os.walk(subdir_path):
                if dir == "Object 1":
                    for f in files1:
                            df = pd.read_excel(os.path.join(subdir1,f), header=None, usecols=signal_width)
                            df.insert(0, "Object Type", "Object 1")
                            object1_data = object1_data.append(df,ignore_index=True)
                    training_set1, validation_set1 = train_test_split(object1_data, test_size=0.2, random_state=21)
                    
                elif dir == "Object 2":
                    for f in files1:
                            df = pd.read_excel(os.path.join(subdir1,f), header=None, usecols=signal_width)
                            df.insert(0, "Object Type", "Object 2")
                            object2_data = object2_data.append(df,ignore_index=True)
                    training_set2, validation_set2 = train_test_split(object2_data, test_size=0.2, random_state=21)
                
                elif dir == "Object 3":
                    for f in files1:
                            df = pd.read_excel(os.path.join(subdir1,f), header=None, usecols=signal_width)
                            df.insert(0, "Object Type", "Object 3")
                            object3_data = object3_data.append(df,ignore_index=True)
                    training_set3, validation_set3 = train_test_split(object3_data, test_size=0.2, random_state=21)

    #Appending training data from all the object types        
    training_set = training_set.append(training_set1,ignore_index=True)
    training_set = training_set.append(training_set2,ignore_index=True)
    training_set = training_set.append(training_set3,ignore_index=True)

    #Appending validation data from all the object types  
    validation_set = validation_set.append(validation_set1,ignore_index=True)
    validation_set = validation_set.append(validation_set2,ignore_index=True)
    validation_set = validation_set.append(validation_set3,ignore_index=True)

    X_train = training_set.iloc[:, 1:].values
    Y_train = training_set.iloc[:, 0].values
    X_val = validation_set.iloc[:, 1:].values
    y_val = validation_set.iloc[:, 0].values

    #Initializing the MLPClassifier
    classifier = MLPClassifier(hidden_layer_sizes=(100,50,50), max_iter=300,activation='relu',
                               solver='adam',random_state=1)

    #Fitting the training data to the network
    classifier.fit(X_train, Y_train)

    #Predicting y for X_val
    y_pred = classifier.predict(X_val)

    #Comparing the predictions against the actual observations in y_val
    cm = confusion_matrix(y_pred, y_val)
    print(cm, "\n")

    #Ploting confusion matrix
    disp = plot_confusion_matrix(classifier, X_val, y_val,
                                    cmap=plt.cm.Blues)
    plt.show()

    #Printing the accuracy
    accuracy = accuracy_score(y_pred, y_val)
    print("Accuracy of MLPClassifier : ", accuracy, "\n")
    accuracy_percent = accuracy * 100
    string_variable = string_variable.set("{0:.2f}".format(accuracy_percent))

    #Printing F1
    print(classification_report(y_pred,y_val) , "\n")
    
    f1Score = f1_score(y_pred,y_val, average=None)
    f1Score = f1Score.tolist()
    f1Score = ['{:.2f}'.format(elem) for elem in f1Score]
    for i in range(3):
        f1[i].set(f1Score[i])
    
    FP = cm.sum(axis=0) - np.diag(cm) 
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)
    
    Tp = TP.tolist()
    for i in range(3):
        tp[i].set(Tp[i])
    
    Tn = TN.tolist()
    for i in range(3):
        tn[i].set(Tn[i])
        
    Fp = FP.tolist()
    for i in range(3):
        fp[i].set(Fp[i])
        
    Fn = FN.tolist()
    for i in range(3):
        fn[i].set(Fn[i])
    
    print("TruePostive(TP) : ", TP, "\n")
    print("TrueNegative(TN) : ", TN, "\n")
    print("FalsePostive(FP) : ", FP, "\n")
    print("False Negative(FN) : ", FN, "\n")

    # False discovery rate
    FDR = FP/(TP+FP)
    print("False Discovery Rate(FDR) : ", FDR, "\n")
    Fdr = FDR.tolist()
    Fdr = ['{:.2f}'.format(elem) for elem in Fdr]
    for i in range(3):
        fdr[i].set(Fdr[i])

    # Negative predictive value
    NPV = TN/(TN+FN)
    print("Negative Preductive Value(NPV) : ", NPV, "\n")
    Npv = NPV.tolist()
    Npv = ['{:.2f}'.format(elem) for elem in Npv]
    for i in range(3):
        npv[i].set(Npv[i])

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    print("True Positive Rate(TPR) : ", TPR, "\n")
    Tpr = TPR.tolist()
    Tpr = ['{:.2f}'.format(elem) for elem in Tpr]
    for i in range(3):
        tpr[i].set(Tpr[i])
     
    # Specificity, selectivity or true negative rate
    TNR = TN/(TN+FP) 
    print("True Negative Rate(TNR) : ", TNR, "\n")
    Tnr = TNR.tolist()
    Tnr = ['{:.2f}'.format(elem) for elem in Tnr]
    for i in range(3):
        tnr[i].set(Tnr[i])

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = 3

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(np.array(pd.get_dummies(y_val))[:, i], np.array(pd.get_dummies(y_pred))[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    lw=2
    plt.figure(figsize=(8,5))
    plt.plot(fpr["macro"], tpr["macro"],
            label='macro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["macro"]),
            color='green', linestyle='dotted', linewidth=4)

    colors = cycle(['purple', 'sienna', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                label='ROC curve of object {0} (area = {1:0.2f})'
                ''.format(i + 1 , roc_auc[i]))
        print("roc_auc_score of object " , i+1, ": ", roc_auc[i])
        
    print("\n")
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.annotate('Random Guess',(.5,.48))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()
    
    Roc = list(roc_auc.values())
    Roc = ['{:.2f}'.format(elem) for elem in Roc]
    for i in range(4):
        roc[i].set(Roc[i])
    tk.messagebox.showinfo("Completed", "Training and Validating model is completed successfully")
    

trainBtn = tk.Button(tab1, text="Train", command=lambda:main() ,font='customFont1', bg="azure", fg="black",height=3, width=15)
trainBtn.grid(columnspan=6, column=0, row=3)

#For Testing
tk.Label(tab1, text="Testing the Model", font='customFont1', fg="black", bg="sky blue",
         width=15).grid(columnspan=1, row=5, column=0)

def browse_file() :
    global file_path
    filename = filedialog.askopenfilename()
    file_path.set(filename)
    
file_path = tk.StringVar()  
tk.Label(tab1, text="Path for Testing the Model : ", font='customFont1', fg="black",
         width=20).grid(columnspan=1, row=6, column=0)
training_path = tk.Entry(tab1, width=60, textvariable=file_path).grid(row=6, column=1)

browseFileBtn = tk.Button(tab1, text="Browse File", command=lambda:browse_file(),
                          font='customFont1', bg="azure", fg="black", height=2, width=15)
browseFileBtn.grid(row=6, column=2)

def browse_outputFile() :
    global file_output_path
    outputfilename = filedialog.askdirectory()
    filename = "/Output.xlsx"
    file_output_path.set(outputfilename + filename)
    
file_output_path = tk.StringVar()
tk.Label(tab1, text="Path to save data file after testing : ", font='customFont1', fg="black",
         width=40).grid(columnspan=1, row=7, column=0)
training_path = tk.Entry(tab1, width=60, textvariable=file_output_path).grid(row=7, column=1)

browseOFileBtn = tk.Button(tab1, text="Browse Output Folder", command=lambda:browse_outputFile(), font='customFont1', bg="azure", fg="black", height=2, width=15)
browseOFileBtn.grid(row=7, column=2)

def testing():
    global file_path, file_output_path
    
    try :
        file_path = file_path.get()
    except :
        tk.messagebox.showwarning("Testing file path empty", "Please specify file path for Testing!")
       
    try :    
        file_output_path = file_output_path.get()
    except :
        tk.messagebox.showwarning("Output file path empty", "Please specify file path for the Output file!")
        
    signal_width = range(start_column, end_column)

    test_data = pd.read_excel(file_path, header = None, usecols = signal_width)

    y_test = classifier.predict(test_data.values)

    workbook = openpyxl.load_workbook(file_path)
    worksheet = workbook.worksheets[0]
    worksheet.insert_cols(3407)
    y = 1
    for x in range(len(y_test)):
        cell_to_write = worksheet.cell(row = y, column = 3407)
        cell_to_write.value = y_test[x]
        y += 1

    workbook.save(file_output_path)
    tk.messagebox.showinfo("Completed", "Testing is completed and Output file is generated")
    
testBtn = tk.Button(tab1, text="Test", command=lambda:testing() ,font='customFont1', bg="azure",
                    fg="black", height=3, width=15)
testBtn.grid(columnspan=6, column=0, row=8)

def Close():
    root.destroy()
    
# Measures

canvas = tk.Canvas(tab2, width=1200, height=1200)
canvas.grid(rowspan=30, columnspan=30)

tk.Label(tab2, text="Measures", font='customFont1', fg="black", bg="sky blue",
         width=13).grid(row=0, column=1)

tk.Label(tab2, text="Object 1", font='customFont1', fg="black", width = 15).grid(row=1, column=2)
tk.Label(tab2, text="Object 2", font='customFont1', fg="black", width = 15).grid(row=1, column=3)
tk.Label(tab2, text="Object 3", font='customFont1', fg="black", width = 15).grid(row=1, column=4)

tp = []
for i in range(3):
    tp.append(IntVar())
    
tk.Label(tab2, text="True Postive (TP) :", font='customFont1', fg="black", width = 15).grid(row=2, column=1)
tk.Entry(tab2, width=10, textvariable=tp[0]).grid(row=2, column=2)
tk.Entry(tab2, width=10, textvariable=tp[1]).grid(row=2, column=3)
tk.Entry(tab2, width=10, textvariable=tp[2]).grid(row=2, column=4)

tn = []
for i in range(3):
    tn.append(IntVar())
    
tk.Label(tab2, text="True Negative (TN) :", font='customFont1', fg="black", width = 15).grid(row=3, column=1)
tk.Entry(tab2, width=10, textvariable=tn[0]).grid(row=3, column=2)
tk.Entry(tab2, width=10, textvariable=tn[1]).grid(row=3, column=3)
tk.Entry(tab2, width=10, textvariable=tn[2]).grid(row=3, column=4)

fp = []
for i in range(3):
    fp.append(IntVar())
    
tk.Label(tab2, text="False Postive (FP) :", font='customFont1', fg="black", width = 15).grid(row=4, column=1)
tk.Entry(tab2, width=10, textvariable=fp[0]).grid(row=4, column=2)
tk.Entry(tab2, width=10, textvariable=fp[1]).grid(row=4, column=3)
tk.Entry(tab2, width=10, textvariable=fp[2]).grid(row=4, column=4)

fn = []
for i in range(3):
    fn.append(IntVar())
    
tk.Label(tab2, text="False Negative (FN) :", font='customFont1', fg="black", width = 15).grid(row=5, column=1)
tk.Entry(tab2, width=10, textvariable=fn[0]).grid(row=5, column=2)
tk.Entry(tab2, width=10, textvariable=fn[1]).grid(row=5, column=3)
tk.Entry(tab2, width=10, textvariable=fn[2]).grid(row=5, column=4)

fdr = []
for i in range(3):
    fdr.append(IntVar())

tk.Label(tab2, text="False Discovery Rate (FDR) :", font='customFont1', fg="black", width = 20).grid(row=6, column=1)
tk.Entry(tab2, width=10, textvariable=fdr[0]).grid(row=6, column=2)
tk.Entry(tab2, width=10, textvariable=fdr[1]).grid(row=6, column=3)
tk.Entry(tab2, width=10, textvariable=fdr[2]).grid(row=6, column=4)

npv = []
for i in range(3):
    npv.append(IntVar())

tk.Label(tab2, text="Negative Preductive Value (NPV) :", font='customFont1', fg="black", width = 23).grid(row=7, column=1)
tk.Entry(tab2, width=10, textvariable=npv[0]).grid(row=7, column=2)
tk.Entry(tab2, width=10, textvariable=npv[1]).grid(row=7, column=3)
tk.Entry(tab2, width=10, textvariable=npv[2]).grid(row=7, column=4)

tpr = []
for i in range(3):
    tpr.append(IntVar())

tk.Label(tab2, text="True Positive Rate (TPR) :", font='customFont1', fg="black", width = 20).grid(row=8, column=1)
tk.Entry(tab2, width=10, textvariable=tpr[0]).grid(row=8, column=2)
tk.Entry(tab2, width=10, textvariable=tpr[1]).grid(row=8, column=3)
tk.Entry(tab2, width=10, textvariable=tpr[2]).grid(row=8, column=4)

tnr = []
for i in range(3):
    tnr.append(IntVar())

tk.Label(tab2, text="True Negative Rate (TNR) :", font='customFont1', fg="black", width = 20).grid(row=9, column=1)
tk.Entry(tab2, width=10, textvariable=tnr[0]).grid(row=9, column=2)
tk.Entry(tab2, width=10, textvariable=tnr[1]).grid(row=9, column=3)
tk.Entry(tab2, width=10, textvariable=tnr[2]).grid(row=9, column=4)

f1 = []
for i in range(3):
    f1.append(IntVar())
    
tk.Label(tab2, text="F1 Score :", font='customFont1', fg="black", width = 15).grid(row=10, column=1)
tk.Entry(tab2, width=10, textvariable=f1[0]).grid(row=10, column=2)
tk.Entry(tab2, width=10, textvariable=f1[1]).grid(row=10, column=3)
tk.Entry(tab2, width=10, textvariable=f1[2]).grid(row=10, column=4)

roc = []
for i in range(4):
    roc.append(IntVar())
    
tk.Label(tab2, text="ROC Score :", font='customFont1', fg="black", width = 15).grid(row=11, column=1)
tk.Entry(tab2, width=10, textvariable=roc[0]).grid(row=11, column=2)
tk.Entry(tab2, width=10, textvariable=roc[1]).grid(row=11, column=3)
tk.Entry(tab2, width=10, textvariable=roc[2]).grid(row=11, column=4)

tk.Label(tab2, text="Precise Output", font='customFont1', fg="black", bg="sky blue",
         width=13).grid(row=0, column=5)

browseOp = tk.Button(tab2, text="Get Output Type", command=lambda:browse_Output_Type(),
                      font='customFont1', bg="azure", fg="black", height=2, width=15)
browseOp.grid(row=4, column=7)

tk.Label(tab2, text="Signal Row :", font='customFont1', fg="black", width = 15).grid(row=2, column=6)
SignalRow = tk.IntVar()
tk.Entry(tab2, textvariable = SignalRow, width=10).grid(row=2, column=7)

tk.Label(tab2, text="Object Type :", font='customFont1', fg="black", width = 15).grid(row=3, column=6)
ObjectType = tk.StringVar()
tk.Label(tab2, textvariable = ObjectType).grid(row=3, column=7)

def browse_Output_Type():
    global ObjectType, SignalRow
    wb = pd.DataFrame()
    
    OutputExcel = file_output_path
    wb = pd.read_excel(OutputExcel, header=None)
    
    signalrow = SignalRow.get()
    value = wb.iloc[signalrow-1, -1]
    ObjectType.set(value)

# Button for closing
#exit_button = tk.Button(root, text="Exit", command=Close, height=3, width=13).grid(columnspan=6, column=5, row=12)

root.mainloop()