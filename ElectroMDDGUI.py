import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, qApp, QAction, QMainWindow, QMenu, QLabel, QTextEdit, QLineEdit, QScrollArea, QGridLayout, QFileDialog, QDialog
from PyQt5.QtGui import QIcon 
from PyQt5.uic import loadUi
from pathlib import Path
import pandas as pd 
import numpy as np
import os, csv

    
class Example(QWidget): 

    def __init__(self):
        super().__init__()
        self.initUI() 
        self.show()

    def initUI(self): 
        
        #6/8/2021: Here are some lines to create the main Grid Layout widgets for the GUI. The remainder of the necessary widgets for the GUI will be placed onto this 
        #Grid Layout. I think I'll make the dimensions 13x5 for now.
        self.selectCSVFilesButton = QPushButton('Select EEG .csv Files', self) 
        self.csvFilesSelectedLabel = QLabel('EEG .CSV File Paths Selected:')
        self.csvFilesSelectedTextLine = QLineEdit()
        self.runDiagnosticsButton = QPushButton('Run Diagnostics') 
        self.diagnosticsWindowLabel = QLabel('Diagnostics Window')
        #Now, I'll create a Parent QScrollArea() widget and a Child QTextEdit() Widget to be contained within it. 
        #According to the main website for the QScrollsArea() widget (Link: https://doc.qt.io/qt-5/qscrollarea.html), 
        #the parent class' .setWidget() method must be used to set the child widget whose contents will be displayed in the 
        #QScrollArea() object. 
        self.diagnosticsWindowTextEditContents = QTextEdit()
        self.diagnosticsWindowTextEditContents.setReadOnly(True)
        self.diagnosticsWindowScrollArea = QScrollArea() 
        self.diagnosticsWindowScrollArea.setWidget(self.diagnosticsWindowTextEditContents)
        #Creating the Main Grid Layout and placing the Widgets in the desired positions within it: 
        #6/8/2021 Note: I was misunderstanding the parameters of the .addWidget class earlier today. It's important to remember that the 
        #first tuple of position parameter pairs specifies the location of the starting cell in the grid where the widget will be placed. Then, 
        #the second of the pairs of tuples specifies the number of rows and columns FROM that first cell that the widget will span over. 
        self.electroMDDGrid = QGridLayout() 
        self.setLayout(self.electroMDDGrid)

        self.electroMDDGrid.addWidget(self.selectCSVFilesButton, 1, 2, 1, 3)
        self.electroMDDGrid.addWidget(self.csvFilesSelectedLabel, 2, 2, 1, 3)
        self.electroMDDGrid.addWidget(self.csvFilesSelectedTextLine, 3, 1, 1, 5)
        self.electroMDDGrid.addWidget(self.runDiagnosticsButton, 6, 2, 1, 3)
        self.electroMDDGrid.addWidget(self.diagnosticsWindowLabel, 8, 2, 1, 3)
        self.electroMDDGrid.addWidget(self.diagnosticsWindowScrollArea, 9, 1, 7, 5)
        
        #Here is a tutorial for the QFileDialog Widget and its methods; this class and 
        #its methods are used to open the File System native to the user's operating system, 
        #reading file paths and file contents as they desire. 6/9/2021: I'll
        #still try and use this class and its methods to import EEG .csv files instead of 
        #having the user manually imput File Paths, at least for now. 
        #Tutorial Links: https://www.tutorialspoint.com/pyqt/pyqt_qfiledialog_widget.htm,
        #https://zetcode.com/gui/pyqt5/dialogs/

        #Creating three lists to contain the names of: (i) The EEG .csv file names, (ii) The names of the numpy Arrays I'll need to create 
        #to apply the classification algorithm(s), and (iii) The names of the pandas DataFrames that will serve as good general-purpose
        #storage data structures for the different EEG files. 
        csvFileNameList = []
        npEEGArrayNameList = []
        eegDFNameList = [] 
        
        #I'll also want two lists to contain the arrays and DataFrames themselves. That way, I can iteratively compute the diagnoses via the 
        #classification algorithm(s) with the highest performance/accuracy for all of the EEG files the user selects. 
        npEEGArrayList = []
        eegDFList = []


        def ReadEEGCSVFile(): 

            csvFileName = QFileDialog.getOpenFileName(self, 'Open EEG Data CSV Files')

            csvFileNameList.append(str(csvFileName))
            npEEGArrayNameList.append(str(csvFileName) + "Numpy Array")
            eegDFNameList.append(str(csvFileName) + "DataFrame")

            if csvFileName[0]:
              eegDFList.append(pd.read_csv(csvFileName[0]))
              npEEGArrayList.append(pd.read_csv(csvFileName[0]).to_numpy())

              #Testing Print Statements: 
              #print(eegDFList[len(eegDFList)-1])
              #print(npEEGArrayList[len(npEEGArrayList)-1])
              
              
            #I should (eventually) make this Line Edit another Scrollable Text Area of some sort, as it will become 
            #irritating to have to scroll through more than a few .csv files in a single line. 
            for fileName in csvFileNameList: 
              self.csvFilesSelectedTextLine.setText(str(csvFileNameList))

            return npEEGArrayList


        #def RunMDDHCDiagnoses(): 
     
        #Step 1: Computing the DFTs of the EEG Data Files from the Cavanagh Study and the Unlabeled Data Selected by the User: 
        #There are only a few ways I can really think this would work from the user's end. I don't know if PyQt5 has a good feature allowing 
        #multiple .csv files to be selected and read at once; currently, my program only allows one file to be selected read at a time. This 
        #would be irritating for a user who must select dozens of (MDD/HC)-labeled .csv files manually to diagnose their patients or other individuals 
        #of interest as MDD or HC. However, this is the only way I know of to do this right now.

        #Step 2: 
              
        #Main Event, Signal, and File Handling of the .CSV Data: 
        self.selectCSVFilesButton.clicked.connect(ReadEEGCSVFile)
        #self.runDiagnosticsButton.clicked.connect(RunMDDHCDiagnoses)

        #These are just a few lines for setting the GUI Window's Dimensions, 
        #titling the window, and then displaying it upon executing the program. 
        self.setGeometry(1000, 1000, 325, 325)
        self.setWindowTitle('ElectroMDD')


    
def main():
    #6/8/2021: I assume most of the .csv file handling execution and calling of any non-PyQt5 widget methods will be placed in the main method. 
    #This category of methods will include all of the ML Classification and similar methods currently conatined within the Model Validation file. 

    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()