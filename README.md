# GANN
The Generative AOD Adversarial Neural Network (GANN) is a neural network model used for satellite aerosol optical depth (AOD) retrieval. 

The GANN_Train.py can be used to train your GANN model based on your sample. The flowing Sample section provides some data as an example.

Our aim is the improvement of accuracy and precision in the satellite remote sensing parameter retrieval based on machine learning.

If you have any questions about the code or data, please contact us by email: fyl1217@outlook.com.

# Training flow

GANN construction：
![image](Model.png.jpg)

GANN training flow：
![image](Training_flow.png.jpg)

# Sameple example

The data named '2014_2018_sample_550_Selection.csv' is the AOD samples matching Terra (MODIS) and AERONET sites from 2014 to 2018 worldwide. In addition, there are Meteorological data, such as temperature, pressure， U-wind and V-wind， that can be used to improve the accuracy of AOD retrieval.

![image](image.png)
