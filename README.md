# COVID-19-CXR-CT-Classification

This is the source code of the  paper "Guangyu Jia, Hak-Keung Lam, Yujia Xu, Classification of COVID-19 Chest X-Ray and CT Images using a Type of Dynamic CNN Modification Method, Computers in Biology and Medicine, 2021, 104425, ISSN 0010-4825". (https://doi.org/10.1016/j.compbiomed.2021.104425)

## Data sources
Four classes of lung diseases are considered in this paper: COVID-19, bacterial pneumonia, viral pneumonia (except for COVID-19) and tuberculosis. In addition, healthy cases are included as the fifth class. The dataset used in this paper is the combination across four publicly available data sources  (DS):

- COVID-19 CXR images are from the open source GitHub repository https://github.com/ieee8023/covid-chestxray-dataset (DS 1), https://github.com/agchung/Actualmed-COVID-chestxray-dataset (DS 2), https://github.com/agchung/Figure1-COVID-chestxray-dataset (DS  3), https://www.kaggle.com/tawsifurrahman/covid19-radiography-database  (DS 4). 
- The images of tuberculosis positive cases are from the dataset [1] which consists of CXR databases respectively obtained in Shenzhen, China and Montgomery, USA (DS 5), and the data source https://tbportals.niaid.nih.gov/ which is from TB Portals Program, Office of Cyber Infrastructure and Computational Biology (OCICB), National Institute of Allergy and Infectious Diseases (NIAID) (DS 6).
- The bacterial and viral pneumonia CXR images are from https://www.kaggle.com/muhammadmasdar/pneumonia-virus-vs-pneumonia-bacteria/data (DS 7).
- The CXR images of normal controls are from https://www.kaggle.com/tawsifurrahman/covid19-radiography-database (DS 4).

The modified MobileNet and modified ResNet proposed in this paper respectively correspond to the files modified_mobilenet.ipynb and  modified-resnet.py. The code of the  robustness test is Robustness.ipynb. Files begin with ''model_selection...'' are models  used for comparative purpose illustrated in  Section 4 of the paper.


[1]  Jaeger, Stefan, et al. "Two public chest X-ray datasets for computer-aided screening of pulmonary diseases." Quantitative imaging in medicine and surgery 4.6 (2014): 475.
