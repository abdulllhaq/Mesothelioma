import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
from PIL import Image



#App description
st.markdown('''
#  Mesothelioma Detector 
This app detects if you have Breast Cancer based on Machine Learning!
- App built by Pranav Sawant and Anshuman Shukla of Team Skillocity.
- Dataset Creators: 
- Dr. William H. Wolberg, General Surgery Dept. University of Wisconsin, Clinical Sciences Center Madison, WI 53792
- W. Nick Street, Computer Sciences Dept. University of Wisconsin, 1210 West Dayton St., Madison, WI 53706
- Olvi L. Mangasarian, Computer Sciences Dept. University of Wisconsin, 1210 West Dayton St., Madison, WI 53706
- Note: User inputs are taken from the sidebar. It is located at the top left of the page (arrow symbol). The values of the parameters can be changed from the sidebar.  
''')
st.write('---')

df = pd.read_csv(r'data.csv')

#titles
st.sidebar.header('Patient Data')
st.subheader('Training Dataset')
st.write(df.describe())


#train data. Fun!
x = df.drop(['Outcome'], axis = 1)
y = df.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)


#User reports
def user_report():
  Age = st.sidebar.slider('Age', 0,100, 54)
  Platelet_Count = st.sidebar.slider('Platelet Count', 0,30, 15 )
  Blood_Lactic_Dehydrogenise = st.sidebar.slider('Blood Lactic Dehydrogenise', 0,40, 20 )
  Alkaline_Phosphatise = st.sidebar.slider('Alkaline Phosphatise', 40,200, 92 )
  Total_Protein = st.sidebar.slider('Total Protein', 0.0,0.25, 0.1 )
  Albumin = st.sidebar.slider('Albumin', 0.0,0.4, 0.1 )
  Glucose = st.sidebar.slider('Glucose', 0.0,0.5, 0.1 )
  Pleural_Lactic_Dehydrogenise = st.sidebar.slider('Pleural Lactic Dehydrogenise', 0.0,0.25, 0.05 )
  Pleural_Protein = st.sidebar.slider('Pleural Protein', 0.0,0.4, 0.2 )
  Pleural_Albumin = st.sidebar.slider('Pleural Albumin', 0.0,0.1, 0.06 )
  Pleural_Glucose = st.sidebar('Pleural Glucose', 0.0,0.1, 0.06)
  Creactive_Protein = st.sidebar('C-reactive Protein', 0.0,0.1, 0.06)
  
  
  
  
  
  
  user_report_data = {
      'Age':Age,
      'Platelet_Count':Platelet_Count,
      'Blood_Lactic_Dehydrogenise':Blood_Lactic_Dehydrogenise,
      'Alkaline_Phosphatise':Alkaline_Phosphatise,
      'Total_Protein':Total_Protein,
      'Albumin':Albumin,
      'Glucose':Glucose,
      'Pleural_Lactic_Dehydrogenise':Pleural_Lactic_Dehydrogenise,
      'Pleural_Protein':Pleural_Protein,
      'Pleural_Albumin':Pleural_Albumin,
      'Pleural_Glucose':Pleural_Glucose,
      'Creactive_Protein':Creactive_Protein,
        
  }
  report_data = pd.DataFrame(user_report_data, index=[0])
  return report_data





user_data = user_report()
st.subheader('Patient Data')
st.write(user_data)





rf  = RandomForestClassifier()
rf.fit(x_train, y_train)
user_result = rf.predict(user_data)



#Visualizations, this is where the beauty begins.
st.title('Graphical Patient Report')



if user_result[0]==0:
  color = 'blue'
else:
  color = 'red'

#Good old glucose
st.header('Platelet Count Value Graph (Yours vs Others)')
fig_Radius = plt.figure()
ax3 = sns.scatterplot(x = 'Age', y = 'Platelet Count', data = df, hue = 'Outcome' , palette='Purples')
ax4 = sns.scatterplot(x = user_data['Age'], y = user_data['Platelet_Count'], s = 150, color = color)
plt.xticks(np.arange(0,100,5))
plt.yticks(np.arange(0,50,5))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_Radius)


#Insulin
st.header('Blood Lactic Dehydrogenise Value Graph (Yours vs Others)')
fig_Texture = plt.figure()
ax9 = sns.scatterplot(x = 'Age', y = 'Blood Lactic Dehydrogenise', data = df, hue = 'Outcome', palette='rainbow')
ax10 = sns.scatterplot(x = user_data['Age'], y = user_data['Blood_Lactic_Dehydrogenise'], s = 150, color = color)
plt.xticks(np.arange(0,100,5))
plt.yticks(np.arange(0,50,5))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_Texture)


#Famous saying BP
st.header('Alkaline Phosphatise Value Graph (Yours vs Others)')
fig_Perimeter = plt.figure()
ax5 = sns.scatterplot(x = 'Age', y = 'Alkaline Phosphatise', data = df, hue = 'Outcome', palette='Blues')
ax6 = sns.scatterplot(x = user_data['Age'], y = user_data['Alkaline_Phosphatise'], s = 150, color = color)
plt.xticks(np.arange(0,100,5))
plt.yticks(np.arange(0,200,10))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_Perimeter)


#Did'nt even know this before nutrition training 
st.header('Total Protein Value Graph (Yours vs Others)')
fig_Area = plt.figure()
ax11 = sns.scatterplot(x = 'Age', y = 'Total Protein', data = df, hue = 'Outcome', palette='Greens')
ax12 = sns.scatterplot(x = user_data['Age'], y = user_data['Total_Protein'], s = 150, color = color)
plt.xticks(np.arange(0,100,5))
plt.yticks(np.arange(100,2500,100))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_Area)


#Something new, cool
st.header('Albumin Value Graph (Yours vs Others)')
fig_Smoothness = plt.figure()
ax13 = sns.scatterplot(x = 'Age', y = 'Albumin', data = df, hue = 'Outcome', palette='rocket')
ax14 = sns.scatterplot(x = user_data['Age'], y = user_data['Albumin'], s = 150, color = color)
plt.xticks(np.arange(0,100,5))
plt.yticks(np.arange(0.0,0.25,0.0125))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_Smoothness)


#Don't even know how thats related to diabetes.The dataset was females only though
st.header('Glucose count Graph (Yours vs Others)')
fig_Compactness = plt.figure()
ax1 = sns.scatterplot(x = 'Age', y = 'Glucose', data = df, hue = 'Outcome', palette = 'magma')
ax2 = sns.scatterplot(x = user_data['Age'], y = user_data['Glucose'], s = 150, color = color)
plt.xticks(np.arange(0,100,5))
plt.yticks(np.arange(0.0,0.4,0.02))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_Compactness)


#Wonder how people measure that 
st.header('Pleural Lactic Dehydrogenise Value Graph (Yours vs Others)')
fig_Concavity = plt.figure()
ax7 = sns.scatterplot(x = 'Age', y = 'Pleural Lactic Dehydrogenise', data = df, hue = 'Outcome', palette='Reds')
ax8 = sns.scatterplot(x = user_data['Age'], y = user_data['Pleural_Lactic_Dehydrogenise'], s = 150, color = color)
plt.xticks(np.arange(0,100,5))
plt.yticks(np.arange(0.0,0.5,0.025))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_Concavity)



st.header('Pleural Protein Value Graph (Yours vs Others)')
fig_Concavepoints = plt.figure()
ax7 = sns.scatterplot(x = 'Age', y = 'Pleural Protein', data = df, hue = 'Outcome', palette='mako')
ax8 = sns.scatterplot(x = user_data['Age'], y = user_data['Pleural_Protein'], s = 150, color = color)
plt.xticks(np.arange(0,100,5))
plt.yticks(np.arange(0.0,0.25,0.0125))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_Concavepoints)




st.header('Pleural Albumin Value Graph (Yours vs Others)')
fig_Symmetry = plt.figure()
ax7 = sns.scatterplot(x = 'Age', y = 'Pleural Albumin', data = df, hue = 'Outcome', palette='flare')
ax8 = sns.scatterplot(x = user_data['Age'], y = user_data['Pleural_Albumin'], s = 150, color = color)
plt.xticks(np.arange(0,100,5))
plt.yticks(np.arange(0.0,0.4,0.02))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_Symmetry)


st.header('Pleural Glucose Value Graph (Yours vs Others)')
fig_FractalDimension = plt.figure()
ax7 = sns.scatterplot(x = 'Age', y = 'Pleural Glucose', data = df, hue = 'Outcome', palette='crest')
ax8 = sns.scatterplot(x = user_data['Age'], y = user_data['Pleural_Glucose'], s = 150, color = color)
plt.xticks(np.arange(0,100,5))
plt.yticks(np.arange(0.0,0.1,0.005))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_FractalDimension)

st.header('C-reactive Protein Value Graph (Yours vs Others)')
fig_FractalDimension = plt.figure()
ax7 = sns.scatterplot(x = 'Age', y = 'C-reactive Protein', data = df, hue = 'Outcome', palette='crest')
ax8 = sns.scatterplot(x = user_data['Age'], y = user_data['Creactive_Protein'], s = 150, color = color)
plt.xticks(np.arange(0,100,5))
plt.yticks(np.arange(0.0,0.1,0.005))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_FractalDimension)








#Finally!
st.subheader('Your Report: ')
output=''
if user_result[0]==0:
  output = 'Congratulations, you do not have  Mesothelioma'
else:
  output = 'Unfortunately, you do have Mesothelioma'
st.title(output)

st.subheader('Accuracy: ')
st.write(str(accuracy_score(y_test, rf.predict(x_test))*100)+'%')

#Most important for users
st.subheader('Lets raise awareness for Breast Cancer and show our support for cancer awareness and help many patients around the world.')
st.write("Breast Cancer Awareness Month: October")

#st.sidebar.subheader("""An article about this app: https://proskillocity.blogspot.com/2021/04/official-launch-of-our-first-web-app.html""")
st.write("Dataset citation : W.N. Street, W.H. Wolberg and O.L. Mangasarian. Nuclear feature extraction for breast tumor diagnosis. IS&T/SPIE 1993 International Symposium on Electronic Imaging: Science and Technology, volume 1905, pages 861-870, San Jose, CA, 1993. O.L. Mangasarian, W.N. Street and W.H. Wolberg. Breast cancer diagnosis and prognosis via linear programming. Operations Research, 43(4), pages 570-577, July-August 1995.")
#st.write("Original owners of the dataset: Original owners: National Institute of Diabetes and Digestive and Kidney Diseases (b) Donor of database: Vincent Sigillito (vgs@aplcen.apl.jhu.edu) Research Center, RMI Group Leader Applied Physics Laboratory The Johns Hopkins University Johns Hopkins Road Laurel, MD 20707 (301) 953-6231 © Date received: 9 May 1990")
st.write("Dataset License: Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)")

st.write("Disclaimer: This is just a learning project based on one particular dataset so please do not depend on it to actually know if you have breast cancer or not. It might still be a false positive or false negative. A doctor is still the best fit for the determination of such diseases.")
image = Image.open('killocity (3).png')

st.image(image, use_column_width=True)

st.sidebar.subheader("An article about this app: https://proskillocity.blogspot.com/2021/06/breast-cancer-detection-web-app.html")
