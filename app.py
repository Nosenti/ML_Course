
import pickle
import streamlit as st
import pickle
import pandas as pd

data= pd.read_csv('https://raw.githubusercontent.com/Theoneste1/model/main/dataset.csv')
# data.info()

# data.isnull().sum()

null_counts = data.isnull().sum().sort_values()
selected = null_counts[null_counts < 8000 ]

percentage = 100 * data.isnull().sum() / len(data)


data_types = data.dtypes
# data_types

missing_values_table = pd.concat([null_counts, percentage, data_types], axis=1)
# missing_values_table

col=['CountryName','Date','StringencyLegacyIndexForDisplay','StringencyIndexForDisplay','ContainmentHealthIndexForDisplay','GovernmentResponseIndexForDisplay',
'EconomicSupportIndexForDisplay','C8_International travel controls','C1_School closing','C3_Cancel public events','C2_Workplace closing','C4_Restrictions on gatherings',
'C6_Stay at home requirements','C7_Restrictions on internal movement','H1_Public information campaigns','E1_Income support','C5_Close public transport','E2_Debt/contract relief','StringencyLegacyIndex','H3_Contact tracing','StringencyIndex','ContainmentHealthIndex','E4_International support','EconomicSupportIndex','E3_Fiscal measures','H5_Investment in vaccines','ConfirmedCases','ConfirmedDeaths']

newdataset=data[col]
newdataset= newdataset.dropna()

from sklearn.preprocessing import LabelEncoder
newdataset['CountryName']=LabelEncoder().fit_transform(newdataset['CountryName'])

X=newdataset[['CountryName','StringencyLegacyIndexForDisplay','StringencyIndexForDisplay',	'StringencyIndex','StringencyLegacyIndex','ContainmentHealthIndexForDisplay','ContainmentHealthIndex','GovernmentResponseIndexForDisplay','ConfirmedCases','ConfirmedDeaths','EconomicSupportIndexForDisplay','E2_Debt/contract relief','EconomicSupportIndex','C3_Cancel public events','C1_School closing']]

from sklearn.feature_selection import VarianceThreshold

selector = VarianceThreshold()
x= selector.fit_transform(X)

df_first_half = x[:5000]
df_second_half = x[5000:]

# """Create clusters/classes of similar records using features selected in (1),  use an unsupervised learning algorithm of your choice."""

# Commented out IPython magic to ensure Python compatibility.
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import streamlit as st
model = KMeans(n_clusters = 6)

pca = PCA(n_components=2).fit(x)
pca_2d = pca.transform(x)

model.fit(pca_2d)

labels = model.predict(pca_2d)

xs = pca_2d[:, 0]
ys = pca_2d[:, 1]
plt.scatter(xs, ys, c = labels)
plt.scatter(model.cluster_centers_[:,0],model.cluster_centers_[:,1],color='purple',marker='*',label='centroid')

kmeans = KMeans(n_clusters=10)
kmeans.fit(df_first_half)
plt.scatter(df_first_half[:,0],df_first_half[:,1], c=kmeans.labels_, cmap='rainbow')

range_n_clusters = [2, 3, 4, 5, 6]

#km.cluster_centers_

from sklearn import datasets
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
transformed = scaler.fit_transform(x)
# Plotting 2d t-Sne
x_axis = transformed[:,0]
y_axis = transformed[:,1]

kmeans = KMeans(n_clusters=4, random_state=42,n_jobs=-1)
y_pred =kmeans.fit_predict(transformed)

predicted_label = kmeans.predict([[7,7.2, 3.5, 0.8, 1.6,7.2, 3.5, 0.8, 1.6,7.2, 3.5, 0.8, 1.67, 7.2, 3.5]])
predicted_label

# """Create a platform where new records of countries can be classified in the clusters"""



# Commented out IPython magic to ensure Python compatibility.
# %%writefile app.py
import streamlit as st
import pickle
import numpy as np

# kmeans=pickle.load(open('unsupervisedmodels.pkl','rb')) 


def predict_kmeans(CountryName, Date, C1_School closing, C1_Flag,
       C2_Workplace closing, C2_Flag, C3_Cancel public events,
       C4_Restrictions on gatherings, C4_Flag, C5_Close public transport,
       C5_Flag, C6_Stay at home requirements, C6_Flag,
       C7_Restrictions on internal movement, C7_Flag,
       C8_International travel controls, E1_Income support, E1_Flag,
       E2_Debt/contract relief, E3_Fiscal measures,
       E4_International support, H1_Public information campaigns,
       H1_Flag, H2_Testing policy, H3_Contact tracing,
       H5_Investment in vaccines, ConfirmedCases, StringencyIndex):
    input=np.array([[CountryName, Date, C1_School closing, C1_Flag,
       C2_Workplace closing, C2_Flag, C3_Cancel public events,
       C4_Restrictions on gatherings, C4_Flag, C5_Close public transport,
       C5_Flag, C6_Stay at home requirements, C6_Flag,
       C7_Restrictions on internal movement, C7_Flag,
       C8_International travel controls, E1_Income support, E1_Flag,
       E2_Debt/contract relief, E3_Fiscal measures,
       E4_International support, H1_Public information campaigns,
       H1_Flag, H2_Testing policy, H3_Contact tracing,
       H5_Investment in vaccines, ConfirmedCases, StringencyIndex]]).astype(np.float64)
    prediction=kmeans.predict(input)
    return prediction

def main():
    st.title("Classification of Countries vs Covid")
    html_temp = """
    <div style="background-color:#996515 ;padding:10px">
    <h2 style="color:white;text-align:center;">Unsupervised KMeans Algorithm </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    CountryName = st.text_input("CountryName",key='0')
    Date = st.text_input("StringencyLegacyIndexForDisplay",key='1')
    C1_School closing = st.text_input("StringencyIndexForDisplay","Type Here",key='2')
    C1_Flag = st.text_input("StringencyIndex",,key='3')
    C2_Workplace closing = st.text_input("StringencyLegacyIndex",,key='4')
    C2_Flag = st.text_input("C2_Flag",key='5')
     C7_Restrictions on internal movement = st.text_input(" C7_Restrictions on internal movement",key='6')
    C7_Flag = st.text_input("C7_Flag",key='7')
    C8_International travel controls = st.text_input("C8_International travel controls",key='8')
    E1_Income support = st.text_input("E1_Income support",key='9')
    E1_Flag = st.text_input("E1_Flag",key='9')
    E2_Debtcontractrelief = st.text_input("E2_Debtcontractrelief",key='10')
    EconomicSupportIndex = st.text_input("EconomicSupportIndex",key='11')
    C3_Cancelpublicevents = st.text_input("C3_Cancelpublicevents",key='12')
    C1_Schoolclosing = st.text_input("C1_Schoolclosing","Type Here",key='13')

    safe_html="""  
      <div style="background-color:green;padding:10px >
       <h2 style="color:white;text-align:center;"></h2>
       </div>
    """
    danger_html="""  
      <div style="background-color:green;padding:10px >
       <h2 style="color:red ;text-align:center;"></h2>
       </div>
    """

    if st.button("Predict"):
        output=predict_kmeans(CountryName,StringencyLegacyIndexForDisplay,StringencyIndexForDisplay,	StringencyIndex,StringencyLegacyIndex,ContainmentHealthIndexForDisplay,ContainmentHealthIndex,GovernmentResponseIndexForDisplay,ConfirmedCases,ConfirmedDeaths,EconomicSupportIndexForDisplay,E2_Debtcontractrelief,EconomicSupportIndex,C3_Cancelpublicevents,C1_Schoolclosing)
        st.success('This country located in this cluster {}'.format(output))


if __name__=='__main__':
    main()



