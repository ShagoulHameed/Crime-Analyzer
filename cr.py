import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import folium_static
from folium.plugins import HeatMap
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.utils.multiclass import unique_labels


#st.set_option('deprecation.showPyplotGlobalUse', False)
#comment this abve line not working in stremli app

@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    return df


@st.cache_data 
def clean_data(df):
    fill_mode_columns = [
        'Location Description', 'Ward', 'Community Area', 'X Coordinate', 
        'Y Coordinate', 'Latitude', 'Longitude', 'Location'
    ]
    for col in fill_mode_columns:
        if col in df.columns:
            mode_value = df[col].mode()
            if not mode_value.empty:
                df[col].fillna(mode_value[0], inplace=True)
    return df

def preprocess_data(df):
    
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    df.dropna(subset=['Date'], inplace=True)
    
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['Hour'] = df['Date'].dt.hour

    df1 = df.groupby(['Month', 'Day', 'District', 'Hour'], as_index=False).agg({"Primary Type": "count"})
    df1 = df1.sort_values(by=['District'], ascending=False)
    df1 = df1[['Month', 'Day', 'Hour', 'Primary Type', 'District']]

    def crime_rate_assign(x):
        if x <= 7:
            return 0
        elif 7 < x <= 15:
            return 1
        else:
            return 2

    df1['Warning'] = df1['Primary Type'].apply(crime_rate_assign)
    df1 = df1[['Month', 'Day', 'Hour', 'District', 'Primary Type', 'Warning']]
    
    return df1

def logistic_regression(X_train, X_test, y_train, y_test):
    logistic_model = LogisticRegression(random_state=1, max_iter=200)
    logistic_model.fit(X_train, y_train)
    y_pred_logistic = logistic_model.predict(X_test)

    accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
    st.write(f"Logistic Regression Accuracy: {accuracy_logistic * 100:.2f}%")

    conf_matrix_logistic = confusion_matrix(y_test, y_pred_logistic)
    conf_matrix_logistic_df = pd.DataFrame(conf_matrix_logistic, index=unique_labels(y_test, y_pred_logistic), columns=unique_labels(y_test, y_pred_logistic))
    st.write("### Confusion Matrix - Logistic Regression")
    st.write(conf_matrix_logistic_df)

    st.write("### Classification Report - Logistic Regression")
    st.write(classification_report(y_test, y_pred_logistic))

    return logistic_model

def decision_tree(X_train, X_test, y_train, y_test):
    decision_tree_model = DecisionTreeClassifier(random_state=10)
    decision_tree_model.fit(X_train, y_train)

    plt.figure(figsize=(12, 12))
    plot_tree(decision_tree_model, feature_names=X_train.columns, filled=True, rounded=True)
    st.pyplot(plt)

    predictions = decision_tree_model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    st.write(f"Decision Tree Accuracy: {accuracy * 100:.2f}%")

    conf_matrix = confusion_matrix(y_test, predictions)
    conf_matrix_df = pd.DataFrame(conf_matrix, index=unique_labels(y_test, predictions), columns=unique_labels(y_test, predictions))
    st.write("### Confusion Matrix - Decision Tree")
    st.write(conf_matrix_df)

    st.write("### Classification Report - Decision Tree")
    st.write(classification_report(y_test, predictions))

    return decision_tree_model

def support_vector_classifier(X_train, X_test, y_train, y_test):
    svc_model = SVC(kernel='linear', random_state=1)
    svc_model.fit(X_train, y_train)
    y_pred_svc = svc_model.predict(X_test)

    accuracy_svc = accuracy_score(y_test, y_pred_svc)
    st.write(f"Support Vector Classifier Accuracy: {accuracy_svc * 100:.2f}%")

    conf_matrix_svc = confusion_matrix(y_test, y_pred_svc)
    conf_matrix_svc_df = pd.DataFrame(conf_matrix_svc, index=unique_labels(y_test, y_pred_svc), columns=unique_labels(y_test, y_pred_svc))
    st.write("### Confusion Matrix - Support Vector Classifier")
    st.write(conf_matrix_svc_df)

    st.write("### Classification Report - Support Vector Classifier")
    st.write(classification_report(y_test, y_pred_svc))

    return svc_model


def predict_future_crimes(model, X_new):
    prediction = model.predict(X_new)
    st.write("### Predicted Crime Incidents")
    st.write(prediction) 



def temporal_analysis(df):
    crimes_per_year = df['Year'].value_counts().sort_index()
    st.write("### Number of Crimes Per Year in Chicago")
    st.bar_chart(crimes_per_year)

def peak_crime_hours(df):
    df['Hour'] = pd.to_datetime(df['Date']).dt.hour
    crimes_per_hour = df['Hour'].value_counts().sort_index()
    st.write("### Number of Crimes Per Hour")
    st.bar_chart(crimes_per_hour)

def geospatial_analysis(df):
    map_hooray = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=11)
    heat_data = [[row['Latitude'], row['Longitude']] for index, row in df.iterrows() if not pd.isnull(row['Latitude']) and not pd.isnull(row['Longitude'])]
    HeatMap(heat_data).add_to(map_hooray)
    st.write("### Crime Hotspots in Chicago")
    folium_static(map_hooray)

def district_crime_rates(df):
    crimes_per_district = df.groupby(['District', 'Primary Type']).size().unstack().fillna(0)
    st.write("### Crime Rates by District")
    st.bar_chart(crimes_per_district)

def crime_type_distribution(df):
    crime_type_counts = df['Primary Type'].value_counts()
    st.write("### Distribution of Crime Types")
    st.bar_chart(crime_type_counts)

def severity_analysis(df):
    severe_crimes = ['HOMICIDE', 'ASSAULT', 'OFFENSE INVOLVING CHILDREN', 'ROBBERY', 'CRIM SEXUAL ASSAULT', 'OFFENSE INVOLVING CHILDREN', 'SEX OFFENSE', 'WEAPONS VIOLATION']
    df['Severity'] = df['Primary Type'].apply(lambda x: 'Severe' if x in severe_crimes else 'Less Severe')
    severity_counts = df['Severity'].value_counts()
    plt.figure(figsize=(6, 6))
    severity_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90)
    plt.title('Severity Analysis of Crimes')
    plt.ylabel('')
    st.write("### Severity Analysis of Crimes")
    st.pyplot(plt)

def arrest_analysis(df):
    arrest_rates_by_type = df.groupby('Primary Type')['Arrest'].mean() * 100
    arrest_rates_by_type.sort_values(ascending=False, inplace=True)
    plt.figure(figsize=(10, 6))
    arrest_rates_by_type.plot(kind='bar')
    plt.title('Arrest Rates by Crime Type')
    plt.xlabel('Crime Type')
    plt.ylabel('Arrest Rate (%)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.write("### Arrest Rates by Crime Type")
    st.pyplot(plt)

def domestic_analysis(df):
    domestic_counts = df['Domestic'].value_counts()
    plt.figure(figsize=(6, 6))
    domestic_counts.plot(kind='pie', labels=['Non-Domestic', 'Domestic'], autopct='%1.1f%%', startangle=90)
    plt.title('Domestic vs. Non-Domestic Crimes')
    plt.ylabel('')
    st.write("### Domestic vs. Non-Domestic Crimes")
    st.pyplot(plt)

def location_analysis(df):
    location_counts = df['Location Description'].value_counts().head(10)
    plt.figure(figsize=(10, 6))
    location_counts.plot(kind='barh')
    plt.title('Top 10 Crime Locations')
    plt.xlabel('Frequency')
    plt.ylabel('Location Description')
    plt.tight_layout()
    st.write("### Top 10 Crime Locations")
    st.pyplot(plt)

def crime_by_location(df):
    location_crime_types = df.groupby(['Location Description', 'Primary Type']).size().unstack().fillna(0)
    top_locations = df['Location Description'].value_counts().head(5).index
    plt.figure(figsize=(12, 6))
    location_crime_types.loc[top_locations].plot(kind='bar', stacked=True)
    plt.title('Crime Types by Location')
    plt.xlabel('Location Description')
    plt.ylabel('Frequency')
    plt.legend(title='Crime Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.write("### Crime Types by Location")
    st.pyplot(plt)

def seasonal_trends(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month

    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'

    df['Season'] = df['Month'].apply(get_season)
    seasonal_crime_counts = df.groupby(['Season', 'Primary Type']).size().unstack().fillna(0)
    seasonal_crime_counts.plot(kind='bar', figsize=(12, 6), stacked=True)
    plt.title('Seasonal Trends in Crime Types')
    plt.xlabel('Season')
    plt.ylabel('Number of Crimes')
    plt.legend(title='Crime Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    st.write("### Seasonal Trends in Crime Types")
    st.pyplot(plt)




#  Streamlit 
def main():
    st.title("Chicago Crime Analysis")
    file_path = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

    if file_path is not None:
        df = load_data(file_path)
        df = clean_data(df)
        df1 = preprocess_data(df)

        st.sidebar.subheader("Select Analysis")
        analysis_option = st.sidebar.selectbox(
            "Choose an analysis",
            (
                "Temporal Analysis", 
                "Peak Crime Hours", 
                "Geospatial Analysis", 
                "District Crime Rates",
                "Crime Type Distribution", 
                "Severity Analysis",
                "Arrest Analysis",
                "Domestic Analysis",
                "Location Analysis",
                "Crime by Location",
                "Seasonal Trends",
                "Logistic Regression",
                "Decision Tree",
                "Support Vector Classifier"
            )
        )

        if analysis_option == "Temporal Analysis":
            temporal_analysis(df)
        elif analysis_option == "Peak Crime Hours":
            peak_crime_hours(df)
        elif analysis_option == "Geospatial Analysis":
            geospatial_analysis(df)
        elif analysis_option == "District Crime Rates":
            district_crime_rates(df)
        elif analysis_option == "Crime Type Distribution":
            crime_type_distribution(df)
        elif analysis_option == "Severity Analysis":
            severity_analysis(df)
        elif analysis_option == "Arrest Analysis":
            arrest_analysis(df)
        elif analysis_option == "Domestic Analysis":
            domestic_analysis(df)
        elif analysis_option == "Location Analysis":
            location_analysis(df)
        elif analysis_option == "Crime by Location":
            crime_by_location(df)
        elif analysis_option == "Seasonal Trends":
            seasonal_trends(df)
        elif analysis_option == "Logistic Regression":
            st.subheader("Logistic Regression Model")
            X = df1[['Month', 'Day', 'Hour', 'District']] 
            y = df1['Warning']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = logistic_regression(X_train, X_test, y_train, y_test)
           #predict_future_crimes(model, X) 
        elif analysis_option == "Decision Tree":
            st.subheader("Decision Tree Model")
            X = df1[['Month', 'Day', 'Hour', 'District']]  
            y = df1['Warning']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = decision_tree(X_train, X_test, y_train, y_test)
            #predict_future_crimes(model, X)  
        elif analysis_option == "Support Vector Classifier":
            st.subheader("Support Vector Classifier Model")
            X = df1[['Month', 'Day', 'Hour', 'District']] 
            y = df1['Warning']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = support_vector_classifier(X_train, X_test, y_train, y_test)
            #predict_future_crimes(model, X)  

if __name__ == "__main__":
    main()
