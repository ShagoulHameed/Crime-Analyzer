# Chicago Crime Analysis
![image](https://github.com/ShagoulHameed/Crime-Analyzer/assets/154894802/c140d76b-c1cc-4a5d-8632-cc8faddd2961)

**Objective:**
We are observing that there has been a significant increase in crime in Chicago in recent days. Therefore, we are transferring you to Chicago as a Senior Investigation Officer under special deputation.

Your primary objective in this role is to leverage historical and recent crime data to identify patterns, trends, and hotspots within Chicago. By conducting a thorough analysis of this data, you will support strategic decision-making, improve resource allocation, and contribute to reducing crime rates and enhancing public safety. Your task is to provide actionable insights that can shape our crime prevention strategies, ensuring a safer and more secure community. This project will be instrumental in aiding law enforcement operations and enhancing the overall effectiveness of our efforts in combating crime in Chicago.

This Streamlit application analyzes crime data in Chicago, providing insights through various analyses and machine learning models. The app includes temporal, geospatial, and classification analyses to help understand crime patterns and predict future crime incidents.

## Features

- **Temporal Analysis**: Visualize the number of crimes per year.
- **Peak Crime Hours**: Analyze the number of crimes per hour.
- **Geospatial Analysis**: Display crime hotspots on a map.
- **District Crime Rates**: Show crime rates by district.
- **Crime Type Distribution**: Show the distribution of different crime types.
- **Severity Analysis**: Analyze the severity of crimes.
- **Arrest Analysis**: Display arrest rates by crime type.
- **Domestic Analysis**: Compare domestic vs. non-domestic crimes.
- **Location Analysis**: Display the top 10 crime locations.
- **Crime by Location**: Show crime types by location.
- **Seasonal Trends**: Analyze seasonal trends in crime types.
- **Logistic Regression**: Perform logistic regression to predict crime severity.
- **Decision Tree**: Use decision tree to classify crime severity.
- **Support Vector Classifier**: Apply SVC for crime severity classification.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/chicago-crime-analysis.git
    ```
2. Navigate to the project directory:
    ```bash
    cd chicago-crime-analysis
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```
2. Upload a CSV file containing the crime data through the sidebar.
3. Select an analysis from the sidebar to visualize the data and perform machine learning tasks.

## Data Preprocessing

### Loading Data
The `load_data` function loads the data from a CSV file and converts the 'Date' column to datetime format.

### Cleaning Data
The `clean_data` function fills missing values in specified columns with their mode values.

### Preprocessing Data
The `preprocess_data` function extracts additional columns such as 'Month', 'Day', 'Hour', and groups the data to create a new DataFrame `df1`.

## Machine Learning Models

### Logistic Regression
The `logistic_regression` function trains a logistic regression model and displays accuracy, confusion matrix, and classification report.

### Decision Tree
The `decision_tree` function trains a decision tree model, displays the tree, and shows accuracy, confusion matrix, and classification report.

### Support Vector Classifier
The `support_vector_classifier` function trains an SVC model and displays accuracy, confusion matrix, and classification report.

## Analysis Functions

- **temporal_analysis**: Shows the number of crimes per year.
      _Crime Trends Over Time_: Examine how the number of crimes has changed over the years. This could include plotting the number of crimes per year, month, or even day to identify trends or seasonal variations.
  ![image](https://github.com/ShagoulHameed/Crime-Analyzer/assets/154894802/723afe06-b843-40cc-8c7d-79e84ca89824)
      _Peak Crime Hours_: Determine the times of day when crimes are most frequently reported by analyzing the 'Date' and 'Time' fields. 
![image](https://github.com/ShagoulHameed/Crime-Analyzer/assets/154894802/a6811cf5-5f97-410f-801b-78a9e63f43f4)
_Per day_  ![image](https://github.com/ShagoulHameed/Crime-Analyzer/assets/154894802/67f4583a-97d0-43ab-bf3c-2e97903d3ee9)
 
- **geospatial_analysis**: Displays crime hotspots on a map.
     _Crime Hotspots_: Use the latitude and longitude coordinates to identify areas with high concentrations of crimes. Tools like heatmaps or kernel density estimation can be useful.
  
    _District/Ward Analysis_: Compare crime rates across different districts and wards to identify which areas are more prone to specific types of crimes.
    District: ![image](https://github.com/ShagoulHameed/Crime-Analyzer/assets/154894802/0656aa30-017c-4f99-950b-6712c75fd4bf)
    Ward: ![image](https://github.com/ShagoulHameed/Crime-Analyzer/assets/154894802/546c14b7-2d18-4d7c-a664-b65936665d7b)

  - **Crime Type Analysis**: Displays the distribution of crime types.
       _Distribution of Crime Types_: Analyze the frequency of different 'Primary Type' and 'Description' fields to understand the most common types of crimes.
    ![image](https://github.com/ShagoulHameed/Crime-Analyzer/assets/154894802/82e080e4-4177-4f32-a551-e0e423a4c722)
       _Severity Analysis_: Investigate the distribution of severe crimes (e.g., HOMICIDE, ASSAULT, OFFENSE INVOLVING CHILDREN, ROBBER, CRIM SEXUAL ASSAULT , OFFENSE INVOLVING CHILDREN,SEX OFFENSE,WEAPONS VIOLATION ) versus less severe crimes (e.g., thefts, fraud).
![image](https://github.com/ShagoulHameed/Crime-Analyzer/assets/154894802/1c7c43c2-7ab4-4f73-b0af-ae95c589869d)

- **arrest_analysis**: Shows arrest rates by crime type.
      _Arrest Rates_: Calculate the percentage of crimes that result in an arrest. This can be broken down by crime type, location, and time period.
    ![image](https://github.com/ShagoulHameed/Crime-Analyzer/assets/154894802/dc58859a-2ef6-49b3-bd26-e894da2ae8d2)
      _Domestic vs. Non-Domestic Crimes_: Compare the characteristics and frequencies of domestic-related incidents versus non-domestic incidents.
![image](https://github.com/ShagoulHameed/Crime-Analyzer/assets/154894802/a32b69e6-52b5-4dfb-be14-2b67cd9632e5)
- **location_analysis**: Displays the top 10 crime locations.
     _Location Description Analysis_: Investigate the most common locations for crimes (e.g., streets, parking lots, apartments) and see how crime types vary by location.
  ![image](https://github.com/ShagoulHameed/Crime-Analyzer/assets/154894802/0366570b-0038-4715-b4a5-b00755198079)
  ![image](https://github.com/ShagoulHameed/Crime-Analyzer/assets/154894802/4ef96379-871f-4034-986d-250799da98fe)
     _Comparison by Beat and Community Area_: Analyze crime data by beat and community area to identify localized crime patterns and hotspots.
![image](https://github.com/ShagoulHameed/Crime-Analyzer/assets/154894802/1b52cbb5-4823-4f17-aba7-e3be92883150)
- **Seasonal and Weather Impact**: Analyzes seasonal trends in crime types.
    _Seasonal Trends_: Examine whether certain types of crimes are more prevalent in specific seasons (e.g., summer vs. winter).
  ![image](https://github.com/ShagoulHameed/Crime-Analyzer/assets/154894802/a0297a6e-ef85-4fce-968f-73d878b1fe61)
- **Repeat Offenders and Recidivism**
    _Repeat Crime Locations_: Identify locations that are repeatedly associated with criminal activity.
        _Recidivism Rates_: If data on repeat offenders is available, analyze recidivism rates and factors contributing to repeat offenses.
![image](https://github.com/ShagoulHameed/Crime-Analyzer/assets/154894802/7b3f86f9-fd2f-4781-a058-37ef94eb6572)

**Predictive Modeling and Risk Assessment**
   -_ Predictive Analysis_: Develop models to predict future crime incidents based on historical data, time, location, and other relevant factors.
    ** Decision Tree Classifier**
    ![image](https://github.com/ShagoulHameed/Crime-Analyzer/assets/154894802/2e8456de-0755-475e-b122-0d1880c48b02)
    ![image](https://github.com/ShagoulHameed/Crime-Analyzer/assets/154894802/e7540c9b-3e88-469b-9283-c21d288fc8a7)
    ![image](https://github.com/ShagoulHameed/Crime-Analyzer/assets/154894802/664e4c1a-4bf0-4482-8884-fb2e413c46a2)
    **Support Vector Classifier**
      ![image](https://github.com/ShagoulHameed/Crime-Analyzer/assets/154894802/ff1c9b75-2067-4655-9a59-75b1ec26ac38)
      ![image](https://github.com/ShagoulHameed/Crime-Analyzer/assets/154894802/9307ddc4-720b-4a82-8df0-d8473e2ea7bb)
     **Logistic Regression**
     ![image](https://github.com/ShagoulHameed/Crime-Analyzer/assets/154894802/ab9d9365-b29e-4519-99f9-8e9db3b753ac)
     ![image](https://github.com/ShagoulHameed/Crime-Analyzer/assets/154894802/e735fb3a-546f-4e6e-81be-3a7186a38ac1)

## Contact

For any inquiries, please contact [Shagoul Hameed](mailto:Shagoul04@gmail.com).
