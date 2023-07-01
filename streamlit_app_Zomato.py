
import streamlit as st
import joblib
import pandas as pd
import category_encoders
import sklearn
import xgboost

st.write ('This is Deployment for Zomato Data (Real Data From Kaggle)')
st.write ('<h1 style="text-align:center;color:purple;"> Zomato Deployment </h1>' , unsafe_allow_html=True)

Model = joblib.load("Model_Zomato.pkl")
Inputs = joblib.load("Input_Zomato.pkl")

def prediction (online_order,book_table,votes,location,approx_cost,listed_in_type, listed_in_city,cuisines_counts,rest_type_counts):
    test_df = pd.DataFrame(columns=Inputs)
    test_df.at[0 , "online_order"] = online_order
    test_df.at[0 , "book_table"] = book_table
    test_df.at[0 , "votes"] = votes
    test_df.at[0 , "location"] = location
    test_df.at[0 , "approx_cost(for two people)"] = approx_cost
    test_df.at[0 , "listed_in(type)"] = listed_in_type
    test_df.at[0 , "listed_in(city)"] = listed_in_city
    test_df.at[0 , "cuisines_counts"] = cuisines_counts
    test_df.at[0 , "rest_type_counts"] = rest_type_counts
    result = Model.predict(test_df)
    return result[0]

def main():
    online_order = st.selectbox("Online_Ava" , ['Yes', 'No'])
    
    book_table = st.selectbox("Booking_Ava" , ['Yes', 'No'])
    
    votes = st.slider("Votes" , min_value=0 , max_value =16832 ,value = 0 , step = 1  )
    
    location = st.selectbox("Location" , ['Banashankari', 'Basavanagudi', 'Mysore Road', 'Jayanagar',
       'Kumaraswamy Layout', 'Rajarajeshwari Nagar', 'Vijay Nagar',
       'Uttarahalli', 'JP Nagar', 'South Bangalore', 'City Market',
       'Bannerghatta Road', 'BTM', 'Kanakapura Road', 'Bommanahalli',
       'Electronic City', 'Wilson Garden', 'Shanti Nagar',
       'Koramangala 5th Block', 'Richmond Road', 'HSR',
       'Koramangala 7th Block', 'Bellandur', 'Sarjapur Road',
       'Marathahalli', 'Whitefield', 'East Bangalore', 'Old Airport Road',
       'Indiranagar', 'Koramangala 1st Block', 'Frazer Town', 'MG Road',
       'Brigade Road', 'Lavelle Road', 'Church Street', 'Ulsoor',
       'Residency Road', 'Shivajinagar', 'Infantry Road',
       'St. Marks Road', 'Cunningham Road', 'Race Course Road',
       'Commercial Street', 'Vasanth Nagar', 'Domlur',
       'Koramangala 8th Block', 'Ejipura', 'Jeevan Bhima Nagar',
       'Old Madras Road', 'Seshadripuram', 'Kammanahalli',
       'Koramangala 6th Block', 'Majestic', 'Langford Town',
       'Central Bangalore', 'Brookefield', 'ITPL Main Road, Whitefield',
       'Varthur Main Road, Whitefield', 'Koramangala 2nd Block',
       'Koramangala 3rd Block', 'Koramangala 4th Block', 'Koramangala',
       'Hosur Road', 'RT Nagar', 'Banaswadi', 'North Bangalore',
       'Nagawara', 'Hennur', 'Kalyan Nagar', 'HBR Layout',
       'Rammurthy Nagar', 'Thippasandra', 'CV Raman Nagar',
       'Kaggadasapura', 'Kengeri', 'Sankey Road', 'Malleshwaram',
       'Sanjay Nagar', 'Sadashiv Nagar', 'Basaveshwara Nagar',
       'Rajajinagar', 'Yeshwantpur', 'New BEL Road', 'West Bangalore',
       'Magadi Road', 'Yelahanka', 'Sahakara Nagar', 'Jalahalli',
       'Hebbal', 'Nagarbhavi', 'Peenya', 'KR Puram'])
    
    approx_cost = st.slider("approx_cost(for two people)" , min_value=40 , max_value=6000 , value=0 , step=1)
    
    listed_in_type = st.selectbox ("listed_in(type)" , ['Buffet', 'Cafes', 'Delivery', 'Desserts', 'Dine-out',
       'Drinks & nightlife', 'Pubs and bars'] )
    
    listed_in_city = st.selectbox ("listed_in(city)" , ['Banashankari', 'Bannerghatta Road', 'Basavanagudi', 'Bellandur',
       'Brigade Road', 'Brookefield', 'BTM', 'Church Street',
       'Electronic City', 'Frazer Town', 'HSR', 'Indiranagar',
       'Jayanagar', 'JP Nagar', 'Kalyan Nagar', 'Kammanahalli',
       'Koramangala 4th Block', 'Koramangala 5th Block',
       'Koramangala 6th Block', 'Koramangala 7th Block', 'Lavelle Road',
       'Malleshwaram', 'Marathahalli', 'MG Road', 'New BEL Road',
       'Old Airport Road', 'Rajajinagar', 'Residency Road',
       'Sarjapur Road', 'Whitefield'] )
    
    cuisines_counts = st.selectbox ("cuisines_numbers" , [1,2,3,4,5,6,7,8] )
    
    rest_type_counts = st.selectbox ("rest_type_counts" , [1,2] )
    
    if st.button("predict"):
        result = prediction(online_order, book_table, votes, location,approx_cost,listed_in_type,listed_in_city,cuisines_counts,rest_type_counts)
        label = ["Fail" , "Success"]
        st.text(f"The Resturant will {label[result]}")
        
if __name__ == '__main__':
    main()
    
