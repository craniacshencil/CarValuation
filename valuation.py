import streamlit as st
import pandas as pd
import time
import pickle
df = pd.read_csv("C:/Users/rajni/OneDrive/Desktop/CarBazaar/train3.csv")
df2 = pd.read_csv("C:/Users/rajni/OneDrive/Desktop/CarBazaar/data_entry_train.csv")

def main():
        #Loading the model
        st.set_page_config(initial_sidebar_state="collapsed",layout="centered")
        pickle_in = open('predictor.pkl', 'rb')
        cat_model = pickle.load(pickle_in)
        st.columns(3)[1].image("header.png",use_column_width="auto")
        st.subheader('About your car: ')
        st.divider()
        brand = st.selectbox("Enter the brand of your car: "
                        , df['oem'].unique())
        #st.divider()
        yr = st.selectbox("Enter registration year of car: "
                        , sorted(df.loc[df.oem == brand]['myear'].unique()
                                , reverse = True))
        #st.divider()
        model = st.selectbox("Enter the model of your car: "
                        , df.loc[(df.oem == brand)
                                & (df.myear == yr)]['model'].unique())
        #st.divider()
        variant = st.selectbox("Enter the variant: "
                        , df.loc[(df.model == model)
                                & (df.myear == yr)]['variant'].unique())
        #st.divider()
        fueltype = st.selectbox("Enter fuel: ",
                                df2.loc[(df2.model == model)
                                & (df2.myear == yr)
                                & (df2.variant == variant)]['fuel'].unique())
        
        #st.divider()
        transmission = st.selectbox('Enter transmission: ',
                                df2.loc[(df2.myear == yr) & 
                                        (df2.model == model) &
                                        (df2.variant == variant) &
                                        (df2.fuel == fueltype)]['transmission'].unique())
        if transmission == 'manual':
                transmission = 0
        else:
                transmission = 1
        
        if fueltype == 'lpg':
                fueltype = 0
        if fueltype == 'cng':
                fueltype = 1
        if fueltype == 'petrol':
                fueltype = 2
        if fueltype == 'diesel':
                fueltype = 3
        if fueltype == 'electric':
                fueltype = 4
        #st.divider()
        owner = st.selectbox("Enter owner-number: ", 
                        df['owner_type'].unique())
        st.info("here 1 = first owner")
        #st.divider()
        st.session_state['kms'] = 'not set'
        kms = st.number_input("Enter kms driven: ", min_value = 0
                        , max_value = 130000, step = 5000)
        if kms != 0:
                st.session_state['kms'] = 'set'
        st.divider()

        confirmation = st.columns(7)[3].button("   Confirm details   ")
        if(confirmation):
                if st.session_state['kms'] == 'not set':
                        st.error("You have not entered kms driven.")
                if(st.session_state['kms'] == 'set'):  
                        progress_text = "Saving Details..."
                        my_bar = st.progress(0, text = progress_text)

                        for percent_complete in range(100):
                                time.sleep(0.01)
                                my_bar.progress(percent_complete + 1, text=progress_text)
                        
                        values = [brand, yr, model, variant, fueltype
                                , transmission, owner, kms]
                        
                        st.success("Details have been registered.")
                        #for i in range(len(values)):
                                #st.write(values[i])
                        pred = predict_price(values, cat_model)    
                        st.write(f"The predicted price is: {round(pred, 2)}")



def predict_price(values, cat_model):
      temp = df.loc[(df.oem == values[0]) 
             & (df.myear == values[1]) 
             & (df.model == values[2]) 
             #& (df.variant == values[3])
             & (df.fuel == values[4])
              ]
      tc = temp['Turbo Charger'].mode()[0]
      kw = temp['Kerb Weight'].mean()
      dt = temp['Drive Type'].mode()[0]
      seats = temp['Seats'].mode()[0]
      tspeed = temp['Top Speed'].mean()
      acc = temp['Acceleration'].mean()
      doors = temp['Doors'].mode()[0]
      cvolume = temp['Cargo Volume'].mean()
      maxTorque = temp['Max Torque Delivered'].mean()
      measure = temp['avg_measure'].mean()
      feat = temp['Features'].mode()[0]
      valves = temp['Valves'].mode()[0]
      tread = temp['Tread'].mean()
      
      #predictors = [myear, transmission, fuel, km, Turbo Charger, kerb weight, drive type
      # ,seats, top speed, Acceleration, Doors, Cargo Volume, owner_type,
      # Max Torque Delivered, avg_measure, Features, valves, tread]
      predictors = [values[1], values[5], values[4], values[7], 
                    tc, kw, dt, seats, tspeed, acc, doors, cvolume,
                    values[6], maxTorque, measure, feat, valves, tread]
      pred = cat_model.predict(predictors)
      return pred

if __name__ == "__main__":
        main()