import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the trained model and the OneHotEncoder
try:
    with open('xgboost_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('onehot_encoder.pkl', 'rb') as f:
        OHE = pickle.load(f)
except FileNotFoundError:
    st.error("Model or OneHotEncoder file not found. Make sure 'xgboost_model.pkl' and 'onehot_encoder.pkl' are in the same directory.")
    st.stop()


st.title("Crop Production and Yield Prediction")

st.write("Enter the details below to predict crop production and yield.")

# Hardcoded unique values for categorical features
state_options = ['Andaman and Nicobar Islands', 'Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chandigarh', 'Chhattisgarh', 'Dadra and Nagar Haveli', 'Daman and Diu', 'Delhi', 'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jammu and Kashmir', 'Jharkhand', 'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Manipur', 'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 'Puducherry', 'Punjab', 'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Telangana', 'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal']
district_options = ['NICOBARS', 'NORTH AND MIDDLE ANDAMAN', 'SOUTH ANDAMANS', 'ANANTAPUR', 'CHITTOOR', 'EAST GODAVARI', 'GUNTUR', 'KADAPA', 'KRISHNA', 'KURNOOL', 'PRAKASAM', 'SPSR NELLORE', 'SRIKAKULAM', 'VISAKHAPATNAM', 'VIZIANAGARAM', 'WEST GODAVARI', 'ANJAW', 'CHANGLANG', 'DIBANG VALLEY', 'EAST KAMENG', 'EAST SIANG', 'KURUNG KUMEY', 'LOHIT', 'LONGDING', 'LOWER DIBANG VALLEY', 'LOWER SUBANSIRI', 'NAMSAI', 'PAPUM PARE', 'TAWANG', 'TIRAP', 'UPPER SIANG', 'UPPER SUBANSIRI', 'WEST KAMENG', 'WEST SIANG', 'BAKSA', 'BARPETA', 'BONGAIGAON', 'CACHAR', 'CHIRANG', 'DARRANG', 'DHEMAJI', 'DHUBRI', 'DIBRUGARH', 'GOALPARA', 'GOLAGHAT', 'HAILAKANDI', 'JORHAT', 'KAMRUP', 'KAMRUP METRO', 'KARBI ANGLONG', 'KARIMGANJ', 'KOKRAJHAR', 'LAKHIMPUR', 'MARIGAON', 'NAGAON', 'NALBARI', 'SIVASAGAR', 'SONITPUR', 'TINSUKIA', 'UDALGURI', 'ARARIA', 'ARWAL', 'AURANGABAD', 'BANKA', 'BEGUSARAI', 'BHAGALPUR', 'BHOJPUR', 'BUXAR', 'DARBHANGA', 'GAYA', 'GOPALGANJ', 'JAMUI', 'JEHANABAD', 'KAIMUR (BHABUA)', 'KATIHAR', 'KHAGARIA', 'KISHANGANJ', 'LAKHISARAI', 'MADHEPURA', 'MADHUBANI', 'MUNGER', 'MUZAFFARPUR', 'NALANDA', 'NAWADA', 'PASCHIM CHAMPARAN', 'PATNA', 'PURBI CHAMPARAN', 'PURNIA', 'ROHTAS', 'SAHARSA', 'SAMASTIPUR', 'SARAN', 'SHEIKHPURA', 'SHEOHAR', 'SITAMARHI', 'SIWAN', 'SUPAUL', 'VAISHALI', 'CHANDIGARH', 'BALOD', 'BALODA BAZAR', 'BALRAMPUR', 'BASTAR', 'BEMETARA', 'BIJAPUR', 'BILASPUR', 'DAKSHIN BASTAR DANTEWADA', 'DHAMTARI', 'DURG', 'GARIYABAND', 'JANJGIR-CHAMPA', 'JASHPUR', 'KABIRDHAM', 'KANKER', 'KONDAGAON', 'KORBA', 'KORIYA', 'MAHASAMUND', 'MUNGELI', 'NARAYANPUR', 'RAIGARH', 'RAIPUR', 'RAJNANDGAON', 'SUKMA', 'SURGUJA', 'SURAJPUR', 'UTTAR BASTAR KANKER', 'DADRA AND NAGAR HAVELI', 'DAMAN', 'DIU', 'CENTRAL', 'EAST', 'NEW DELHI', 'NORTH', 'NORTH EAST', 'NORTH WEST', 'SOUTH', 'SOUTH WEST', 'WEST', 'NORTH GOA', 'SOUTH GOA', 'AHMADABAD', 'AMRELI', 'ANAND', 'BANAS KANTHA', 'BHARUCH', 'BHAVNAGAR', 'BOTAD', 'CHHOTAUDEPUR', 'DANG', 'DEVBHUMI DWARKA', 'GANDHINAGAR', 'GIR SOMNATH', 'JAMNAGAR', 'JUNAGADH', 'KACHCHH', 'KHEDA', 'MAHISAGAR', 'MEHSANA', 'MORBI', 'NARMADA', 'NAVSARI', 'PANCHMAHAL', 'PATAN', 'PORBANDAR', 'RAJKOT', 'SABAR KANTHA', 'SURAT', 'SURENDRANAGAR', 'TAPI', 'VADODARA', 'VALSAD', 'AMBALA', 'BHIWANI', 'CHARKHI DADRI', 'FARIDABAD', 'FATEHABAD', 'GURUGRAM', 'HISAR', 'JHAJJAR', 'JIND', 'KAITHAL', 'KARNAL', 'KURUKSHETRA', 'MAHENDRAGARH', 'MEWAT', 'PALWAL', 'PANCHKULA', 'PANIPAT', 'REWARI', 'ROHTAK', 'SIRSA', 'SONIPAT', 'YAMUNANAGAR', 'BILASPUR', 'CHAMBA', 'HAMIRPUR', 'KANGRA', 'KINNAUR', 'KULLU', 'LAHAUL AND SPITI', 'MANDI', 'SHIMLA', 'SIRMAUR', 'SOLAN', 'UNA', 'ANANTNAG', 'BARAMULLA', 'BUDGAM', 'DODA', 'GANDERBAL', 'JAMMU', 'KARGIL', 'KATHUA', 'KISHTWAR', 'KULGAM', 'KUPWARA', 'LEH LADAKH', 'POONCH', 'PULWAMA', 'RAJAURI', 'RAMBAN', 'REASI', 'SAMBA', 'SHOPIAN', 'SRINAGAR', 'UDHAMPUR', 'BOKARO', 'CHATRA', 'DEOGHAR', 'DHANBAD', 'DUMKA', 'EAST SINGHBHUM', 'GARHWA', 'GIRIDIH', 'GODDA', 'GUMLA', 'HAZARIBAGH', 'JAMTARA', 'KHUNTI', 'KODERMA', 'LATEHAR', 'LOHARDAGA', 'PAKUR', 'PALAMU', 'RAMGARH', 'RANCHI', 'SAHIBGANJ', 'SARAIKELA KHARSWAN', 'SIMDEGA', 'WEST SINGHBHUM', 'BAGALKOT', 'BENGALURU RURAL', 'BENGALURU URBAN', 'BELAGAVI', 'BELLARY', 'BIDAR', 'CHAMARAJANAGAR', 'CHIKKABALLAPURA', 'CHIKKAMAGALURU', 'CHITRADURGA', 'DAKSHINA KANNADA', 'DAVANAGERE', 'DHARWAD', 'GADAG', 'HASSAN', 'HAVERI', 'KALABURAGI', 'KODAGU', 'KOLAR', 'KOPPAL', 'MANDYA', 'MYSURU', 'RAICHUR', 'RAMANAGARA', 'SHIVAMOGGA', 'TUMAKURU', 'UDUPI', 'UTTARA KANNADA', 'VIJAYAPURA', 'YADGIR', 'ALAPPUZHA', 'ERNAKULAM', 'IDUKKI', 'KANNUR', 'KASARAGOD', 'KOLLAM', 'KOTTAYAM', 'KOZHIKODE', 'MALAPPURAM', 'PALAKKAD', 'PATHANAMTHITTA', 'THIRUVANANTHAPURAM', 'THRISSUR', 'WAYANAD', 'AGAR MALWA', 'ALIRAJPUR', 'ANUPPUR', 'ASHOKNAGAR', 'BALAGHAT', 'BARWANI', 'BETUL', 'BHIND', 'BHOPAL', 'BURHANPUR', 'CHHATARPUR', 'CHHINDWARA', 'DAMOH', 'DATIA', 'DEWAS', 'DHAR', 'DINDORI', 'GUNA', 'GWALIOR', 'HARDA', 'HOSHANGABAD', 'INDORE', 'JABALPUR', 'JHABUA', 'KATNI', 'KHANDWA', 'KHARGONE', 'MANDLA', 'MANDSAUR', 'MORENA', 'NARSINGHPUR', 'NEEMUCH', 'PANNA', 'RAISEN', 'RAJGARH', 'RATLAM', 'REWA', 'SAGAR', 'SATNA', 'SEHORE', 'SEONI', 'SHAHDOL', 'SHAJAPUR', 'SHEOPUR', 'SHIVPURI', 'SIDHI', 'SINGRAULI', 'TIKAMGARH', 'UJJAIN', 'UMARIA', 'VIDISHA', 'AHMEDNAGAR', 'AKOLA', 'AMRAVATI', 'AURANGABAD', 'BEED', 'BHANDARA', 'BULDHANA', 'CHANDRAPUR', 'DHULE', 'GADCHIROLI', 'GONDIYA', 'HINGOLI', 'JALGAON', 'JALNA', 'KOLHAPUR', 'LATUR', 'MUMBAI', 'NAGPUR', 'NANDED', 'NANDURBAR', 'NASIK', 'OSMANABAD', 'PALGHAR', 'PARBHANI', 'PUNE', 'RAIGAD', 'RATNAGIRI', 'SANGLI', 'SATARA', 'SINDHUDURG', 'SOLAPUR', 'THANE', 'WARDHA', 'WASHIM', 'YAVATMAL', 'BISHNUPUR', 'CHANDEL', 'CHURACHANDPUR', 'IMPHAL EAST', 'IMPHAL WEST', 'KANGPOKPI', 'KAKCHING', 'KAMJONG', 'NONEY', 'PHERZAWL', 'SENAPATI', 'TAMENGLONG', 'TENGNOUPAL', 'THOUBAL', 'UKHRUL', 'EAST GARO HILLS', 'EAST JAINTIA HILLS', 'EAST KHASI HILLS', 'NORTH GARO HILLS', 'RI BHOI', 'SOUTH GARO HILLS', 'SOUTH WEST GARO HILLS', 'SOUTH WEST KHASI HILLS', 'WEST JAINTIA HILLS', 'WEST KHASI HILLS', 'AIZAWL', 'CHAMPHAI', 'KOLASIB', 'LAWNGTLAI', 'LUNGLEI', 'MAMIT', 'SAIHA', 'SERCHHIP', 'DIMAPUR', 'KIPHIRE', 'KOHIMA', 'LONGLENG', 'MOKOKCHUNG', 'MON', 'PEREN', 'PHEK', 'TUENSANG', 'WOKHA', 'ZUNHEBOTO', 'ANUGUL', 'BALANGIR', 'BALASORE', 'BARGARH', 'BHADRAK', 'BOUDH', 'CUTTACK', 'DEOGARH', 'DHENKANAL', 'GAJAPATI', 'GANJAM', 'JAGATSINGHPUR', 'JAJPUR', 'JHARSUGUDA', 'KALAHANDI', 'KANDHAMAL', 'KENDRAPARA', 'KENDUJHAR', 'KHORDHA', 'KORAPUT', 'MALKANGIRI', 'MAYURBHANJ', 'NABARANGPUR', 'NAYAGARH', 'NUAPADA', 'PURI', 'RAYAGADA', 'SAMBALPUR', 'SONEPUR', 'SUNDARGARH', 'KARAIKAL', 'MAHE', 'PUDUCHERRY', 'YANAM', 'AMRITSAR', 'BARNALA', 'BATHINDA', 'FARIDKOT', 'FATEHGARH SAHIB', 'FAZILKA', 'FIROZEPUR', 'GURDASPUR', 'HOSHIARPUR', 'JALANDHAR', 'KAPURTHALA', 'LUDHIANA', 'MANSA', 'MOGA', 'MUKTSAR', 'NAWANSHAHR (Shahid Bhagat Singh Nagar)', 'PATHANKOT', 'PATIALA', 'RUPNAGAR', 'SANGRUR', 'SAS NAGAR (Mohali)', 'TARAN TARAN', 'AJMER', 'ALWAR', 'BANSWARA', 'BARAN', 'BARMER', 'BHARATPUR', 'BHILWARA', 'BIKANER', 'BUNDI', 'CHITTAURGARH', 'CHURU', 'DAUSA', 'DHOLPUR', 'DUNGARPUR', 'GANGANAGAR', 'HANUMANGARH', 'JAIPUR', 'JAISALMER', 'JALORE', 'JHALAWAR', 'JHUNJHUNU', 'JODHPUR', 'KARAULI', 'KOTA', 'NAGAUR', 'PALI', 'PRATAPGARH', 'RAJSAMAND', 'SAWAI MADHOPUR', 'SIKAR', 'SIROHI', 'TONK', 'UDAIPUR', 'EAST SIKKIM', 'NORTH SIKKIM', 'SOUTH SIKKIM', 'WEST SIKKIM', 'ARIYALUR', 'CHENGALPATTU', 'CHENNAI', 'COIMBATORE', 'CUDDALORE', 'DHARMAPURI', 'DINDIGUL', 'ERODE', 'KALLAKURICHI', 'KANCHIPURAM', 'KANYAKUMARI', 'KARUR', 'KRISHNAGIRI', 'MADURAI', 'MAYILADUTHURAI', 'NAGAPATTINAM', 'NAMAKKAL', 'PERAMBALUR', 'PUDUKKOTTAI', 'RAMANATHAPURAM', 'SALEM', 'SIVAGANGA', 'TENKASI', 'THANJAVUR', 'THENI', 'THOOTHUKUDI', 'TIRUCHIRAPPALLI', 'TIRUNELVELI', 'TIRUPATHUR', 'TIRUPPUR', 'TIRUVALLUR', 'TIRUVANNAMALAI', 'TIRUVARUR', 'VELLORE', 'VILLUPURAM', 'VIRUDHUNAGAR', 'ADILABAD', 'BHADRADRI KOTHAGUDEM', 'JAGITIAL', 'JANGAON', 'JAYASHANKAR BHUPALPALLY', 'Jogulamba Gadwal', 'KAMAREDDY', 'KARIMNAGAR', 'KHAMMAM', 'KOMARAM BHEEM ASIFABAD', 'MAHABUBABAD', 'MAHBUBNAGAR', 'MANCHERIAL', 'MEDAK', 'MEDCHAL MALKAJGIRI', 'MULUGU', 'NAGARKURNOOL', 'NALGONDA', 'NARAYANPET', 'NIRMAL', 'NIZAMABAD', 'PEDDAPALLI', 'RAJANNA SIRCILLA', 'RANGAREDDY', 'SANGAREDDY', 'SIDDIPET', 'SURYAPET', 'VIKARABAD', 'WANAPARTHY', 'WARANGAL RURAL', 'WARANGAL URBAN', 'YADADRI BHUVANAGIRI', 'DHALAI', 'GOMATI', 'KHOWAI', 'NORTH TRIPURA', 'SEPAHIJALA', 'SOUTH TRIPURA', 'UNAKOTI', 'WEST TRIPURA', 'AGRA', 'ALIGARH', 'AMBEDKAR NAGAR', 'AMETHI', 'AMROHA', 'AURAIYA', 'AYODHYA', 'AZAMGARH', 'BAGHPAT', 'BAHRAICH', 'BALIA', 'BALRAMPUR', 'BANDA', 'BARABANKI', 'BAREILLY', 'BASTI', 'Bhadohi', 'BIJNOR', 'BADAUN', 'BULANDSHAHR', 'CHANDAULI', 'CHITRAKOOT', 'DEORIA', 'ETAH', 'ETAWAH', 'AYODHAYA', 'FARRUKHABAD', 'FATEHPUR', 'FIROZABAD', 'GAUTAM BUDDHA NAGAR', 'GHAZIABAD', 'GHAZIPUR', 'GONDA', 'GORAKHPUR', 'HAMIRPUR', 'HAPUR', 'HARDOI', 'HATHRAS', 'JALAUN', 'JAUNPUR', 'JHANSI', 'KANNAUJ', 'KANPUR DEHAT', 'KANPUR NAGAR', 'KASGANJ', 'KAUSHAMBI', 'KHERI', 'KUSHI NAGAR', 'LAKHIMPUR KHERI', 'LALITPUR', 'LUCKNOW', 'MAHARAJGANJ', 'MAHOBA', 'MAINPURI', 'MATHURA', 'MAU', 'MEERUT', 'MIRZAPUR', 'MORADABAD', 'MUZAFFARNAGAR', 'PILIBHIT', 'PRATAPGARH', 'PRAYAGRAJ', 'RAE BARELI', 'RAMPUR', 'SAHARANPUR', 'SAMBHAL', 'SANT KABIR NAGAR', 'SHAHJAHANPUR', 'SHAMLI', 'SHRAVASTI', 'SIDDHARTH NAGAR', 'SITAPUR', 'SONBHADRA', 'SULTANPUR', 'UNNAO', 'VARANASI', 'ALMORA', 'BAGESHWAR', 'CHAMOLI', 'CHAMPAWAT', 'DEHRADUN', 'GARHWAL', 'HARDWAR', 'NAINITAL', 'PAURI GARHWAL', 'PITHORAGARH', 'RUDRA PRAYAG', 'TEHRI GARHWAL', 'UDAM SINGH NAGAR', 'UTTAR KASHI', '24 PARAGANAS NORTH', '24 PARAGANAS SOUTH', 'ALIPURDUAR', 'BANKURA', 'BIRBHUM', 'BURDWAN (BARDHAMAN)', 'COOCHBEHAR', 'DARJEELING', 'DINAPUR DAKSHIN', 'DINAPUR UTTAR', 'HOOGHLY', 'HOWRAH', 'JALPAIGURI', 'JHAGRAM', 'MALDAH', 'MURSHIDABAD', 'NADIA', 'PASCHIM MEDINIPUR', 'PASCHIM BARDHAMAN', 'PURBA MEDINIPUR', 'PURBA BARDHAMAN', 'PURULIA', 'SOUTH 24 PARGANAS', 'UTTAR DINAJPUR']
crop_options = ['Arecanut', 'Banana', 'Black pepper', 'Cashewnut', 'Coconut', 'Dry chillies', 'Ginger', 'Rice', 'Sugarcane', 'Sweet potato', 'Arhar/Tur', 'Bajra', 'Castor seed', 'Coriander', 'Cotton(lint)', 'Gram', 'Groundnut', 'Horse-gram', 'Jowar', 'Linseed', 'Maize', 'Mesta', 'Moong(Green Gram)', 'Niger seed', 'Onion', 'Potato', 'Ragi', 'Rapeseed &Mustard', 'Safflower', 'Sesamum', 'Small millets', 'Soyabean', 'Sunflower', 'Tapioca', 'Tobacco', 'Turmeric', 'Urad', 'Wheat', 'Jute', 'Masoor', 'Peas & beans (Pulses)', 'Barley', 'Garlic', 'Khesari', 'Sannhamp', 'Guar seed', 'Moth', 'Cardamom', 'Cowpea(Lobia)', 'Dry Ginger']
season_options = ['Kharif', 'Whole Year', 'Rabi', 'Autumn', 'Summer', 'Winter']


# Input fields for features
state = st.selectbox("State", state_options)
district = st.selectbox("District", district_options)
crop = st.selectbox("Crop", crop_options)
season = st.selectbox("Season", season_options)
area = st.number_input("Area", min_value=0.0, format="%f")

if st.button("Predict"):
    # Create a DataFrame with the input values
    input_data = pd.DataFrame({
        'State': [state],
        'District': [district],
        'Crop': [crop],
        'Season': [season],
        'Area': [area]
    })

    # Separate categorical and numerical columns
    categorical_cols = input_data.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = input_data.select_dtypes(exclude=['object']).columns.tolist()

    # Preprocess the input data using the loaded OneHotEncoder
    try:
        input_categorical_encoded = OHE.transform(input_data[categorical_cols])
        input_categorical_encoded_df = pd.DataFrame(input_categorical_encoded, index=input_data.index)
    except ValueError as e:
        st.error(f"Error during one-hot encoding: {e}. This might be due to unseen categories in the input data.")
        st.stop()


    # Combine processed categorical and numerical data
    input_processed = pd.concat([input_data[numerical_cols], input_categorical_encoded_df], axis=1)

    # Ensure columns match the training data columns (important for consistent prediction)
    # This step is crucial if your training data had a different order of one-hot encoded columns
    # or if some categories were not present in the training data but appear in the input.
    # A robust way to handle this is to save the list of columns from X_train_final and reindex here.
    # For this example, I'm assuming the order is consistent, but be aware of this in production.

    # Make prediction
    predicted_production = model.predict(input_processed)[0]

    # Calculate predicted yield
    predicted_yield = predicted_production / area if area > 0 else 0

    st.subheader("Prediction Results:")
    st.write(f"Predicted Production: {predicted_production:.2f}")
    st.write(f"Predicted Yield: {predicted_yield:.2f}")
