import streamlit as st
import pandas as pd
import pickle
import numpy as np
import joblib

st.title("Crop Production and Yield Prediction")
st.write("Enter the details below to predict crop production and yield.")

# ===============================
# ✅ Load model, encoder, columns
# ===============================
try:
    model = joblib.load("xgboost_model.pkl")
    OHE = joblib.load("onehot_encoder.pkl")
    feature_cols = joblib.load("feature_columns.pkl")

except FileNotFoundError as e:
    st.error(f"Missing file: {e}")
    st.stop()

# ===============================
# ✅ Input options
# ===============================
state_options=['Andaman and Nicobar Islands', 'Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chandigarh', 'Chhattisgarh', 'Dadra and Nagar Haveli', 'Daman and Diu', 'Delhi', 'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jammu and Kashmir', 'Jharkhand', 'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Manipur', 'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 'Puducherry', 'Punjab', 'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Telangana', 'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal']    # your full list here (same as before)
district_options=['24 PARAGANAS NORTH', '24 PARAGANAS SOUTH', 'ADILABAD', 'AGAR MALWA', 'AGRA', 'AHMADABAD', 'AHMEDNAGAR', 'AIZAWL', 'AJMER', 'AKOLA', 'ALAPPUZHA', 'ALIGARH', 'ALIPURDUAR', 'ALIRAJPUR', 'ALLAHABAD', 'ALMORA', 'ALWAR', 'AMBALA', 'AMBEDKAR NAGAR', 'AMETHI', 'AMRAVATI', 'AMRELI', 'AMRITSAR', 'AMROHA', 'ANAND', 'ANANTAPUR', 'ANANTNAG', 'ANJAW', 'ANUGUL', 'ANUPPUR', 'ARARIA', 'ARAVALLI', 'ARIYALUR', 'ARWAL', 'ASHOKNAGAR', 'AURAIYA', 'AURANGABAD', 'AZAMGARH', 'Andaman and Nicobar Islands', 'BADGAM', 'BAGALKOT', 'BAGALKOTE', 'BAGESHWAR', 'BAGHPAT', 'BAHRAICH', 'BAKSA', 'BALAGHAT', 'BALANGIR', 'BALESHWAR', 'BALLARI', 'BALLIA', 'BALOD', 'BALODA BAZAR', 'BALRAMPUR', 'BANAS KANTHA', 'BANDA', 'BANDIPORA', 'BANGALORE RURAL', 'BANKA', 'BANKURA', 'BANSWARA', 'BARABANKI', 'BARAMULLA', 'BARAN', 'BAREILLY', 'BARGARH', 'BARMER', 'BARNALA', 'BARPETA', 'BARWANI', 'BASTAR', 'BASTI', 'BATHINDA', 'BEED', 'BEGUSARAI', 'BELAGAVI', 'BELGAUM', 'BELLARY', 'BEMETARA', 'BENGALURU URBAN', 'BETUL', 'BHADRADRI', 'BHADRAK', 'BHAGALPUR', 'BHANDARA', 'BHARATPUR', 'BHARUCH', 'BHAVNAGAR', 'BHILWARA', 'BHIND', 'BHIWANI', 'BHOJPUR', 'BHOPAL', 'BIDAR', 'BIJAPUR', 'BIJNOR', 'BIKANER', 'BILASPUR', 'BIRBHUM', 'BISHNUPUR', 'BOKARO', 'BONGAIGAON', 'BOTAD', 'BOUDH', 'BUDAUN', 'BULANDSHAHR', 'BULDHANA', 'BUNDI', 'BURHANPUR', 'BUXAR', 'CACHAR', 'CHAMARAJANAGAR', 'CHAMARAJANAGARA', 'CHAMBA', 'CHAMOLI', 'CHAMPAWAT', 'CHAMPHAI', 'CHANDAULI', 'CHANDEL', 'CHANDIGARH', 'CHANDRAPUR', 'CHANGLANG', 'CHARKI DADRI', 'CHATRA', 'CHENGALPATTU', 'CHENNAI', 'CHHATARPUR', 'CHHINDWARA', 'CHHOTAUDEPUR', 'CHIKBALLAPUR', 'CHIKKABALLAPURA', 'CHIKKAMAGALURU', 'CHIKMAGALUR', 'CHIRANG', 'CHITRADURGA', 'CHITRAKOOT', 'CHITTOOR', 'CHITTORGARH', 'CHURACHANDPUR', 'CHURU', 'COIMBATORE', 'COOCHBEHAR', 'CUDDALORE', 'CUTTACK', 'DADRA AND NAGAR HAVELI', 'DAKSHIN KANNAD', 'DAKSHINA KANNADA', 'DAMAN', 'DAMOH', 'DANG', 'DANTEWADA', 'DARBHANGA', 'DARJEELING', 'DARRANG', 'DATIA', 'DAUSA', 'DAVANGERE', 'DEHRADUN', 'DELHI_TOTAL', 'DEOGARH', 'DEOGHAR', 'DEORIA', 'DEVBHUMI DWARKA', 'DEWAS', 'DHALAI', 'DHAMTARI', 'DHANBAD', 'DHAR', 'DHARMAPURI', 'DHARWAD', 'DHEMAJI', 'DHENKANAL', 'DHOLPUR', 'DHUBRI', 'DHULE', 'DIBANG VALLEY', 'DIBRUGARH', 'DIMA HASAO', 'DIMAPUR', 'DINAJPUR DAKSHIN', 'DINAJPUR UTTAR', 'DINDIGUL', 'DINDORI', 'DIU', 'DODA', 'DOHAD', 'DUMKA', 'DUNGARPUR', 'DURG', 'Daman and Diu', 'Delhi', 'EAST DISTRICT', 'EAST GARO HILLS', 'EAST GODAVARI', 'EAST JAINTIA HILLS', 'EAST KAMENG', 'EAST KHASI HILLS', 'EAST SIANG', 'EAST SINGHBUM', 'ERNAKULAM', 'ERODE', 'ETAH', 'ETAWAH', 'FAIZABAD', 'FARIDABAD', 'FARIDKOT', 'FARRUKHABAD', 'FATEHABAD', 'FATEHGARH SAHIB', 'FATEHPUR', 'FAZILKA', 'FIROZABAD', 'FIROZEPUR', 'GADAG', 'GADCHIROLI', 'GAJAPATI', 'GANDERBAL', 'GANDHINAGAR', 'GANGANAGAR', 'GANJAM', 'GARHWA', 'GARIYABAND', 'GAURELLA-PENDRA-MARWAHI', 'GAUTAM BUDDHA NAGAR', 'GAYA', 'GHAZIABAD', 'GHAZIPUR', 'GIR SOMNATH', 'GIRIDIH', 'GOALPARA', 'GODDA', 'GOLAGHAT', 'GOMATI', 'GONDA', 'GONDIA', 'GOPALGANJ', 'GORAKHPUR', 'GULBARGA', 'GUMLA', 'GUNA', 'GUNTUR', 'GURDASPUR', 'GURGAON', 'GWALIOR', 'Goa', 'HAILAKANDI', 'HAMIRPUR', 'HANUMAKONDA', 'HANUMANGARH', 'HAPUR', 'HARDA', 'HARDOI', 'HARIDWAR', 'HASSAN', 'HATHRAS', 'HAVERI', 'HAZARIBAGH', 'HINGOLI', 'HISAR', 'HOJAI', 'HOOGHLY', 'HOSHANGABAD', 'HOSHIARPUR', 'HOWRAH', 'HYDERABAD', 'IDUKKI', 'IMPHAL EAST', 'IMPHAL WEST', 'INDORE', 'JABALPUR', 'JAGATSINGHAPUR', 'JAGITIAL', 'JAIPUR', 'JAISALMER', 'JAJAPUR', 'JALANDHAR', 'JALAUN', 'JALGAON', 'JALNA', 'JALORE', 'JALPAIGURI', 'JAMMU', 'JAMNAGAR', 'JAMTARA', 'JAMUI', 'JANGOAN', 'JANJGIR-CHAMPA', 'JASHPUR', 'JAUNPUR', 'JAYASHANKAR', 'JEHANABAD', 'JHABUA', 'JHAJJAR', 'JHALAWAR', 'JHANSI', 'JHARGRAM', 'JHARSUGUDA', 'JHUNJHUNU', 'JIND', 'JODHPUR', 'JOGULAMBA', 'JORHAT', 'JUNAGADH', 'KABIRDHAM', 'KACHCHH', 'KADAPA', 'KAIMUR (BHABUA)', 'KAITHAL', 'KALABURAGI', 'KALAHANDI', 'KALIMPONG', 'KALLAKURICHI', 'KAMAREDDY', 'KAMLE', 'KAMRUP', 'KAMRUP METRO', 'KANCHIPURAM', 'KANDHAMAL', 'KANGRA', 'KANKER', 'KANNAUJ', 'KANNIYAKUMARI', 'KANNUR', 'KANPUR DEHAT', 'KANPUR NAGAR', 'KAPURTHALA', 'KARAIKAL', 'KARAULI', 'KARBI ANGLONG', 'KARGIL', 'KARIMGANJ', 'KARIMNAGAR', 'KARNAL', 'KARUR', 'KASARAGOD', 'KASGANJ', 'KATHUA', 'KATIHAR', 'KATNI', 'KAUSHAMBI', 'KENDRAPARA', 'KENDUJHAR', 'KHAGARIA', 'KHAMMAM', 'KHANDWA', 'KHARGONE', 'KHEDA', 'KHERI', 'KHORDHA', 'KHOWAI', 'KHUNTI', 'KINNAUR', 'KIPHIRE', 'KISHANGANJ', 'KISHTWAR', 'KODAGU', 'KODERMA', 'KOHIMA', 'KOKRAJHAR', 'KOLAR', 'KOLASIB', 'KOLHAPUR', 'KOLLAM', 'KOMARAM BHEEM ASIFABAD', 'KONDAGAON', 'KOPPAL', 'KORAPUT', 'KORBA', 'KOREA', 'KOTA', 'KOTTAYAM', 'KOZHIKODE', 'KRA DAADI', 'KRISHNA', 'KRISHNAGIRI', 'KULGAM', 'KULLU', 'KUPWARA', 'KURNOOL', 'KURUKSHETRA', 'KURUNG KUMEY', 'KUSHI NAGAR', 'LAHUL AND SPITI', 'LAKHIMPUR', 'LAKHISARAI', 'LALITPUR', 'LATEHAR', 'LATUR', 'LAWNGTLAI', 'LEH LADAKH', 'LEPARADA', 'LOHARDAGA', 'LOHIT', 'LONGDING', 'LONGLENG', 'LOWER DIBANG VALLEY', 'LOWER SIANG', 'LOWER SUBANSIRI', 'LUCKNOW', 'LUDHIANA', 'LUNGLEI', 'MADHEPURA', 'MADHUBANI', 'MADURAI', 'MAHABUBABAD', 'MAHARAJGANJ', 'MAHASAMUND', 'MAHBUBNAGAR', 'MAHE', 'MAHENDRAGARH', 'MAHESANA', 'MAHISAGAR', 'MAHOBA', 'MAINPURI', 'MAJULI', 'MALAPPURAM', 'MALDAH', 'MALKANGIRI', 'MAMIT', 'MANCHERIAL', 'MANDI', 'MANDLA', 'MANDSAUR', 'MANDYA', 'MANSA', 'MARIGAON', 'MATHURA', 'MAU', 'MAYURBHANJ', 'MEDAK', 'MEDCHAL', 'MEDCHAL MALKAJGIRI', 'MEDINIPUR EAST', 'MEDINIPUR WEST', 'MEERUT', 'MEWAT', 'MIRZAPUR', 'MOGA', 'MOKOKCHUNG', 'MON', 'MORADABAD', 'MORBI', 'MORENA', 'MUKTSAR', 'MULUGU', 'MUMBAI SUBURBAN', 'MUNGELI', 'MUNGER', 'MURSHIDABAD', 'MUZAFFARNAGAR', 'MUZAFFARPUR', 'MYSORE', 'MYSURU', 'NABARANGPUR', 'NADIA', 'NAGAON', 'NAGAPATTINAM', 'NAGARKURNOOL', 'NAGAUR', 'NAGPUR', 'NAINITAL', 'NALANDA', 'NALBARI', 'NALGONDA', 'NAMAKKAL', 'NAMSAI', 'NANDED', 'NANDURBAR', 'NARAYANAPET', 'NARAYANPUR', 'NARMADA', 'NARSINGHPUR', 'NASHIK', 'NAVSARI', 'NAWADA', 'NAWANSHAHR', 'NAYAGARH', 'NEEMUCH', 'NICOBARS', 'NIRMAL', 'NIWARI', 'NIZAMABAD', 'NORTH AND MIDDLE ANDAMAN', 'NORTH DISTRICT', 'NORTH GARO HILLS', 'NORTH GOA', 'NORTH TRIPURA', 'NUAPADA', 'OSMANABAD', 'PAKKE KESSANG', 'PAKUR', 'PALAKKAD', 'PALAMU', 'PALGHAR', 'PALI', 'PALWAL', 'PANCH MAHALS', 'PANCHKULA', 'PANIPAT', 'PANNA', 'PAPUM PARE', 'PARBHANI', 'PASCHIM BARDHAMAN', 'PASHCHIM CHAMPARAN', 'PATAN', 'PATHANAMTHITTA', 'PATHANKOT', 'PATIALA', 'PATNA', 'PAURI GARHWAL', 'PEDDAPALLI', 'PERAMBALUR', 'PEREN', 'PHEK', 'PILIBHIT', 'PITHORAGARH', 'PONDICHERRY', 'POONCH', 'PORBANDAR', 'PRAKASAM', 'PRATAPGARH', 'PUDUKKOTTAI', 'PULWAMA', 'PUNE', 'PURBA BARDHAMAN', 'PURBI CHAMPARAN', 'PURI', 'PURNIA', 'PURULIA', 'RAE BARELI', 'RAICHUR', 'RAIGAD', 'RAIGARH', 'RAIPUR', 'RAISEN', 'RAJANNA', 'RAJAURI', 'RAJGARH', 'RAJKOT', 'RAJNANDGAON', 'RAJSAMAND', 'RAMANAGARA', 'RAMANATHAPURAM', 'RAMBAN', 'RAMGARH', 'RAMPUR', 'RANCHI', 'RANGAREDDI', 'RANIPET', 'RATLAM', 'RATNAGIRI', 'RAYAGADA', 'REASI', 'REWA', 'REWARI', 'RI BHOI', 'ROHTAK', 'ROHTAS', 'RUDRA PRAYAG', 'RUPNAGAR', 'S', 'SABAR KANTHA', 'SAGAR', 'SAHARANPUR', 'SAHARSA', 'SAHEBGANJ', 'SAIHA', 'SALEM', 'SAMASTIPUR', 'SAMBA', 'SAMBALPUR', 'SAMBHAL', 'SANGAREDDY', 'SANGLI', 'SANGRUR', 'SANT KABEER NAGAR', 'SANT RAVIDAS NAGAR', 'SARAIKELA KHARSAWAN', 'SARAN', 'SATARA', 'SATNA', 'SAWAI MADHOPUR', 'SEHORE', 'SENAPATI', 'SEONI', 'SEPAHIJALA', 'SERCHHIP', 'SHAHDOL', 'SHAHID BHAGAT SINGH NAGAR', 'SHAHJAHANPUR', 'SHAJAPUR', 'SHAMLI', 'SHEIKHPURA', 'SHEOHAR', 'SHEOPUR', 'SHI YOMI', 'SHIMLA', 'SHIMOGA', 'SHIVAMOGGA', 'SHIVPURI', 'SHOPIAN', 'SHRAVASTI', 'SIANG', 'SIDDHARTH NAGAR', 'SIDDIPET', 'SIDHI', 'SIKAR', 'SIMDEGA', 'SINDHUDURG', 'SINGRAULI', 'SIRMAUR', 'SIROHI', 'SIRSA', 'SITAMARHI', 'SITAPUR', 'SIVAGANGA', 'SIVASAGAR', 'SIWAN', 'SOLAN', 'SOLAPUR', 'SONBHADRA', 'SONEPUR', 'SONIPAT', 'SONITPUR', 'SOUTH ANDAMANS', 'SOUTH DISTRICT', 'SOUTH GARO HILLS', 'SOUTH GOA', 'SOUTH SALMARA MANCACHAR', 'SOUTH TRIPURA', 'SOUTH WEST GARO HILLS', 'SOUTH WEST KHASI HILLS', 'SPSR NELLORE', 'SRIKAKULAM', 'SRINAGAR', 'SUKMA', 'SULTANPUR', 'SUNDARGARH', 'SUPAUL', 'SURAJPUR', 'SURAT', 'SURENDRANAGAR', 'SURGUJA', 'SURYAPET', 'TAMENGLONG', 'TAPI', 'TARN TARAN', 'TAWANG', 'TEHRI GARHWAL', 'TENKASI', 'THANE', 'THANJAVUR', 'THE NILGIRIS', 'THENI', 'THIRUVALLUR', 'THIRUVANANTHAPURAM', 'THIRUVARUR', 'THOOTHUKUDI', 'THOUBAL', 'THRISSUR', 'TIKAMGARH', 'TINSUKIA', 'TIRAP', 'TIRUCHIRAPPALLI', 'TIRUNELVELI', 'TIRUPATHUR', 'TIRUPPUR', 'TIRUVANNAMALAI', 'TONK', 'TUENSANG', 'TUMAKURU', 'TUMKUR', 'TUTICORIN', 'UDAIPUR', 'UDALGURI', 'UDAM SINGH NAGAR', 'UDHAMPUR', 'UDUPI', 'UJJAIN', 'UKHRUL', 'UMARIA', 'UNA', 'UNAKOTI', 'UNNAO', 'UPPER SIANG', 'UPPER SUBANSIRI', 'UTTAR KANNAD', 'UTTAR KASHI', 'UTTARA KANNADA', 'VADODARA', 'VAISHALI', 'VALSAD', 'VARANASI', 'VELLORE', 'VIDISHA', 'VIJAYAPURA', 'VIKARABAD', 'VILLUPURAM', 'VIRUDHUNAGAR', 'VISAKHAPATANAM', 'VIZIANAGARAM', 'WANAPARTHY', 'WARANGAL', 'WARANGAL URBAN', 'WARDHA', 'WASHIM', 'WAYANAD', 'WEST DISTRICT', 'WEST GARO HILLS', 'WEST GODAVARI', 'WEST JAINTIA HILLS', 'WEST KAMENG', 'WEST KARBI ANGLONG', 'WEST KHASI HILLS', 'WEST SIANG', 'WEST SINGHBHUM', 'WEST TRIPURA', 'WOKHA', 'YADADRI', 'YADAGIRI', 'YADGIR', 'YAMUNANAGAR', 'YANAM', 'YAVATMAL', 'ZUNHEBOTO']
crop_options = ['Arecanut', 'Arhar/Tur', 'Bajra', 'Banana', 'Barley', 'Black pepper', 'Cardamom', 'Cashewnut', 'Castor seed', 'Coconut', 'Coriander', 'Cotton(lint)', 'Cowpea(Lobia)', 'Dry Ginger', 'Dry chillies', 'Garlic', 'Ginger', 'Gram', 'Groundnut', 'Guar seed', 'Horse-gram', 'Jowar', 'Jute', 'Khesari', 'Linseed', 'Maize', 'Masoor', 'Mesta', 'Moong(Green Gram)', 'Moth', 'Niger seed', 'Onion', 'Peas & beans (Pulses)', 'Potato', 'Ragi', 'Rapeseed &Mustard', 'Rice', 'Safflower', 'Sannhamp', 'Sesamum', 'Small millets', 'Soyabean', 'Sugarcane', 'Sunflower', 'Sweet potato', 'Tapioca', 'Tobacco', 'Turmeric', 'Urad', 'Wheat']     # your crop list
season_options = ['Autumn', 'Kharif', 'Rabi', 'Summer', 'Whole Year', 'Winter']

# ===============================
# ✅ User Inputs
# ===============================
state = st.selectbox("State", state_options)
district = st.selectbox("District", district_options)
crop = st.selectbox("Crop", crop_options)
season = st.selectbox("Season", season_options)
area = st.number_input("Area (in hectares)", min_value=0.0, format="%f")

# ===============================
# ✅ Prediction Button
# ===============================
if st.button("Predict"):

    # Build DataFrame
    input_data = pd.DataFrame({
        'State': [state],
        'District': [district],
        'Crop': [crop],
        'Season': [season],
        'Area': [area]
    })

    # Separate types
    categorical_cols = ['State', 'District', 'Crop', 'Season']
    numerical_cols = ['Area']

    # ======================================================
    # ✅ Correct One-Hot Encoding WITH column names
    # ======================================================
    try:
        encoded = OHE.transform(input_data[categorical_cols])
        encoded_cols = OHE.get_feature_names_out(categorical_cols)
        encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=input_data.index)

    except Exception as e:
        st.error(f"OHE Error: {e}")
        st.stop()

    # ======================================================
    # ✅ Combine + realign EXACTLY as model expects
    # ======================================================
    input_processed = pd.concat([input_data[numerical_cols], encoded_df], axis=1)

    # Reindex to training columns
    input_processed = input_processed.reindex(columns=feature_cols, fill_value=0)

    # ======================================================
    # ✅ Predict
    # ======================================================
    predicted_production = model.predict(input_processed)[0]

    # Avoid negatives due to tree model edge cases
    predicted_production = max(0, predicted_production)

    predicted_yield = predicted_production / area if area > 0 else 0

    # ======================================================
    # ✅ Output
    # ======================================================
    st.subheader("Prediction Results")
    st.write(f"**Predicted Production:** {predicted_production:.2f} tons")
    st.write(f"**Predicted Yield:** {predicted_yield:.2f} tons/ha")



