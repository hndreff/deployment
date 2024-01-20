import streamlit as st
import numpy as np

# import ml package
import joblib
import os

attribute_info = """
                 - - Your Store:    american, mexican, other, indian, italian, sandwich,
                                    thai, cafe, salad, pizza, chinese, singaporean,
                                    burger, breakfast, mediterranean, japanese, catering,
                                    filipino, convenience-store, greek, korean, vegan,
                                    asian, barbecue, fast, dessert, smoothie, seafood,
                                    vietnamese, cajun, steak, soup, vegetarian, persian,
                                    sushi, latin-american, hawaiian, chocolate, burmese,
                                    british, nepalese, pasta, alcohol, dim-sum, peruvian,
                                    turkish, malaysian, ethiopian, middle-eastern, afghan,
                                    bubble-tea, german, french, caribbean, gluten-free,
                                    comfort-food, gastropub, pakistani, moroccan, spanish,
                                    southern, tapas, russian, brazilian, european, cheese,
                                    african, argentine, kosher, irish, lebanese, belgian,
                                    indonesian, alcohol-plus-food.
                 - Your Order's protocol:   Protocol one, Protocol two, Protocol three, 
                                            Protocol four, Protocol five, Protocol six, Protocol seven.
                 - Your total's items:  one item, two items, three items, four items, five items, 
                                        six items, seven items, eight items, nine items, ten items.
                 - Subtotal: 0 - 10000
                 - Number of distinct items: one, two, three, four, five, six, seven, eight, nine
                 - Minimum item price: 0 - 3000
                 - Maximum item price: 0 - 3500
                 - Total onshift partners: 1 - 171
                 - Total busy partners: 1 - 154
                 - Total outstanding orders: 1 - 262
                 """

sto = {'american':1, 'mexican':2, 'other':3, 'indian':4, 'italian':5, 'sandwich':6,
        'thai':7, 'cafe':8, 'salad':9, 'pizza':10, 'chinese':11, 'singaporean':12,
        'burger':13, 'breakfast':14, 'mediterranean':15, 'japanese':16, 'catering':17,
        'filipino':18, 'convenience-store':19, 'greek':20, 'korean':21, 'vegan':22,
        'asian':23, 'barbecue':24, 'fast':25, 'dessert':26, 'smoothie':27, 'seafood':28,
        'vietnamese':29, 'cajun':30, 'steak':31, 'soup':32, 'vegetarian':33, 'persian':34,
        'sushi':35, 'latin-american':36, 'hawaiian':37, 'chocolate':38, 'burmese':39,
        'british':40, 'nepalese':41, 'pasta':42, 'alcohol':43, 'dim-sum':44, 'peruvian':45,
        'turkish':46, 'malaysian':47, 'ethiopian':48, 'middle-eastern':49, 'afghan':50,
        'bubble-tea':51, 'german':52, 'french':53, 'caribbean':54, 'gluten-free':55,
        'comfort-food':56, 'gastropub':57, 'pakistani':58, 'moroccan':59, 'spanish':60,
        'southern':61, 'tapas':62, 'russian':63, 'brazilian':64, 'european':65, 'cheese':66,
        'african':67, 'argentine':68, 'kosher':69, 'irish':70, 'lebanese':71, 'belgian':72,
        'indonesian':73, 'alcohol-plus-food':74}
ord = {'Protocol one':1, 'Protocol two':2, 'Protocol three':3, 'Protocol four':4, 
       'Protocol five':5, 'Protocol six':6, 'Protocol seven':7}
total = {'one item':1, 'two items':2, 'three items':3, 'four items':4, 'five items':5,
          'six items':6, 'seven items':7, 'eight items':8, 'nine items':9, 'ten items':10}
dis = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8, 'nine':9}

def get_value(val, my_dict):
    for key, value in my_dict.items():
        if val == key:
            return value
        
def load_model(model_file):
    loaded_model = joblib.load(open(os.path.join(model_file), 'rb'))
    return loaded_model

def run_ml_app():
    st.subheader("ML Section")
    with st.expander("Attribute Info"):
        st.markdown(attribute_info)

    st.subheader("Input Your Data")
    store_primary_encoded = st.selectbox('Your store', ['american', 'mexican', 'other', 'indian', 'italian', 'sandwich',
                                            'thai', 'cafe', 'salad', 'pizza', 'chinese', 'singaporean',
                                            'burger', 'breakfast', 'mediterranean', 'japanese', 'catering',
                                            'filipino', 'convenience-store', 'greek', 'korean', 'vegan',
                                            'asian', 'barbecue', 'fast', 'dessert', 'smoothie', 'seafood',
                                            'vietnamese', 'cajun', 'steak', 'soup', 'vegetarian', 'persian',
                                            'sushi', 'latin-american', 'hawaiian', 'chocolate', 'burmese',
                                            'british', 'nepalese', 'pasta', 'alcohol', 'dim-sum', 'peruvian',
                                            'turkish', 'malaysian', 'ethiopian', 'middle-eastern', 'afghan',
                                            'bubble-tea', 'german', 'french', 'caribbean', 'gluten-free',
                                            'comfort-food', 'gastropub', 'pakistani', 'moroccan', 'spanish',
                                            'southern', 'tapas', 'russian', 'brazilian', 'european', 'cheese',
                                            'african', 'argentine', 'kosher', 'irish', 'lebanese', 'belgian',
                                            'indonesian', 'alcohol-plus-food'])
    order_protocol = st.selectbox("Your order's protocol", ['Protocol one', 'Protocol two', 'Protocol three', 
                                                            'Protocol four', 'Protocol five', 'Protocol six', 'Protocol seven'])
    total_items = st.selectbox("Your total items", ['one item', 'two items', 'three items', 'four items', 
                                                    'five items', 'six items', 'seven items', 'eight items',
                                                      'nine items', 'ten items'])
    subtotal = st.number_input("Subtotal",0,10000)
    num_distinct_items = st.selectbox("Number of distinct items", ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'])
    min_item_price = st.number_input("Minimum item price",0,3000)
    max_item_price = st.number_input("Maximum item price",0,3500)
    total_onshift_partners = st.number_input("Total onshift partners",1,171)
    total_busy_partners = st.number_input("Total busy partners",1,154)
    total_outstanding_orders = st.number_input("Total outstanding orders",1,262)
    

    with st.expander("Your Selected Options"):
        result = {
            'store_primary_encoded':store_primary_encoded,
            'order_protocol':order_protocol,
            'total_items':total_items,
            'subtotal':subtotal,
            'num_distinct_items':num_distinct_items,
            'min_item_price':min_item_price,
            'max_item_price':max_item_price,
            'total_onshift_partners':total_onshift_partners,
            'total_busy_partners':total_busy_partners,
            'total_outstanding_orders':total_outstanding_orders,
        }
    
    # st.write(result)

    encoded_result = []
    for i in result.values():
        if type(i) == int:
            encoded_result.append(i)
        elif i in ['american', 'mexican', 'other', 'indian', 'italian', 'sandwich',
                    'thai', 'cafe', 'salad', 'pizza', 'chinese', 'singaporean',
                    'burger', 'breakfast', 'mediterranean', 'japanese', 'catering',
                    'filipino', 'convenience-store', 'greek', 'korean', 'vegan',
                    'asian', 'barbecue', 'fast', 'dessert', 'smoothie', 'seafood',
                    'vietnamese', 'cajun', 'steak', 'soup', 'vegetarian', 'persian',
                    'sushi', 'latin-american', 'hawaiian', 'chocolate', 'burmese',
                    'british', 'nepalese', 'pasta', 'alcohol', 'dim-sum', 'peruvian',
                    'turkish', 'malaysian', 'ethiopian', 'middle-eastern', 'afghan',
                    'bubble-tea', 'german', 'french', 'caribbean', 'gluten-free',
                    'comfort-food', 'gastropub', 'pakistani', 'moroccan', 'spanish',
                    'southern', 'tapas', 'russian', 'brazilian', 'european', 'cheese',
                    'african', 'argentine', 'kosher', 'irish', 'lebanese', 'belgian',
                    'indonesian', 'alcohol-plus-food']:
            res = get_value(i, sto)
            encoded_result.append(res)
        elif i in ['Protocol one', 'Protocol two', 'Protocol three', 
                    'Protocol four', 'Protocol five', 'Protocol six', 'Protocol seven']:
            res = get_value(i, ord)
            encoded_result.append(res)
        elif i in ['one item', 'two items', 'three items', 'four items', 'five items', 
                   'six items', 'seven items', 'eight items', 'nine items', 'ten items']:
            res = get_value(i, total)
            encoded_result.append(res)
        elif i in ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']:
            res = get_value(i, dis)
            encoded_result.append(res)
    
    # st.write(encoded_result)
    # prediction section
    st.subheader('Prediction Result')
    single_array = np.array(encoded_result).reshape(1, -1)
    # st.write(single_array)

    model = load_model("linear_regression.pkl")
    scaler = joblib.load("robust_scaler.pkl")

    new_data_scaled = scaler.transform(single_array)
    predicted_target = model.predict(new_data_scaled)
    # Assuming 'predicted_target' is the predicted value in minutes
    predicted_minutes = predicted_target[0]  # Assuming predicted_target is a NumPy array

    # Convert minutes to hours and minutes
    hours = predicted_minutes // 60  # Get the whole number of hours
    minutes = predicted_minutes % 60  # Get the remaining minutes

    # Display the result
    st.success(f'Predicted Time: {int(hours)} hours {int(minutes)} minutes')
