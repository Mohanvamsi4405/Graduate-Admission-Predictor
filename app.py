# import pickle
# import logging
# from flask import Flask, render_template, request
# from flask_cors import cross_origin
# from sklearn.preprocessing import StandardScaler
# import sklearn.preprocessing 
# app = Flask(__name__)  # initializing a flask app

# @app.route('/', methods=['GET'])  # route to display the home page
# @cross_origin()
# def homePage():
#     return render_template("index.html")


# @app.route('/about.html')
# def about():
#     return render_template("about.html")

# @app.route('/contact.html')
# def contact():
#     return render_template("contact.html")

# @app.route('/HowItWorks.html')
# def how_it_works():
#     return render_template("howitworks.html")

# @app.route('/technology.html')
# def technology():
#     return render_template("technology.html")

# @app.route('/predictor.html', methods=['POST', 'GET'])  # route to show the predictions in a web UI
# @cross_origin()
# def predict():
#     if request.method == 'POST':
#         try:
#             # Reading the inputs given by the user
#             gre_score = float(request.form['gre_score'])
#             toefl_score = float(request.form['toefl_score'])
#             university_rating = float(request.form['university_rating'])
#             sop = float(request.form['sop'])
#             lor = float(request.form['lor'])
#             cgpa = float(request.form['cgpa'])
#             is_research = request.form['research']
#             research = 1 if is_research == 'yes' else 0

#             # Load the model and scaler
#             loaded_model = pickle.load(open("Admission_Model.pickle", 'rb'))
#             # scaler = pickle.load(open("StandardScaler.pickle", 'rb'))  # Assuming you have a scaler file

#             # Transform the input data
#             input_data = [[gre_score, toefl_score, university_rating, sop, lor, cgpa, research]]
#             scaled_data = scaler.transform(input_data)

#             # Predict using the loaded model
#             prediction = loaded_model.predict(scaled_data)
#             prediction_percentage = round(prediction[0] * 100)

#             # Show the prediction results in a UI
#             return render_template('results.html', prediction=prediction_percentage)
        
#         except Exception as e:
#             logging.error(f"Error: {e}")
#             return render_template('results.html', prediction="Error in processing your request.")
    
#     else:
#         return render_template('predictor.html')

# if __name__ == "__main__":
#     app.run(debug=True)  # running the app
#     app.run(debug=True)  # running the app
    
    
    
    
    
    
# from flask import Flask, render_template, request
# import numpy as np
# import pickle
# from sklearn.preprocessing import StandardScaler

# app = Flask(__name__)

# # Load the trained model
# try:
#     model = pickle.load(open("Admission_Prediction_elastic_model.pickle", "rb"))
# except Exception as e:
#     model = None
#     print(f"Error loading model: {e}")

# # Home Page
# @app.route('/')
# def home():
#     return render_template('index.html')

# # About Page
# @app.route('/about.html')
# def about():
#     return render_template('about.html')

# # Contact Page
# @app.route('/contact.html')
# def contact():
#     return render_template('contact.html')

# # How It Works Page
# @app.route('/HowItWorks.html')
# def how_it_works():
#     return render_template('HowItWorks.html')

# # Technology Page
# @app.route('/technology.html')
# def technology():
#     return render_template('technology.html')

# # Predictor Page (GET request loads the form)
# @app.route('/predictor.html', methods=['GET'])
# def predictor():
#     return render_template('predictor.html')

# # Prediction Route (Handles POST request from form)
# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#             # Reading the inputs given by the user
#             gre_score = float(request.form['gre_score'])
#             toefl_score = float(request.form['toefl_score'])
#             university_rating = float(request.form['university_rating'])
#             sop = float(request.form['sop'])
#             lor = float(request.form['lor'])
#             cgpa = float(request.form['cgpa'])
#             is_research = request.form['research']
#             research = 1 if is_research == 'yes' else 0

#             # Load the model
#             loaded_model = pickle.load(open("Admission_Prediction_elastic_model.pickle", 'rb'))
#             scaler = pickle.load(open("Admission_Prediction_scalar.pickle", 'rb'))

#             # Predict using the loaded model
            
#             input_data = [[gre_score, toefl_score, university_rating, sop, lor, cgpa, research]]
#             input_data=scaler.transform(input_data)
#             print(input_data)
#             prediction = loaded_model.predict(input_data)
#             prediction_percentage = round(prediction[0]*100)
#             print(prediction_percentage)
#             if(prediction_percentage>100 or prediction_percentage<0):
#                 prediction_percentage=0
            
#             # Show the prediction results in a UI
#             return render_template('results.html', prediction=prediction_percentage)
        
#     except Exception as e:
#         return render_template('results.html', prediction=f"Error: {str(e)}")

# if __name__ == '__main__':
#     app.run(debug=True)


# from flask import Flask, render_template, request
# import numpy as np
# import pickle
# import pandas as pd
# from sklearn.preprocessing import StandardScaler

# app = Flask(__name__)

# # Load the trained model and scaler
# try:
#     model = pickle.load(open("Admission_Prediction_elastic_model.pickle", "rb"))
#     scaler = pickle.load(open("Admission_Prediction_scalar.pickle", "rb"))
# except Exception as e:
#     model = None
#     scaler = None
#     print(f"Error loading model or scaler: {e}")

# # Define feature names (must match the training data)
# feature_names = [
#     'GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR', 'CGPA', 'Research'
# ]

# # University Recommendation Mapping
# university_recommendations = {
#     5: {
#         "min_chance": 90,
#         "universities": [
#             "Stanford University", "Harvard University", "MIT", "Caltech", "Princeton University", 
#             "Yale University", "University of Chicago", "Columbia University", 
#             "University of Pennsylvania", "Johns Hopkins University"
#         ]
#     },
#     4: {
#         "min_chance": 80,
#         "universities": [
#             "University of California, Berkeley", "University of California, Los Angeles (UCLA)", 
#             "University of Michigan", "University of Virginia", 
#             "University of North Carolina, Chapel Hill", "University of Southern California (USC)", 
#             "New York University (NYU)", "Carnegie Mellon University", 
#             "University of Illinois, Urbana-Champaign", "University of Wisconsin-Madison"
#         ]
#     },
#     3: {
#         "min_chance": 70,
#         "universities": [
#             "University of Toronto", "University of British Columbia", "McGill University", 
#             "University of Melbourne", "University of Sydney", "University of Manchester", 
#             "University of Edinburgh", "University of Amsterdam", "University of Copenhagen"
#         ]
#     },
#     2: {
#         "min_chance": 60,
#         "universities": [
#             "University of Arizona", "University of Colorado, Boulder", "University of Utah", 
#             "University of Kansas", "University of Nebraska-Lincoln", "University of Oklahoma", 
#             "University of South Carolina", "University of Tennessee", "University of Kentucky", 
#             "University of Arkansas"
#         ]
#     },
#     1: {
#         "min_chance": 50,
#         "universities": [
#             "University of Idaho", "University of Wyoming", "University of Mississippi", 
#             "University of New Mexico", "University of Maine", "University of North Dakota", 
#             "University of South Dakota", "University of Montana", "University of Alaska, Fairbanks", 
#             "University of Hawaii, Manoa"
#         ]
#     }
# }

# # Common University for all intervals
# common_university = "University of Global Excellence"

# # Home Page
# @app.route('/')
# def home():
#     return render_template('index.html')

# # About Page
# @app.route('/about.html')
# def about():
#     return render_template('about.html')

# # Contact Page
# @app.route('/contact.html')
# def contact():
#     return render_template('contact.html')

# # How It Works Page
# @app.route('/HowItWorks.html')
# def how_it_works():
#     return render_template('HowItWorks.html')

# # Technology Page
# @app.route('/technology.html')
# def technology():
#     return render_template('technology.html')

# # Predictor Page (GET request loads the form)
# @app.route('/predictor.html', methods=['GET'])
# def predictor():
#     return render_template('predictor.html')

# # Prediction Route (Handles POST request from form)
# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Reading the inputs given by the user
#         gre_score = float(request.form['gre_score'])
#         toefl_score = float(request.form['toefl_score'])
#         university_rating = float(request.form['university_rating'])
#         sop = float(request.form['sop'])
#         lor = float(request.form['lor'])
#         cgpa = float(request.form['cgpa'])
#         is_research = request.form['research']
#         research = 1 if is_research == 'yes' else 0

#         # Prepare input data as a DataFrame with feature names
#         input_data = pd.DataFrame([[
#             gre_score, toefl_score, university_rating, sop, lor, cgpa, research
#         ]], columns=feature_names)

#         # Scale the input data
#         scaled_data = scaler.transform(input_data)

#         # Predict using the loaded model
#         prediction = model.predict(scaled_data)
#         prediction_percentage = round(prediction[0] * 100)

#         # Ensure prediction is within 0-100%
#         if prediction_percentage > 100:
#             prediction_percentage = 100
#         elif prediction_percentage < 0:
#             prediction_percentage = 0

#         # Generate recommendations
#         recommendations = []

#         # If predicted chance is >= 95%, recommend top 5 universities
#         if prediction_percentage >= 95:
#             recommendations = university_recommendations[5]["universities"][:5]
#         else:
#             # Generate 5% intervals from 50% to 95%
#             intervals = []
#             for lower_bound in range(50, 95, 5):  # 50-55, 55-60, ..., 90-95
#                 upper_bound = lower_bound + 5
#                 if upper_bound > 95:
#                     upper_bound = 95
#                 intervals.append((lower_bound, upper_bound))

#             # Recommend universities for each interval
#             for lower_bound, upper_bound in intervals:
#                 universities_in_interval = []
#                 for rating, data in university_recommendations.items():
#                     if data["min_chance"] >= lower_bound and data["min_chance"] < upper_bound:
#                         universities_in_interval.extend(data["universities"])

#                 # Select 4 unique universities and 1 common university (from previous interval or higher)
#                 unique_universities = list(set(universities_in_interval))[:4]  # Ensure uniqueness
#                 recommendations.extend(unique_universities + [common_university])

#         # Show the prediction results and recommendations in a UI
#         return render_template(
#             'results.html', 
#             prediction=prediction_percentage, 
#             recommendations=recommendations
#         )

#     except Exception as e:
#         return render_template('results.html', prediction=f"Error: {str(e)}", recommendations=[])

# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, render_template, request
import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the trained model and scaler
try:
    model = pickle.load(open("Admission_Prediction_elastic_model.pickle", "rb"))
    scaler = pickle.load(open("Admission_Prediction_scalar.pickle", "rb"))
except Exception as e:
    model = None
    scaler = None
    print(f"Error loading model or scaler: {e}")

# Define feature names (must match the training data)
feature_names = [
    'GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR', 'CGPA', 'Research'
]

# University Recommendation Mapping
university_recommendations = {
    95: ['Stanford University'],
    90: ['Harvard University', 'MIT', 'Caltech', 'Princeton University', 'University of Oxford', 'University of Cambridge', 'ETH Zurich', 'National University of Singapore', 'Tsinghua University'],
    85: ['Yale University', 'Peking University'],
    80: ['UC Berkeley', 'University of Chicago', 'Columbia University', 'Technical University of Munich', 'University of Toronto', 'Nanyang Technological University', 'University of Tokyo'],
    75: ['USC', 'Imperial College London', 'LSE', 'UCL', 'University of Amsterdam', 'University of British Columbia', 'Kyoto University'],
    70: ['University of Manchester', 'Delft University of Technology', 'McGill University', 'KAIST'],
    65: ['University of Warwick', 'University of Copenhagen', 'University of Ottawa', 'Seoul National University'],
    60: ['University of Bristol', 'University of Birmingham', 'Heidelberg University', 'KU Leuven', 'University of Alberta', 'University of Waterloo'],
    55: ['University of Glasgow', 'Sorbonne University', 'Simon Fraser University'],
    50: ['University of Helsinki', 'Western University', 'King Abdulaziz University'],
    45: ['Politecnico di Milano', "Queen's University", 'University of Montreal', 'King Fahd University of Petroleum and Minerals']
}

# Common University for all intervals
common_university = "University of Global Excellence"

@app.route('/')
def homePage():
    return render_template("index.html")

@app.route('/about.html')
def about():
    return render_template("about.html")

@app.route('/contact.html')
def contact():
    return render_template("contact.html")

@app.route('/HowItWorks.html')
def how_it_works():
    return render_template("HowItWorks.html")

@app.route('/technology.html')
def technology():
    return render_template("technology.html")

@app.route('/University.html')
def university():
    return render_template("University.html")

@app.route('/predictor.html')
def predictor():
    return render_template("predictor.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None or scaler is None:
            return render_template('results.html', prediction="Error: Model or Scaler is not loaded.", recommendations=[])

        # Reading the inputs given by the user
        gre_score = float(request.form['gre_score'])
        toefl_score = float(request.form['toefl_score'])
        university_rating = float(request.form['university_rating'])
        sop = float(request.form['sop'])
        lor = float(request.form['lor'])
        cgpa = float(request.form['cgpa'])
        is_research = request.form['research']
        research = 1 if is_research == 'yes' else 0

        # Prepare input data as a DataFrame with feature names
        input_data = pd.DataFrame([[
            gre_score, toefl_score, university_rating, sop, lor, cgpa, research
        ]], columns=feature_names)

        # Scale the input data
        scaled_data = scaler.transform(input_data)

        # Predict using the loaded model
        prediction = model.predict(scaled_data)
        prediction_percentage = round(prediction[0] * 100)

        # Ensure prediction is within 0-100%
        prediction_percentage = max(0, min(prediction_percentage, 100))

        # Generate recommendations
        recommendations = []
        for min_percentage, universities in university_recommendations.items():
            if prediction_percentage >= min_percentage:
                recommendations = universities
                break

        # Show the prediction results and recommendations in a UI
        return render_template(
            'results.html', 
            prediction=prediction_percentage, 
            recommendations=recommendations
        )

    except Exception as e:
        return render_template('results.html', prediction=f"Error: {str(e)}", recommendations=[])

if __name__ == '__main__':
    app.run(debug=True)
