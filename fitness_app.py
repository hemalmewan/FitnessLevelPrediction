import numpy as np
from flask import Flask,request,jsonify,render_template
import pickle

##load pickle file
with open('best_model.pkl','rb') as f:
    best_model=pickle.load(f)

##careate Flask App
app=Flask(__name__)

##import html file
@app.route("/")
def Home():
    return render_template("home.html")

@app.route("/second_page")
def second_page():
    return render_template("web_page.html")


##make prediction and get the data
@app.route("/predict",methods=['POST'])
def predict():
   if request.method=='POST':
       age=int(request.form['age'])
       gender=request.form['gender']
       activity=request.form['activity'].lower()
       duration=float(request.form['duration'])
       intensity=request.form['intensity']
       calories=float(request.form['calories'])
       avg_heart_rate=float(request.form['avg_heart_rate'])
       hours_sleep=float(request.form['hours_sleep'])
       strees_level=int(request.form['strees_level'])
       daily_steps=int(request.form['daily_steps'])
       water_intake=float(request.form['water_intake'])
       heart_rate=float(request.form['heart_rate'])
       smoking=request.form['smoking']
       blood_pressure=float(request.form['blood_pressure'])
       bmi=float(request.form['bmi'])
         
       ##endocde categorical variables
       ##############Gender###################
       gender_male = 1 if gender == 'Male' else 0
       gender_female=1 if gender=='Female' else 0
       gender_other=1 if gender=='Other' else 0
       ################Intensity#########################
       intensity_encoded = {'low': 0, 'medium': 1, 'high': 2}.get(intensity.lower(), 1)
       ####################Smoking Status##########################
       smoking_never= 1 if smoking == 'Never' else 0
       smoking_former=1 if smoking=='Former' else 0
       smoking_current=1 if smoking=='Current' else 0
       ########################Activity Type#########################
       activity_list = ['running', 'walking', 'cycling', 'swimming', 'yoga',
                 'weightlifting', 'hiking', 'dancing', 'basketball', 'football']
       
       activity_encode=[1 if activity==act else 0 for act in activity_list]

       feature=[age,gender_male,gender_female,gender_other]+activity_encode+[duration,intensity_encoded, 
                calories,avg_heart_rate,hours_sleep,strees_level, 
                daily_steps,water_intake,heart_rate,smoking_never,smoking_former,
                smoking_current,blood_pressure,bmi]
       
      
       prediction = best_model.predict(np.array([feature]))[0] ##make the prediction based on the given inputs
       fitness_score=round(prediction,2) ##round the prediction result in 2 decimal places

       ##based on the result show message to user 
       if fitness_score>=16:
           color_msg="success" ##green
           message="You're in excellent shape! Keep it up!"
           level="advanced"
       elif 10<=fitness_score<16:
           color_msg="warning" ##orange/yellow
           message = "You're doing okay. Consider more activity and rest."
           level="intermediate"
       else:
           color_msg="danger" ##red
           message = "Health alert! Time to improve your fitness habits."
           level="beginner"
        
        ##get the recommended videos and save them as a dictionary
       recommended_videos = {
            'beginner': [
            {
                'title': '10 Minute Beginner Workout',
                'url': 'https://www.youtube.com/watch?v=UBMk30rjy0o'
            },
            {
                'title': 'Full Body Stretch Routine',
                'url': 'https://www.youtube.com/watch?v=sTANio_2E0Q'
            }
        ],
            'intermediate': [
            {
                'title': '30-Minute HIIT Workout',
                'url': 'https://www.youtube.com/watch?v=ml6cT4AZdqI'
            },
            {
                'title': 'Core & Cardio Challenge',
                'url': 'https://www.youtube.com/watch?v=1f8yoFFdkcY'
            }
        ],
            'advanced': [
            {
                'title': '45-Minute Advanced Training',
                'url': 'https://www.youtube.com/watch?v=qWy_aOlB45Y'
            },
            {
                'title': 'High Intensity Circuit',
                'url': 'https://www.youtube.com/watch?v=UoC_O3HzsH0'
            }
        ]
    }
       videos_to_show = recommended_videos.get(level, [])
   
           
       return render_template("web_page.html",prediction_test=fitness_score,feedback=message,
                              color_code=color_msg,videos=videos_to_show)
   
if __name__=="__main__":
    app.run(host='0.0.0.0', port=5000,debug=True)


