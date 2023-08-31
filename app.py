from flask import Flask , request, render_template, jsonify
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application


@app.route('/')
def home_page():
    return render_template('index.html')


@app.route('/predict', methods = ['GET', 'POST'], endpoint = 'predict_datapoint')


def predict_datapoint():
    if request.method == 'GET':
        return render_template('form.html')
    
    else:
        data = CustomData(
            Username = str(request.form.get('Username')),
            Caption = str(request.form.get('Caption')),
            Followers = int(request.form.get('Followers')),
            Hashtags = str(request.form.get('Hashtags'))
        )

        final_new_data = data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(final_new_data)

        results = round(pred[0],2)

        return render_template('results.html', final_result = results)



if __name__ == "__main__":
    
    app.run(host = '0.0.0.0', debug = True)


