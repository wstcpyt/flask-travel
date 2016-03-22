from flask import Flask, render_template, redirect, request
from flask_restful import Resource, Api
from sklearn.externals import joblib
from settings import APP_STATIC
import os

# This is a init commit from yutong
app = Flask(__name__)

# restful api
api = Api(app)

class TaxiPredict(Resource):
    def get(self):
        return "hello world"
    def put(self):
        startgeocode_json = (request.form['startgeocode'])
        endgeocode_json = (request.form['endgeocode'])
        tripdistance_json = (request.form['tripdistance'])
        import json
        startgeocode = json.loads(startgeocode_json)
        endgeocode = json.loads(endgeocode_json)
        startlat = float(startgeocode['lat'])
        startlng = float(startgeocode['lng'])
        endlat = float(endgeocode['lat'])
        endlng = float(endgeocode['lng'])
        tripdist = float(tripdistance_json.split(" ")[0])
        hour = int(request.form['hour'])
        dayofweek = int(request.form['dayofweek'])
        lowspeedclf = joblib.load(os.path.join(APP_STATIC, 'costtime.pkl'))
        lowspeedx = [startlat,startlng,endlat,endlng,hour,dayofweek,tripdist]
        lowspeedy = int(lowspeedclf.predict(lowspeedx)[0])
        tripdurationclf = joblib.load(os.path.join(APP_STATIC, 'trip_duration.pkl'))
        tripduration_x = lowspeedx
        trip_duration_y = tripdurationclf.predict(tripduration_x)[0]

        duration_list = []
        lowspeed_list = []
        for i in range(24):
            x = [startlat,startlng,endlat,endlng,i,dayofweek,tripdist]
            duration_list.append([i,int(tripdurationclf.predict(x)[0])])
            lowspeed_list.append([i,int(lowspeedclf.predict(x)[0])])

        return {"lowspeedtime":lowspeedy, "tripduration":trip_duration_y, "duration_list":duration_list, "lowspeed_list":lowspeed_list}

api.add_resource(TaxiPredict,'/taxipredict')

#CORS ENABLE
@app.after_request
def after_request(response):
  response.headers.add('Access-Control-Allow-Origin', '*')
  response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
  response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
  return response

@app.route('/')
def main():
    return redirect('/index')


@app.route('/index')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run()
