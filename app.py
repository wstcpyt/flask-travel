from flask import Flask, render_template, redirect, request
from flask_restful import Resource, Api
from sklearn.externals import joblib
from settings import APP_STATIC
import dill
import os
import pandas as pd
import Geohash
import geocoder
from datetime import datetime

# This is a init commit from yutong
app = Flask(__name__)
app.config['DEBUG'] = False

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

class DensityPredict(Resource):
    def get(self):
        return "predict density"
    def put(self):
        hour = int(request.form['hour'])
        date = request.form['date']
        prcp = float(request.form['prcp'])*100
        snow = float(request.form['snow']) * 10
        tmax = float(request.form['tmax']) * 10
        tmin = float(request.form['tmin']) * 10
        date = pd.to_datetime(date)
        with open(os.path.join(APP_STATIC, 'uniquegeohash.pkl'), 'rb') as f:
            uniquegeohash = dill.load(f)
        with open(os.path.join(APP_STATIC, 'predict_pickup_density.pkl'), 'rb') as f:
            model = dill.load(f)
        x_dict = [{"pickup_geohash": geostr, "hour": hour, "dayofweek": date.dayofweek, 'month': date.month,'PRCP':prcp,'SNOW':snow,'TMAX':tmax,'TMIN':tmin} for geostr in uniquegeohash]
        x_df = pd.DataFrame(x_dict)
        y = model.predict(x_df)
        geodecode = [Geohash.decode(geocode) for geocode in uniquegeohash]
        yzipgeo = zip(y, geodecode)
        sortedlist = sorted(yzipgeo, key=lambda x: -x[0])
        top10address = []
        top10dict = {}
        for y, geodecode in sortedlist[0:50]:
            key = ",".join(geodecode)
            top10dict[key] = top10dict.get(key,0) + y
        top10res = []
        for key in top10dict:
            temptuple = (float(key.split(",")[0]),float(key.split(",")[1]))
            top10res.append([top10dict[key],temptuple])
        top10res = sorted(top10res,key=lambda x:-x[0])
        top10res = top10res[0:10] if len(top10res) > 10 else top10res
        for u,geodecode in top10res:
            g = geocoder.google([geodecode[0], geodecode[1]], method='reverse').address
            top10address.append(g)
        return {"top10": sortedlist[0:10],"top10address":top10address}

api.add_resource(DensityPredict,'/densitypredict')

#CORS ENABLE
@app.after_request
def after_request(response):
  response.headers.add('Access-Control-Allow-Origin', '*')
  response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
  response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
  return response



@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run()
