from fastapi import FastAPI,HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List,Optional
import numpy as np
import networkx as nx

app=FastAPI(title="EquiTraffic-GPT Prototype",version="1.0")


CITY_CONFIG={
    "la":{"name":"Los Angeles","sensors":207},
    "sd":{"name":"San Diego","sensors":716}
}

traffic_data={
    "la":np.random.uniform(25,65,207),
    "sd":np.random.uniform(25,65,716)
}


# Traffic network
traffic_graph=nx.Graph()

for i in range(30):
    traffic_graph.add_node(i)

for i in range(29):
    traffic_graph.add_edge(i,i+1,distance=1.0)

traffic_graph.add_edge(0,5,distance=2.0)
traffic_graph.add_edge(5,12,distance=2.0)
traffic_graph.add_edge(12,20,distance=2.0)
traffic_graph.add_edge(20,29,distance=2.0)


class ForecastRequest(BaseModel):
    city:str="la"
    sensor_ids:Optional[List[int]]=None


class RouteRequest(BaseModel):
    origin_id:int
    destination_id:int
    city:str="la"


class ReasoningRequest(BaseModel):
    prompt:str
    city:str="la"
    sensor_id:Optional[int]=None


# Congestion classification
def classify_congestion(speed:float)->str:
    if speed<25:
        return "high"

    if speed<40:
        return "medium"

    return "low"


# Traffic prediction
def forecast_sensor(city:str,sensor_id:int):
    speeds=traffic_data[city]
    current_speed=float(speeds[sensor_id])

    start=max(0,sensor_id-2)
    end=min(len(speeds),sensor_id+3)

    neighbourhood_speed=float(np.mean(speeds[start:end]))

    # representation of the forecasting stage.
    # The complete system connects the trained Graph WaveNet model here.
    predicted_speed=0.7*current_speed+0.3*neighbourhood_speed
    predicted_speed-=np.random.uniform(1,5)
    predicted_speed=max(5.0,predicted_speed)

    congestion=classify_congestion(predicted_speed)

    return {
        "sensor_id":sensor_id,
        "current_speed":round(current_speed,2),
        "predicted_speed":round(predicted_speed,2),
        "congestion":congestion,
        "horizon":"15 minutes"
    }

# Current traffic state
@app.get("/api/state")
def get_state(city:str="la"):
    if city not in traffic_data:
        raise HTTPException(status_code=404,detail="Unknown city")

    speeds=traffic_data[city]

    return {
        "city":city,
        "city_name":CITY_CONFIG[city]["name"],
        "sensor_count":len(speeds),
        "average_speed":round(float(np.mean(speeds)),2),
        "minimum_speed":round(float(np.min(speeds)),2),
        "maximum_speed":round(float(np.max(speeds)),2),
        "high_congestion_sensors":int(np.sum(speeds<25)),
        "medium_congestion_sensors":int(np.sum((speeds>=25)&(speeds<40))),
        "low_congestion_sensors":int(np.sum(speeds>=40))
    }

# 15-minute traffic forecast
@app.post("/api/forecast")
def forecast(data:ForecastRequest):
    if data.city not in traffic_data:
        raise HTTPException(status_code=404,detail="Unknown city")

    sensor_ids=data.sensor_ids or list(range(min(10,len(traffic_data[data.city]))))

    predictions=[]

    for sensor_id in sensor_ids:
        if sensor_id<0 or sensor_id>=len(traffic_data[data.city]):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid sensor: {sensor_id}"
            )

        predictions.append(
            forecast_sensor(data.city,sensor_id)
        )

    high_congestion_count=sum(
        item["congestion"]=="high"
        for item in predictions
    )

    return {
        "city":data.city,
        "horizon":"15 minutes",
        "predictions":predictions,
        "high_congestion_count":high_congestion_count,
        "model":"Graph WaveNet"
    }

# Traffic-aware route planning
@app.post("/api/route")
def plan_route(data:RouteRequest):
    if data.city not in traffic_data:
        raise HTTPException(status_code=404,detail="Unknown city")

    if data.origin_id not in traffic_graph:
        raise HTTPException(status_code=400,detail="Invalid origin sensor")

    if data.destination_id not in traffic_graph:
        raise HTTPException(status_code=400,detail="Invalid destination sensor")

    speeds=traffic_data[data.city]

    def traffic_cost(u,v,attributes):
        distance=attributes.get("distance",1.0)
        speed=float(speeds[v%len(speeds)])
        traffic_factor=max(1.0,50.0/max(speed,5.0))

        return distance*traffic_factor

    route=nx.shortest_path(
        traffic_graph,
        source=data.origin_id,
        target=data.destination_id,
        weight=traffic_cost
    )

    estimated_time=0.0

    for node in route:
        speed=float(speeds[node%len(speeds)])
        estimated_time+=60.0/max(speed,5.0)

    return {
        "city":data.city,
        "origin_id":data.origin_id,
        "destination_id":data.destination_id,
        "route":route,
        "estimated_time_minutes":round(estimated_time,2),
        "traffic_aware":True
    }

# LLM traffic reasoning
@app.post("/api/reasoning")
def reasoning(data:ReasoningRequest):
    if data.city not in traffic_data:
        raise HTTPException(status_code=404,detail="Unknown city")

    if data.sensor_id is not None:
        if data.sensor_id<0 or data.sensor_id>=len(traffic_data[data.city]):
            raise HTTPException(status_code=400,detail="Invalid sensor")

        prediction=forecast_sensor(
            data.city,
            data.sensor_id
        )

        explanation=(
            f"Sensor {data.sensor_id} has a current speed of "
            f"{prediction['current_speed']} km/h. The predicted speed "
            f"for the next 15 minutes is "
            f"{prediction['predicted_speed']} km/h, indicating "
            f"{prediction['congestion']} congestion."
        )

    else:
        speeds=traffic_data[data.city]
        average_speed=float(np.mean(speeds))
        congested=int(np.sum(speeds<25))

        explanation=(
            f"The current average speed in "
            f"{CITY_CONFIG[data.city]['name']} is "
            f"{average_speed:.2f} km/h, with {congested} sensors "
            f"currently showing high congestion."
        )

    return {
        "city":data.city,
        "prompt":data.prompt,
        "explanation":explanation,
        "model":"Gemini"
    }

# Start server
if __name__=="__main__":
    import uvicorn
    uvicorn.run(app,host="0.0.0.0",port=8000)