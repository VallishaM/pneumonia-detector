from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
from werkzeug.utils import secure_filename
import cv2
from keras.models import load_model
import numpy as np
from pymongo import MongoClient
import os.path
import json
from datetime import datetime

app = Flask(__name__)
CACHE = "./dynamic/cache.json"
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "static/uploads")
connection_string = "mongodb+srv://uname:meymey@cluster0.aaslhf7.mongodb.net/?retryWrites=true&w=majority"
client = MongoClient(connection_string)
dataset = client["test"]["dataset"]
app.secret_key = "secret key"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(["jpg", "jpeg"])

model = load_model("pneumonia.h5")

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def home():
    imgs = os.listdir(app.config["UPLOAD_FOLDER"])
    for img in imgs:
        os.remove(os.path.join(app.config["UPLOAD_FOLDER"], img))
    return render_template("upload.html")


@app.route("/analytics")
def get_analytics():
    dateformat = "%Y-%m-%d %H:%M:%S"
    if os.path.exists(CACHE):
        now_time = datetime.now()
        with open(CACHE, "r") as openfile:
            json_object = json.load(openfile)
            if (
                now_time - datetime.strptime(json_object["time"], dateformat)
            ).total_seconds() < 300:
                print(
                    (
                        now_time - datetime.strptime(json_object["time"], dateformat)
                    ).total_seconds(),
                    "cache",
                )
                return render_template(
                    "analytics.html",
                    male=float(json_object["male"]),
                    age0=float(json_object["age0"]),
                    age1=float(json_object["age1"]),
                    age2=float(json_object["age2"]),
                    age3=float(json_object["age3"]),
                )

    male = len(
        list(
            dataset.find(
                {"gender": "M", "labels": {"$ne": "No Finding"}}, {"age": 1, "_id": 0}
            )
        )
    ) / len(list(dataset.find({"labels": {"$ne": "No Finding"}}, {"age": 1, "_id": 0})))
    male = float(str(male)[0:5])
    age_mean = dataset.find({"labels": {"$ne": "No Finding"}}, {"age": 1, "_id": 0})
    count = [0, 0, 0, 0]
    total_count = 0
    for i in age_mean:
        try:
            age_i = int(i["age"])
        except:
            continue
        total_count += 1

        if age_i < 25:
            count[0] += 1
        elif age_i < 45:
            count[1] += 1
        elif age_i < 75:
            count[2] += 1
        else:
            count[3] += 1

    now = datetime.now()
    # convert to string
    date_time_str = now.strftime(dateformat)
    dic = {
        "time": date_time_str,
        "male": male,
        "age0": round(count[0] * 100 / total_count, 2),
        "age1": round(count[1] * 100 / total_count, 2),
        "age2": round(count[2] * 100 / total_count, 2),
        "age3": round(count[3] * 100 / total_count, 2),
    }
    flag = False
    if os.path.exists(CACHE):
        now_time = datetime.now()
        with open(CACHE, "r") as openfile:
            json_object = json.load(openfile)
            if (
                now_time - datetime.strptime(json_object["time"], dateformat)
            ).total_seconds() > 300:
                flag = True
    if not os.path.exists(CACHE):
        flag = True
    if flag:
        print("Redo")
        with open(CACHE, "w") as outfile:
            json.dump(dic, outfile)

    return render_template(
        "analytics.html",
        male=male,
        age0=round(count[0] * 100 / total_count, 2),
        age1=round(count[1] * 100 / total_count, 2),
        age2=round(count[2] * 100 / total_count, 2),
        age3=round(count[3] * 100 / total_count, 2),
    )


@app.route("/predict", methods=["POST"])
def upload_image():

    if "file" not in request.files:
        flash("No file part")
        return redirect(request.url)

    file = request.files["file"]

    if file.filename == "":
        flash("No image selected for uploading")
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))

        pred_class, confidence = "", 0
        img = cv2.imread(os.path.join(app.config["UPLOAD_FOLDER"], filename))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.convertScaleAbs(img, alpha=1.3, beta=0)
            img = cv2.resize(img, (128, 128))
            img = np.array(img) / 255
            img = np.reshape(img, (1, 128, 128, 1))
            new_pred = ""
            prediction = model.predict(img)[0][0]
            if prediction < 0.5:
                pred_class = "NORMAL"
                new_pred = "No Finding"
                confidence = 1 - prediction
            else:
                pred_class = "PNEUMONIA"
                new_pred = "Pneumonia"
                confidence = prediction

            flash("Image successfully uploaded and displayed below")
        if confidence > 0.85:
            dataset.insert_one(
                {
                    "labels": new_pred,
                    "gender": request.form.get("gender"),
                    "age": int(request.form.get("age")),
                }
            )
        return render_template(
            "upload.html",
            filename=filename,
            pred=pred_class,
            conf=round(confidence * 100, 2),
        )

    else:
        flash("Allowed image types are - jpg, jpeg")
        return redirect(request.url)


@app.route("/display/<filename>")
def display_image(filename):
    # print('display_image filename: ' + filename)
    return redirect(url_for("static", filename="uploads/" + filename), code=301)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
