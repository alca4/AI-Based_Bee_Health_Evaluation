from flask import Flask, request, redirect, jsonify, flash, send_from_directory
from werkzeug.utils import secure_filename
import os
import tempfile
from app_video.app_video import showVideo
from audio import showAudio
from image import showImage
import json
import time
from moviepy.editor import VideoFileClip



app = Flask(__name__)
app.secret_key = "a123456"  
app.config["UPLOAD_FOLDER"] = "static/uploads/"  
app.config["VIDEO_UPLOAD_FOLDER"] = "app_video/"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "webp"}
HISTORY_JSON = "data.json"


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():

    return """
    <h1>File Upload</h1>
    <form method="post" action="/upload" enctype="multipart/form-data">
        <input type="file" name="file" />
        <input type="submit" value="Upload" />
    </form>
    """


@app.route("/test", methods=["POST", "GET"])
def test():
    return jsonify({"error": "No file part"})


@app.route("/loadHistory", methods=["POST", "GET"])
def loadHistory():

    items = []
    with open(HISTORY_JSON, "r") as json_file:
        print(f"json_file===={json_file} items={items}")
        items = json.load(json_file)

    sorted_list = sorted(items, key=lambda x: x['date'], reverse=True)

    return jsonify({"data": sorted_list,"success":True})


def secure_filename(filename):
    return filename.replace(os.path.sep, "_")


@app.route("/upload", methods=["POST", "GET"])
def upload_file():
    if "photo" not in request.files:
        return jsonify({"error": "No file part"})
    file = request.files["photo"]
    ctype = request.form["type"]
    print(f"request.type={ctype}")
    if file.filename == "":
        flash("No selected file")
        return jsonify({"error": "No selected file"})

    upload_folder = "static/uploads/"
    data = {}
    if file:  
        filename = secure_filename(file.filename)
        print(f"filename={filename}")
        file.save(os.path.join(upload_folder, filename))  
        # return f'File uploaded successfully: {filename}'
        url = f"{upload_folder}{filename}"
        if ctype == "video":
            # file.save(os.path.join(upload_folder, filename))
            data = showVideo(filename)
            # pass
        if ctype == "audio":
            # audio


# Upload mp4
            video = VideoFileClip(url)
            print(f"url={url}")

            # Retrieve WAV files
            video.audio.write_audiofile(f"{upload_folder}output_audio.wav")
            data = showAudio(f"output_audio.wav")

        if ctype == "image":
            # audio
            data = showImage(filename)

        items = []
        with open(HISTORY_JSON, "r") as json_file:
            print(f"json_file===={json_file} items={items}")
            items = json.load(json_file)
        items.append({"data": data, "type": ctype, "date": int(time.time())})


        # Saved JSON Files
        with open(HISTORY_JSON, "w") as json_file:
            json.dump(items, json_file, indent=4)  

        print(f"url===={url} items={items}")

        return jsonify(
            {
                "success": True,
                "message": "File uploaded successfully",
                "url": url,
                "data": data,
            }
        )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
    # showVideo("bee2.mp4")
