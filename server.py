import numpy as np
from PIL import Image
from feature_extractor import FeatureExtractor
from datetime import datetime
from flask import Flask, request, render_template
from pathlib import Path

app = Flask(__name__)

#adding logos to html page
UNO_logo = Path("./static/logos/UNO_logo.jpg")
SD_logo = Path("./static/logos/SouthDakota_logo.jpg")

# Read image features
fe = FeatureExtractor()
features = []
img_paths = []
for feature_path in Path("./static/feature").glob("*.npy"):
    features.append(np.load(feature_path))
    img_paths.append(Path("./static/img") / (feature_path.stem + ".jpg"))
features = np.array(features)



@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['query_img']
        # Save query image
        img = Image.open(file.stream)  # PIL image
        uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":", ".") + "_" + file.filename
        img.save(uploaded_img_path)

        # Run search
        query = fe.extract(img)
        dists = np.linalg.norm(features-query, axis=1)  # L2 distances to features
        ids = np.argsort(dists)[:30]  # Top 30 results
        scores = [(dists[id], img_paths[id],img_paths[id].stem + ".jpg") for id in ids]

        return render_template('index.html',
                               query_path=uploaded_img_path,
                               scores=scores,
							   uno_logo=UNO_logo,
							   sd_logo=SD_logo)
    else:
        return render_template('index.html',
							   uno_logo=UNO_logo,
							   sd_logo=SD_logo)


if __name__=="__main__":
    app.run("0.0.0.0",debug = True)




