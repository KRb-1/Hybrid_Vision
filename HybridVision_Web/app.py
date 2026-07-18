from flask import Flask, render_template

app = Flask(__name__, template_folder="templates")
app.secret_key = "secret key"


@app.route("/")
def home():
    return render_template("Home.html")


@app.route("/detection", methods=["GET"])
def detection():
    return render_template("Detection.html")

@app.route("/comparison", methods=["GET"])
def comparison():
    return render_template("Comparison.html")

@app.route("/tracking", methods=["GET"])
def tracking():
    return render_template("Tracking.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8888, debug=True)