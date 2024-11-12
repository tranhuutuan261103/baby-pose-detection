from dotenv import load_dotenv
load_dotenv()
from flask import Flask
from flask_cors import CORS
from controllers.baby_pose_detection_controller import bpd_bp
from controllers.baby_sleep_position_controller import bsp_bp

if __name__ == "__main__":
    app = Flask(__name__)
    CORS(app)
    app.register_blueprint(bpd_bp)
    app.register_blueprint(bsp_bp)
    app.run(debug=True, port=6666)