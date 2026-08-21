
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    session,
    send_from_directory,
    url_for
)

import sqlite3
import os
import time
import requests
import numpy as np

from config import (
    OPENWEATHER_API_KEY
    
)
from openai import OpenAI

from PIL import Image

from werkzeug.utils import secure_filename

from werkzeug.security import (
    generate_password_hash,
    check_password_hash
)


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

app.secret_key = "smart_crop_care_ai"




# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


DATABASE = os.path.join(
    BASE_DIR,
    "cropcare.db"
)


UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "uploads"
)


MODEL_FOLDER = os.path.join(
    BASE_DIR,
    "model"
)


MODEL_PATH = os.path.join(
    MODEL_FOLDER,
    "model.h5"
)


CLASS_FILE = os.path.join(
    MODEL_FOLDER,
    "class_names.txt"
)


# ============================================================
# FLASK CONFIGURATION
# ============================================================

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

app.config["MAX_CONTENT_LENGTH"] = (
    10 * 1024 * 1024
)


# ============================================================
# CREATE REQUIRED FOLDERS
# ============================================================

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

os.makedirs(
    MODEL_FOLDER,
    exist_ok=True
)


# ============================================================
# ALLOWED IMAGE FORMATS
# ============================================================

ALLOWED_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg",
    "webp"
}


def allowed_file(filename):

    return (
        "." in filename
        and
        filename.rsplit(
            ".",
            1
        )[1].lower()
        in ALLOWED_EXTENSIONS
    )


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_db():

    conn = sqlite3.connect(
        DATABASE,
        timeout=30
    )

    conn.row_factory = sqlite3.Row

    conn.execute(
        "PRAGMA busy_timeout = 30000"
    )

    conn.execute(
        "PRAGMA journal_mode = WAL"
    )

    return conn


# ============================================================
# CREATE DATABASE TABLES
# ============================================================

def create_tables():

    conn = get_db()

    cursor = conn.cursor()


    # --------------------------------------------------------
    # USERS TABLE
    # --------------------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            fullname TEXT NOT NULL,

            mobile TEXT UNIQUE NOT NULL,

            email TEXT UNIQUE NOT NULL,

            password TEXT NOT NULL

        )
    """)


    # --------------------------------------------------------
    # DISEASE HISTORY TABLE
    # --------------------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS disease_history (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            user_id INTEGER NOT NULL,

            image_name TEXT,

            disease TEXT,

            confidence TEXT,

            health_score INTEGER DEFAULT 0,

            risk_level TEXT DEFAULT 'Unknown',

            risk_icon TEXT DEFAULT '⚪',

            recommendation TEXT,

            date TIMESTAMP
                DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY(user_id)
                REFERENCES users(id)

        )
    """)


    conn.commit()

    conn.close()


# ============================================================
# INITIALIZE DATABASE
# ============================================================

create_tables()

# ============================================================
# UPDATE OLD DATABASE
# ============================================================

def update_database():

    conn = get_db()
    cursor = conn.cursor()

    # Existing disease_history table me naye columns add karo
    new_columns = [
        ("health_score", "INTEGER DEFAULT 0"),
        ("risk_level", "TEXT DEFAULT 'Medium'"),
        ("risk_icon", "TEXT DEFAULT '🟡'"),
        ("recommendation", "TEXT DEFAULT ''")
    ]

    for column_name, column_type in new_columns:

        try:

            cursor.execute(
                f"""
                ALTER TABLE disease_history
                ADD COLUMN {column_name}
                {column_type}
                """
            )

            print(
                f"✅ Added column: {column_name}"
            )

        except sqlite3.OperationalError as e:

            if "duplicate column name" in str(e).lower():

                print(
                    f"✓ Column already exists: {column_name}"
                )

            else:

                print(
                    f"⚠️ Database update error for {column_name}:",
                    e
                )

    conn.commit()
    conn.close()


update_database()


# ============================================================
# AI MODEL VARIABLES
# ============================================================

model = None

class_names = []
# ============================================================
# AI MODEL
# ============================================================

def load_class_names():

    global class_names

    class_names = []


    # --------------------------------------------------------
    # FIRST: READ class_names.txt
    # --------------------------------------------------------

    if os.path.exists(CLASS_FILE):

        try:

            with open(
                CLASS_FILE,
                "r",
                encoding="utf-8"
            ) as file:

                names = [
                    line.strip()
                    for line in file
                    if line.strip()
                ]


            if names:

                class_names = names

                print(
                    "✅ Class names loaded:"
                )

                for index, name in enumerate(
                    class_names
                ):

                    print(
                        index,
                        "=",
                        name
                    )

                return


        except Exception as e:

            print(
                "⚠️ Error reading class_names.txt:",
                e
            )


    # --------------------------------------------------------
    # BACKUP: DATASET FOLDER
    # --------------------------------------------------------

    dataset_folder = os.path.join(
        BASE_DIR,
        "dataset"
    )


    if os.path.exists(dataset_folder):

        folders = []

        for item in os.listdir(
            dataset_folder
        ):

            path = os.path.join(
                dataset_folder,
                item
            )

            if os.path.isdir(path):

                folders.append(item)


        folders.sort()


        if len(folders) >= 2:

            class_names = folders

            print(
                "✅ Classes loaded from dataset:"
            )

            for index, name in enumerate(
                class_names
            ):

                print(
                    index,
                    "=",
                    name
                )


# ============================================================
# LOAD AI MODEL
# ============================================================

def load_ai_model():

    global model

    print()
    print(
        "=========================================="
    )

    print(
        "🌱 SMART CROP CARE AI"
    )

    print(
        "=========================================="
    )


    # --------------------------------------------------------
    # LOAD TENSORFLOW
    # --------------------------------------------------------

    try:

        import tensorflow as tf

        print(
            "✅ TensorFlow loaded"
        )

        print(
            "TensorFlow version:",
            tf.__version__
        )


    except Exception as e:

        print(
            "❌ TensorFlow could not be loaded"
        )

        print(
            "Error:",
            e
        )

        model = None

        return


    # --------------------------------------------------------
    # CHECK MODEL FILE
    # --------------------------------------------------------

    if not os.path.exists(
        MODEL_PATH
    ):

        print(
            "❌ model.h5 not found!"
        )

        print(
            "Expected location:"
        )

        print(
            MODEL_PATH
        )

        model = None

        return


    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    try:

        model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False
        )

        print(
            "✅ AI model loaded successfully"
        )

        print(
            "Model input shape:",
            model.input_shape
        )

        print(
            "Model output shape:",
            model.output_shape
        )


    except Exception as e:

        print(
            "❌ Error loading AI model:"
        )

        print(
            e
        )

        model = None

        return


    # --------------------------------------------------------
    # LOAD CLASS NAMES
    # --------------------------------------------------------

    load_class_names()


    if class_names:

        print(
            "✅ AI classes:"
        )

        for index, name in enumerate(
            class_names
        ):

            print(
                index,
                "=",
                name
            )

    else:

        print(
            "⚠️ Class names not found"
        )


    print(
        "=========================================="
    )


# ============================================================
# START AI MODEL
# ============================================================

load_ai_model()


# ============================================================
# AI PREDICTION
# ============================================================

def predict_disease(image_path):

    global model
    global class_names


    # --------------------------------------------------------
    # CHECK MODEL
    # --------------------------------------------------------

    if model is None:

        return (
            "AI Model Not Available",
            "N/A"
        )


    try:

        # ----------------------------------------------------
        # OPEN IMAGE
        # ----------------------------------------------------

        image = Image.open(
            image_path
        ).convert("RGB")


        # ----------------------------------------------------
        # GET MODEL INPUT SIZE
        # ----------------------------------------------------

        input_shape = model.input_shape


        if (
            len(input_shape) == 4
            and input_shape[1] is not None
            and input_shape[2] is not None
        ):

            height = int(
                input_shape[1]
            )

            width = int(
                input_shape[2]
            )

        else:

            height = 224
            width = 224


        # ----------------------------------------------------
        # RESIZE IMAGE
        # ----------------------------------------------------

        image = image.resize(
            (
                width,
                height
            )
        )


        # ----------------------------------------------------
        # CONVERT IMAGE TO NUMPY
        # ----------------------------------------------------

        image_array = np.array(
            image,
            dtype="float32"
        )


        # ----------------------------------------------------
        # NORMALIZE
        # ----------------------------------------------------

        image_array = (
            image_array / 255.0
        )


        # ----------------------------------------------------
        # ADD BATCH DIMENSION
        # ----------------------------------------------------

        image_array = np.expand_dims(
            image_array,
            axis=0
        )


        # ----------------------------------------------------
        # PREDICT
        # ----------------------------------------------------

        prediction = model.predict(
            image_array,
            verbose=0
        )


        prediction = np.array(
            prediction
        )


        # ----------------------------------------------------
        # MULTI-CLASS MODEL
        # ----------------------------------------------------

        if (
            prediction.ndim == 2
            and prediction.shape[1] > 1
        ):

            probabilities = prediction[0]

            predicted_index = int(
                np.argmax(
                    probabilities
                )
            )

            confidence = (
                float(
                    probabilities[
                        predicted_index
                    ]
                )
                * 100
            )


        # ----------------------------------------------------
        # BINARY MODEL
        # ----------------------------------------------------

        else:

            value = float(
                prediction.flatten()[0]
            )


            if value >= 0.5:

                predicted_index = 1

                confidence = (
                    value * 100
                )

            else:

                predicted_index = 0

                confidence = (
                    (1 - value) * 100
                )


        # ----------------------------------------------------
        # GET CLASS NAME
        # ----------------------------------------------------

        if (
            class_names
            and
            predicted_index < len(
                class_names
            )
        ):

            disease = class_names[
                predicted_index
            ]

        else:

            disease = (
                "Class "
                + str(
                    predicted_index
                )
            )


        # ----------------------------------------------------
        # RETURN RESULT
        # ----------------------------------------------------

        return (
            disease,
            f"{confidence:.2f}%"
        )


    except Exception as e:

        print(
            "❌ Prediction error:"
        )

        print(
            e
        )


        return (
            "Prediction Error",
            "N/A"
        )
    # ============================================================
# CROP HEALTH SCORE
# ============================================================

def calculate_health_score(
    disease,
    confidence
):

    try:

        confidence_value = float(
            str(confidence).replace(
                "%",
                ""
            )
        )

    except (
        ValueError,
        TypeError
    ):

        confidence_value = 0


    disease_name = str(
        disease
    ).lower()


    # --------------------------------------------------------
    # HEALTHY PLANT
    # --------------------------------------------------------

    if (
        "healthy" in disease_name
        or "normal" in disease_name
        or "class1" in disease_name
    ):

        health_score = (
            70
            + (
                confidence_value
                * 0.30
            )
        )


    # --------------------------------------------------------
    # DISEASED PLANT
    # --------------------------------------------------------

    elif (
        "disease" in disease_name
        or "diseased" in disease_name
        or "class2" in disease_name
    ):

        health_score = (
            100
            - confidence_value
        )


    # --------------------------------------------------------
    # UNKNOWN
    # --------------------------------------------------------

    else:

        health_score = 50


    health_score = max(
        0,
        min(
            100,
            round(
                health_score
            )
        )
    )


    # --------------------------------------------------------
    # RISK LEVEL
    # --------------------------------------------------------

    if health_score >= 75:

        risk_level = "Low"
        risk_icon = "🟢"


    elif health_score >= 50:

        risk_level = "Medium"
        risk_icon = "🟡"


    else:

        risk_level = "High"
        risk_icon = "🔴"


    return (
        health_score,
        risk_level,
        risk_icon
    )


# ============================================================
# SMART PLANT CARE RECOMMENDATION
# ============================================================

def get_recommendation(
    disease,
    health_score=None,
    risk_level=None
):

    disease_lower = str(
        disease
    ).lower()


    # --------------------------------------------------------
    # HEALTHY PLANT
    # --------------------------------------------------------

    if (
        "healthy" in disease_lower
        or "normal" in disease_lower
        or "class1" in disease_lower
    ):

        return (
            "🌱 Plant appears healthy. "
            "Continue proper watering, "
            "adequate sunlight and regular "
            "monitoring. Check the leaves "
            "regularly for early symptoms."
        )


    # --------------------------------------------------------
    # DISEASED PLANT
    # --------------------------------------------------------

    if (
        "disease" in disease_lower
        or "diseased" in disease_lower
        or "class2" in disease_lower
    ):

        if risk_level == "High":

            return (
                "🔴 High Risk detected. "
                "Remove severely affected "
                "leaves if appropriate, keep "
                "the affected plant area clean "
                "and monitor nearby plants. "
                "Consider expert agricultural "
                "advice for treatment."
            )


        if risk_level == "Medium":

            return (
                "🟡 Medium Risk detected. "
                "Monitor the crop carefully, "
                "avoid overwatering and remove "
                "clearly damaged leaves. "
                "Recheck the plant regularly."
            )


        return (
            "⚠️ Possible plant disease "
            "detected. Keep the crop area "
            "clean, maintain proper watering "
            "and monitor the affected leaves."
        )


    # --------------------------------------------------------
    # UNKNOWN CONDITION
    # --------------------------------------------------------

    return (
        "🔍 Plant condition could not be "
        "clearly identified. Upload a clear "
        "close-up image of the affected leaf "
        "for better AI analysis."
    )


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# SIGNUP
# ============================================================

@app.route(
    "/signup",
    methods=["GET", "POST"]
)
def signup():

    if request.method == "POST":

        fullname = request.form.get(
            "fullname",
            ""
        ).strip()

        mobile = request.form.get(
            "mobile",
            ""
        ).strip()

        email = request.form.get(
            "email",
            ""
        ).strip().lower()

        password = request.form.get(
            "password",
            ""
        )

        confirm_password = request.form.get(
            "confirm_password",
            ""
        )


        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        if not fullname:

            return (
                "❌ Please enter full name"
            )


        if not mobile:

            return (
                "❌ Please enter mobile number"
            )


        if not email:

            return (
                "❌ Please enter email"
            )


        if not password:

            return (
                "❌ Please enter password"
            )


        if password != confirm_password:

            return (
                "❌ Passwords do not match"
            )


        if len(password) < 6:

            return (
                "❌ Password must be "
                "at least 6 characters"
            )


        # ----------------------------------------------------
        # HASH PASSWORD
        # ----------------------------------------------------

        hashed_password = (
            generate_password_hash(
                password
            )
        )


        # ----------------------------------------------------
        # SAVE USER
        # ----------------------------------------------------

        conn = None

        try:

            conn = get_db()

            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO users
                (
                    fullname,
                    mobile,
                    email,
                    password
                )
                VALUES (?, ?, ?, ?)
            """, (
                fullname,
                mobile,
                email,
                hashed_password
            ))

            conn.commit()


        except sqlite3.IntegrityError:

            if conn:

                conn.rollback()

            return (
                "❌ Mobile number or "
                "Email already registered"
            )


        except sqlite3.OperationalError as e:

            if conn:

                conn.rollback()

            print(
                "❌ Database error:",
                e
            )

            return (
                "❌ Database is busy. "
                "Please stop the Flask server "
                "and start it again."
            )


        finally:

            if conn:

                conn.close()


        return redirect(
            url_for("login")
        )


    return render_template(
        "signup.html"
    )


# ============================================================
# LOGIN
# ============================================================

@app.route(
    "/login",
    methods=["GET", "POST"]
)
def login():

    if request.method == "POST":

        email = request.form.get(
            "email",
            ""
        ).strip().lower()

        password = request.form.get(
            "password",
            ""
        )


        conn = get_db()

        cursor = conn.cursor()


        cursor.execute("""
            SELECT
                id,
                fullname,
                password
            FROM users
            WHERE email = ?
        """, (
            email,
        ))


        user = cursor.fetchone()

        conn.close()


        if user:

            try:

                password_ok = (
                    check_password_hash(
                        user["password"],
                        password
                    )
                )

            except Exception:

                password_ok = False


            if password_ok:

                session["user_id"] = (
                    user["id"]
                )

                session["fullname"] = (
                    user["fullname"]
                )


                return redirect(
                    url_for(
                        "dashboard"
                    )
                )


        return (
            "❌ Invalid Email or Password"
        )


    return render_template(
        "login.html"
    )

    # ============================================================
# DASHBOARD
# ============================================================

@app.route("/dashboard")
def dashboard():

    # --------------------------------------------------------
    # LOGIN CHECK
    # --------------------------------------------------------

    if "user_id" not in session:

        return redirect(
            url_for("login")
        )


    user_id = session["user_id"]


    # --------------------------------------------------------
    # DATABASE
    # --------------------------------------------------------

    conn = get_db()

    cursor = conn.cursor()


    # --------------------------------------------------------
    # TOTAL SCANS
    # --------------------------------------------------------

    cursor.execute("""
        SELECT COUNT(*)
        FROM disease_history
        WHERE user_id = ?
    """, (
        user_id,
    ))

    total_scans = cursor.fetchone()[0]


    # --------------------------------------------------------
    # HEALTHY COUNT
    # --------------------------------------------------------

    cursor.execute("""
        SELECT COUNT(*)
        FROM disease_history
        WHERE user_id = ?
        AND (
            LOWER(disease) LIKE '%healthy%'
            OR LOWER(disease) LIKE '%normal%'
            OR LOWER(disease) LIKE '%class1%'
        )
    """, (
        user_id,
    ))

    healthy_count = cursor.fetchone()[0]


    # --------------------------------------------------------
    # DISEASED COUNT
    # --------------------------------------------------------

    cursor.execute("""
        SELECT COUNT(*)
        FROM disease_history
        WHERE user_id = ?
        AND NOT (
            LOWER(disease) LIKE '%healthy%'
            OR LOWER(disease) LIKE '%normal%'
            OR LOWER(disease) LIKE '%class1%'
        )
    """, (
        user_id,
    ))

    diseased_count = cursor.fetchone()[0]


    conn.close()


    # --------------------------------------------------------
    # DASHBOARD
    # --------------------------------------------------------

    return render_template(
        "dashboard.html",

        fullname=session.get(
            "fullname",
            "Farmer"
        ),

        total_scans=total_scans,

        healthy_count=healthy_count,

        diseased_count=diseased_count
    )


# ============================================================
# DETECT CROP DISEASE
# ============================================================

@app.route(
    "/detect",
    methods=["GET", "POST"]
)
def detect():

    # --------------------------------------------------------
    # LOGIN CHECK
    # --------------------------------------------------------

    if "user_id" not in session:

        return redirect(
            url_for("login")
        )


    # --------------------------------------------------------
    # OPEN DETECT PAGE
    # --------------------------------------------------------

    if request.method == "GET":

        return render_template(
            "detect.html"
        )


    # --------------------------------------------------------
    # CHECK IMAGE FIELD
    # --------------------------------------------------------

    if "crop_image" not in request.files:

        return (
            "❌ Please select a plant "
            "or leaf image."
        )


    file = request.files[
        "crop_image"
    ]


    # --------------------------------------------------------
    # CHECK EMPTY FILE
    # --------------------------------------------------------

    if file.filename == "":

        return (
            "❌ No image selected."
        )


    # --------------------------------------------------------
    # CHECK FILE TYPE
    # --------------------------------------------------------

    if not allowed_file(
        file.filename
    ):

        return (
            "❌ Please upload JPG, "
            "JPEG, PNG or WEBP image."
        )


    # --------------------------------------------------------
    # SECURE FILENAME
    # --------------------------------------------------------

    original_name = secure_filename(
        file.filename
    )


    name, extension = os.path.splitext(
        original_name
    )


    filename = (
        name
        + "_"
        + str(
            int(
                time.time()
            )
        )
        + extension
    )


    file_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )


    # --------------------------------------------------------
    # SAVE IMAGE
    # --------------------------------------------------------

    try:

        file.save(
            file_path
        )


        # Check that image can be opened
        test_image = Image.open(
            file_path
        )

        test_image.verify()


    except Exception as e:

        print(
            "❌ Image error:",
            e
        )


        if os.path.exists(
            file_path
        ):

            os.remove(
                file_path
            )


        return (
            "❌ Invalid or corrupted image."
        )


    # --------------------------------------------------------
    # AI PREDICTION
    # --------------------------------------------------------

    disease, confidence = (
        predict_disease(
            file_path
        )
    )


    # --------------------------------------------------------
    # CROP HEALTH SCORE
    # --------------------------------------------------------

    (
        health_score,
        risk_level,
        risk_icon
    ) = calculate_health_score(
        disease,
        confidence
    )


    print(
        "🌿 HEALTH SCORE:",
        health_score
    )

    print(
        "⚠️ RISK LEVEL:",
        risk_level
    )


    # --------------------------------------------------------
    # SMART RECOMMENDATION
    # --------------------------------------------------------

    recommendation = (
        get_recommendation(
            disease,
            health_score,
            risk_level
        )
    )


    # --------------------------------------------------------
    # SAVE ANALYSIS HISTORY
    # --------------------------------------------------------

    conn = None

    try:

        conn = get_db()

        cursor = conn.cursor()


        cursor.execute("""
            INSERT INTO disease_history
            (
                user_id,
                image_name,
                disease,
                confidence,
                health_score,
                risk_level,
                risk_icon,
                recommendation
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session["user_id"],
            filename,
            disease,
            confidence,
            health_score,
            risk_level,
            risk_icon,
            recommendation
        ))


        conn.commit()


    except sqlite3.OperationalError as e:

        if conn:

            conn.rollback()

        print(
            "❌ History database error:",
            e
        )


        # Image analysis can still be shown
        print(
            "⚠️ Analysis completed, "
            "but history could not be saved."
        )


    finally:

        if conn:

            conn.close()


    # --------------------------------------------------------
    # IMAGE URL
    # --------------------------------------------------------

    image_url = url_for(
        "uploaded_file",
        filename=filename
    )


    # --------------------------------------------------------
    # RESULT PAGE
    # --------------------------------------------------------

    return render_template(
        "result.html",

        image_path=image_url,

        message=(
            "✅ Plant image "
            "analyzed successfully!"
        ),

        disease=disease,

        confidence=confidence,

        health_score=health_score,

        risk_level=risk_level,

        risk_icon=risk_icon,

        recommendation=recommendation
    )

    # ============================================================
# HISTORY
# ============================================================

@app.route("/history")
def history():

    if "user_id" not in session:

        return redirect(
            url_for("login")
        )

    conn = get_db()

    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            image_name,
            disease,
            confidence,
            health_score,
            risk_level,
            risk_icon,
            recommendation,
            date
        FROM disease_history
        WHERE user_id = ?
        ORDER BY id DESC
    """, (
        session["user_id"],
    ))

    records = cursor.fetchall()

    conn.close()

    return render_template(
        "history.html",
        records=records
    )


# ============================================================
# UPLOADED IMAGES
# ============================================================

@app.route("/uploads/<filename>")
def uploaded_file(filename):

    return send_from_directory(
        app.config["UPLOAD_FOLDER"],
        filename
    )


# ============================================================
# PROFILE
# ============================================================

@app.route("/profile")
def profile():

    if "user_id" not in session:

        return redirect(
            url_for("login")
        )

    conn = get_db()

    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            id,
            fullname,
            mobile,
            email
        FROM users
        WHERE id = ?
    """, (
        session["user_id"],
    ))

    user = cursor.fetchone()

    conn.close()

    return render_template(
        "profile.html",
        user=user
    )


# ============================================================
# ANALYSIS REPORT
# ============================================================

@app.route("/report")
def report():

    if "user_id" not in session:

        return redirect(
            url_for("login")
        )


    user_id = session["user_id"]

    conn = get_db()

    cursor = conn.cursor()


    # --------------------------------------------------------
    # TOTAL SCANS
    # --------------------------------------------------------

    cursor.execute("""
        SELECT COUNT(*)
        FROM disease_history
        WHERE user_id = ?
    """, (
        user_id,
    ))

    total = cursor.fetchone()[0]


    # --------------------------------------------------------
    # HEALTHY SCANS
    # --------------------------------------------------------

    cursor.execute("""
        SELECT COUNT(*)
        FROM disease_history
        WHERE user_id = ?
        AND (
            LOWER(disease) LIKE '%healthy%'
            OR LOWER(disease) LIKE '%normal%'
            OR LOWER(disease) LIKE '%class1%'
        )
    """, (
        user_id,
    ))

    healthy_count = cursor.fetchone()[0]


    # --------------------------------------------------------
    # DISEASED SCANS
    # --------------------------------------------------------

    cursor.execute("""
        SELECT COUNT(*)
        FROM disease_history
        WHERE user_id = ?
        AND NOT (
            LOWER(disease) LIKE '%healthy%'
            OR LOWER(disease) LIKE '%normal%'
            OR LOWER(disease) LIKE '%class1%'
        )
    """, (
        user_id,
    ))

    diseased_count = cursor.fetchone()[0]


    # --------------------------------------------------------
    # AVERAGE HEALTH SCORE
    # --------------------------------------------------------

    cursor.execute("""
        SELECT
            AVG(health_score)
        FROM disease_history
        WHERE user_id = ?
    """, (
        user_id,
    ))

    average_score = cursor.fetchone()[0]


    if average_score is None:

        average_score = 0

    else:

        average_score = round(
            float(average_score),
            1
        )


    # --------------------------------------------------------
    # HIGH RISK COUNT
    # --------------------------------------------------------

    cursor.execute("""
        SELECT COUNT(*)
        FROM disease_history
        WHERE user_id = ?
        AND LOWER(risk_level) = 'high'
    """, (
        user_id,
    ))

    high_risk_count = cursor.fetchone()[0]


    # --------------------------------------------------------
    # MEDIUM RISK COUNT
    # --------------------------------------------------------

    cursor.execute("""
        SELECT COUNT(*)
        FROM disease_history
        WHERE user_id = ?
        AND LOWER(risk_level) = 'medium'
    """, (
        user_id,
    ))

    medium_risk_count = cursor.fetchone()[0]


    # --------------------------------------------------------
    # LOW RISK COUNT
    # --------------------------------------------------------

    cursor.execute("""
        SELECT COUNT(*)
        FROM disease_history
        WHERE user_id = ?
        AND LOWER(risk_level) = 'low'
    """, (
        user_id,
    ))

    low_risk_count = cursor.fetchone()[0]


    # --------------------------------------------------------
    # DISEASE SUMMARY
    # --------------------------------------------------------

    cursor.execute("""
        SELECT
            disease,
            COUNT(*) AS total
        FROM disease_history
        WHERE user_id = ?
        GROUP BY disease
        ORDER BY total DESC
    """, (
        user_id,
    ))

    disease_summary = cursor.fetchall()


    # --------------------------------------------------------
    # RECENT ANALYSIS
    # --------------------------------------------------------

    cursor.execute("""
        SELECT
            disease,
            confidence,
            health_score,
            risk_level,
            date
        FROM disease_history
        WHERE user_id = ?
        ORDER BY id DESC
        LIMIT 10
    """, (
        user_id,
    ))

    recent_records = cursor.fetchall()


    conn.close()


    # --------------------------------------------------------
    # REPORT PAGE
    # --------------------------------------------------------

    return render_template(
        "report.html",

        total=total,

        healthy_count=healthy_count,

        diseased_count=diseased_count,

        average_score=average_score,

        high_risk_count=high_risk_count,

        medium_risk_count=medium_risk_count,

        low_risk_count=low_risk_count,

        disease_summary=disease_summary,

        recent_records=recent_records
    )
    # ============================================================
# AI FARMING ASSISTANT
# ============================================================
# ============================================================
# AI FARMING CHATBOT
# ============================================================

# ============================================================
# FARMING ASSISTANT
# ============================================================

@app.route(
    "/chatbot",
    methods=["GET", "POST"]
)
def chatbot():

    if "user_id" not in session:

        return redirect(
            url_for("login")
        )


    answer = None
    question = None


    if request.method == "POST":

        question = request.form.get(
            "question",
            ""
        ).strip().lower()


        # ====================================================
        # WHEAT LEAVES YELLOW
        # ====================================================

        if (
            (
                "गेहूं" in question
                or "गेंहू" in question
                or "wheat" in question
            )
            and
            (
                "पीली" in question
                or "पीला" in question
                or "yellow" in question
                or "yellowing" in question
            )
            and
            (
                "पत्ती" in question
                or "पत्तियां" in question
                or "पत्ते" in question
                or "leaf" in question
                or "leaves" in question
            )
        ):

            answer = (
                "🌾 गेहूं की पत्तियां पीली होने के कई कारण "
                "हो सकते हैं। सबसे सामान्य कारण nitrogen "
                "की कमी, पानी की अधिकता या कमी, खराब drainage "
                "और कुछ रोग हो सकते हैं। यदि पुरानी पत्तियां "
                "पहले पीली हो रही हैं तो nitrogen deficiency "
                "की संभावना हो सकती है। मिट्टी की नमी जांचें "
                "और बिना जरूरत ज्यादा सिंचाई न करें। यदि "
                "पत्तियों पर धब्बे या असामान्य निशान भी दिखाई "
                "दे रहे हैं, तो disease की संभावना के लिए "
                "leaf की clear image से जांच करें।"
            )


        # ====================================================
        # GENERAL YELLOW LEAVES
        # ====================================================

        elif (
            (
                "पीली पत्ती" in question
                or "पीली पत्तियां" in question
                or "पीले पत्ते" in question
                or "yellow leaf" in question
                or "yellow leaves" in question
                or "leaves yellow" in question
                or "पत्तियां पीली" in question
            )
        ):

            answer = (
                "🌿 पत्तियां पीली होने के सामान्य कारणों में "
                "nutrient deficiency, पानी की अधिकता या कमी, "
                "खराब drainage और plant disease शामिल हो सकते "
                "हैं। मिट्टी की नमी और पत्तियों की स्थिति जांचें। "
                "अगर पत्तियों पर धब्बे या कीड़े दिखाई दे रहे हैं, "
                "तो disease की संभावना भी जांचें।"
            )


        # ====================================================
        # WATERING
        # ====================================================

        elif (
            "water" in question
            or "watering" in question
            or "irrigation" in question
            or "पानी" in question
            or "सिंचाई" in question
        ):

            answer = (
                "💧 फसल को उसकी crop और मिट्टी की जरूरत के "
                "अनुसार पानी दें। मिट्टी में लंबे समय तक पानी "
                "जमा न होने दें। बहुत अधिक या बहुत कम सिंचाई "
                "दोनों से पौधे को नुकसान हो सकता है।"
            )


        # ====================================================
        # FERTILIZER / NUTRIENTS
        # ====================================================

        elif (
            "fertilizer" in question
            or "fertiliser" in question
            or "खाद" in question
            or "उर्वरक" in question
            or "nitrogen" in question
            or "नाइट्रोजन" in question
            or "nutrient" in question
            or "पोषक" in question
        ):

            answer = (
                "🌱 Fertilizer का चुनाव crop और soil की "
                "जरूरत के अनुसार करें। Nitrogen की कमी से "
                "कई crops में पुरानी पत्तियां पीली पड़ सकती "
                "हैं। बिना soil condition जाने बहुत अधिक "
                "fertilizer डालने से बचें। संभव हो तो soil "
                "test के आधार पर fertilizer की मात्रा तय करें।"
            )


        # ====================================================
        # DISEASE
        # ====================================================

        elif (
            "disease" in question
            or "बीमारी" in question
            or "रोग" in question
            or "infection" in question
            or "संक्रमण" in question
        ):

            answer = (
                "🔎 Crop disease की सही पहचान के लिए affected "
                "leaf की clear image बहुत उपयोगी होती है। "
                "पत्तियों पर spots, रंग बदलना, सूखना या fungus "
                "जैसे लक्षण देखें। Dashboard से Crop Disease "
                "Detection खोलकर clear image upload करें।"
            )


        # ====================================================
        # PEST / INSECTS
        # ====================================================

        elif (
            "pest" in question
            or "insect" in question
            or "कीड़ा" in question
            or "कीड़े" in question
            or "कीट" in question
            or "insects" in question
        ):

            answer = (
                "🐛 अगर crop में कीड़े दिखाई दे रहे हैं तो पहले "
                "उनकी पहचान करें। पत्तियों के नीचे और नए shoots "
                "को ध्यान से देखें। बिना कीट की पहचान किए "
                "chemical pesticide का उपयोग न करें। जरूरत होने "
                "पर crop की clear image लेकर जांच करें।"
            )


        # ====================================================
        # WEATHER
        # ====================================================

        elif (
            "weather" in question
            or "मौसम" in question
            or "rain" in question
            or "बारिश" in question
            or "वर्षा" in question
        ):

            answer = (
                "🌦️ Weather crop-care planning में महत्वपूर्ण "
                "है। बारिश होने की संभावना में unnecessary "
                "irrigation से बचें। तेज हवा में spraying avoid "
                "करें और बहुत ज्यादा गर्मी में soil moisture "
                "पर नजर रखें।"
            )


        # ====================================================
        # WHEAT
        # ====================================================

        elif (
            "गेहूं" in question
            or "गेंहू" in question
            or "wheat" in question
        ):

            answer = (
                "🌾 गेहूं की अच्छी growth के लिए उचित सिंचाई, "
                "balanced nutrients, अच्छी मिट्टी और disease तथा "
                "pest monitoring जरूरी है। अगर गेहूं में कोई "
                "विशेष समस्या है तो उसके लक्षण बताएं, जैसे "
                "पत्तियां पीली होना, धब्बे, सूखना या कीड़े लगना।"
            )


        # ====================================================
        # TOMATO
        # ====================================================

        elif (
            "tomato" in question
            or "टमाटर" in question
        ):

            answer = (
                "🍅 टमाटर में पानी की सही मात्रा, अच्छी drainage "
                "और balanced nutrition जरूरी है। पत्तियों के "
                "पीले होने, spots या curling जैसे symptoms पर "
                "ध्यान दें।"
            )


        # ====================================================
        # RICE / PADDY
        # ====================================================

        elif (
            "rice" in question
            or "paddy" in question
            or "धान" in question
        ):

            answer = (
                "🌾 धान में उचित पानी management, nutrients और "
                "weed तथा disease monitoring महत्वपूर्ण है। "
                "पत्तियों का रंग बदलना या spots दिखाई देने पर "
                "समस्या की जल्दी पहचान करें।"
            )


        # ====================================================
        # HEALTH SCORE
        # ====================================================

        elif (
            "health score" in question
            or "health" in question
            or "स्वास्थ्य" in question
            or "हेल्थ स्कोर" in question
        ):

            answer = (
                "📊 Health Score crop की overall condition को "
                "समझने के लिए उपयोग किया जाता है। Score जितना "
                "अधिक होगा, सामान्यतः crop की स्थिति उतनी बेहतर "
                "मानी जाती है।"
            )


        # ====================================================
        # REPORT
        # ====================================================

        elif (
            "report" in question
            or "analysis" in question
            or "रिपोर्ट" in question
            or "विश्लेषण" in question
        ):

            answer = (
                "📈 Analysis Report में आपके scans, healthy और "
                "diseased classifications तथा उपलब्ध crop "
                "analysis information देखी जा सकती है।"
            )


        # ====================================================
        # HELP
        # ====================================================

        elif (
            "help" in question
            or "मदद" in question
            or "क्या पूछ" in question
            or "what can you do" in question
        ):

            answer = (
                "🤖 मैं farming assistant हूँ। आप crop disease, "
                "पानी, सिंचाई, fertilizer, nitrogen deficiency, "
                "पीली पत्तियां, insects, weather और crop-care "
                "से जुड़े सवाल पूछ सकते हैं।"
            )


        # ====================================================
        # DEFAULT
        # ====================================================

        else:

            answer = (
                "🤖 मैं आपके farming question को समझने की "
                "कोशिश कर रहा हूँ। कृपया crop का नाम और समस्या "
                "बताएं। उदाहरण: 'मेरी गेहूं की पत्तियां पीली "
                "क्यों हो रही हैं?' या 'टमाटर में पत्तियों पर "
                "दाग क्यों हैं?'"
            )


    return render_template(
        "chatbot.html",
        answer=answer,
        question=question
    )

# ============================================================
# WEATHER ADVICE
# ============================================================

# ============================================================
# WEATHER ADVICE
# ============================================================

@app.route(
    "/weather",
    methods=["GET", "POST"]
)
def weather():

    if "user_id" not in session:

        return redirect(
            url_for("login")
        )

    advice = None
    weather_data = None
    city = None

    if request.method == "POST":

        city = request.form.get(
            "city",
            ""
        ).strip()

        if city:

            try:

                api_url = (
                    "https://api.openweathermap.org/data/2.5/weather"
                )

                params = {
                    "q": city,
                    "appid": OPENWEATHER_API_KEY,
                    "units": "metric"
                }

                response = requests.get(
                    api_url,
                    params=params,
                    timeout=10
                )

                data = response.json()

                if response.status_code == 200:

                    weather_data = {
                        "city": data["name"],
                        "temperature": data["main"]["temp"],
                        "humidity": data["main"]["humidity"],
                        "condition": data["weather"][0]["description"],
                        "wind": data["wind"]["speed"]
                    }

                    # ------------------------------------------------
                    # TEMPERATURE ADVICE
                    # ------------------------------------------------

                    temperature = float(
                        data["main"]["temp"]
                    )

                    if temperature >= 35:

                        advice = (
                            "🌡️ Temperature high है। "
                            "Crop को adequate water दें "
                            "और बहुत गर्म समय में "
                            "unnecessary irrigation avoid करें।"
                        )

                    elif temperature <= 15:

                        advice = (
                            "❄️ Temperature low है। "
                            "Sensitive crops को cold "
                            "conditions से protect करें।"
                        )

                    else:

                        advice = (
                            "🌱 Temperature moderate है। "
                            "Normal crop care continue करें।"
                        )

                    # ------------------------------------------------
                    # WEATHER CONDITION ADVICE
                    # ------------------------------------------------

                    condition = data["weather"][0]["description"]

                    condition_lower = condition.lower()

                    if "rain" in condition_lower:

                        advice += (
                            " 🌧️ Rain की संभावना में "
                            "extra irrigation avoid करें।"
                        )

                    elif (
                        "sunny" in condition_lower
                        or "clear" in condition_lower
                    ):

                        advice += (
                            " ☀️ Sunny weather में "
                            "soil moisture monitor करें।"
                        )

                    elif "cloud" in condition_lower:

                        advice += (
                            " ☁️ Cloudy conditions में "
                            "soil moisture देखकर "
                            "irrigation करें।"
                        )

                    elif "storm" in condition_lower:

                        advice += (
                            " ⛈️ Storm conditions में "
                            "spraying avoid करें और "
                            "plants को support दें।"
                        )

                    elif "snow" in condition_lower:

                        advice += (
                            " ❄️ Cold conditions में "
                            "sensitive crops को protect करें।"
                        )

                else:

                    advice = (
                        "❌ City नहीं मिली। "
                        "City का सही नाम डालें।"
                    )

            except requests.RequestException:

                advice = (
                    "❌ Weather service से connection "
                    "नहीं हो पाया।"
                )

            except (KeyError, ValueError):

                advice = (
                    "❌ Weather data सही format में "
                    "नहीं मिला।"
                )

        else:

            advice = (
                "❌ Please enter a city name."
            )

    return render_template(
        "weather.html",
        advice=advice,
        weather_data=weather_data,
        city=city
    )

# ============================================================
# LANGUAGE SWITCH
# ============================================================

@app.route("/set-language/<language>")
def set_language(language):

    if language not in ["en", "hi"]:
        language = "en"

    session["language"] = language

    return redirect(
        request.referrer
        or url_for("dashboard")
    )
# ============================================================
# LOGOUT
# ============================================================

@app.route("/logout")
def logout():

    session.clear()

    return redirect(
        url_for("home")
    )


# ============================================================
# APPLICATION ERROR HANDLER
# ============================================================

@app.errorhandler(413)
def file_too_large(error):

    return (
        "❌ Image is too large. "
        "Please upload an image smaller "
        "than 10 MB."
    ), 413


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    print()

    print(
        "=========================================="
    )

    print(
        "🌱 SMART CROP CARE AI"
    )

    print(
        "=========================================="
    )

    print(
        "🚀 Starting Flask application..."
    )

    print(
        "📊 Dashboard: /dashboard"
    )

    print(
        "🔍 Disease Detection: /detect"
    )

    print(
        "📈 Analysis Report: /report"
    )

    print(
        "📋 Disease History: /history"
    )

    print(
        "🤖 AI Assistant: /chatbot"
    )

    print(
        "🌦️ Weather Advice: /weather"
    )

    print(
        "=========================================="
    )


    app.run(
        debug=True
    )

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    session,
    send_from_directory,
    url_for
)

import sqlite3
import os
import time
import requests
import numpy as np

from config import (
    OPENWEATHER_API_KEY
    
)
from openai import OpenAI

from PIL import Image

from werkzeug.utils import secure_filename

from werkzeug.security import (
    generate_password_hash,
    check_password_hash
)


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

app.secret_key = "smart_crop_care_ai"




# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


DATABASE = os.path.join(
    BASE_DIR,
    "cropcare.db"
)


UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "uploads"
)


MODEL_FOLDER = os.path.join(
    BASE_DIR,
    "model"
)


MODEL_PATH = os.path.join(
    MODEL_FOLDER,
    "model.h5"
)


CLASS_FILE = os.path.join(
    MODEL_FOLDER,
    "class_names.txt"
)


# ============================================================
# FLASK CONFIGURATION
# ============================================================

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

app.config["MAX_CONTENT_LENGTH"] = (
    10 * 1024 * 1024
)


# ============================================================
# CREATE REQUIRED FOLDERS
# ============================================================

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

os.makedirs(
    MODEL_FOLDER,
    exist_ok=True
)


# ============================================================
# ALLOWED IMAGE FORMATS
# ============================================================

ALLOWED_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg",
    "webp"
}


def allowed_file(filename):

    return (
        "." in filename
        and
        filename.rsplit(
            ".",
            1
        )[1].lower()
        in ALLOWED_EXTENSIONS
    )


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_db():

    conn = sqlite3.connect(
        DATABASE,
        timeout=30
    )

    conn.row_factory = sqlite3.Row

    conn.execute(
        "PRAGMA busy_timeout = 30000"
    )

    conn.execute(
        "PRAGMA journal_mode = WAL"
    )

    return conn


# ============================================================
# CREATE DATABASE TABLES
# ============================================================

def create_tables():

    conn = get_db()

    cursor = conn.cursor()


    # --------------------------------------------------------
    # USERS TABLE
    # --------------------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            fullname TEXT NOT NULL,

            mobile TEXT UNIQUE NOT NULL,

            email TEXT UNIQUE NOT NULL,

            password TEXT NOT NULL

        )
    """)


    # --------------------------------------------------------
    # DISEASE HISTORY TABLE
    # --------------------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS disease_history (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            user_id INTEGER NOT NULL,

            image_name TEXT,

            disease TEXT,

            confidence TEXT,

            health_score INTEGER DEFAULT 0,

            risk_level TEXT DEFAULT 'Unknown',

            risk_icon TEXT DEFAULT '⚪',

            recommendation TEXT,

            date TIMESTAMP
                DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY(user_id)
                REFERENCES users(id)

        )
    """)


    conn.commit()

    conn.close()


# ============================================================
# INITIALIZE DATABASE
# ============================================================

create_tables()

# ============================================================
# UPDATE OLD DATABASE
# ============================================================

def update_database():

    conn = get_db()
    cursor = conn.cursor()

    # Existing disease_history table me naye columns add karo
    new_columns = [
        ("health_score", "INTEGER DEFAULT 0"),
        ("risk_level", "TEXT DEFAULT 'Medium'"),
        ("risk_icon", "TEXT DEFAULT '🟡'"),
        ("recommendation", "TEXT DEFAULT ''")
    ]

    for column_name, column_type in new_columns:

        try:

            cursor.execute(
                f"""
                ALTER TABLE disease_history
                ADD COLUMN {column_name}
                {column_type}
                """
            )

            print(
                f"✅ Added column: {column_name}"
            )

        except sqlite3.OperationalError as e:

            if "duplicate column name" in str(e).lower():

                print(
                    f"✓ Column already exists: {column_name}"
                )

            else:

                print(
                    f"⚠️ Database update error for {column_name}:",
                    e
                )

    conn.commit()
    conn.close()


update_database()


# ============================================================
# AI MODEL VARIABLES
# ============================================================

model = None

class_names = []
# ============================================================
# AI MODEL
# ============================================================

def load_class_names():

    global class_names

    class_names = []


    # --------------------------------------------------------
    # FIRST: READ class_names.txt
    # --------------------------------------------------------

    if os.path.exists(CLASS_FILE):

        try:

            with open(
                CLASS_FILE,
                "r",
                encoding="utf-8"
            ) as file:

                names = [
                    line.strip()
                    for line in file
                    if line.strip()
                ]


            if names:

                class_names = names

                print(
                    "✅ Class names loaded:"
                )

                for index, name in enumerate(
                    class_names
                ):

                    print(
                        index,
                        "=",
                        name
                    )

                return


        except Exception as e:

            print(
                "⚠️ Error reading class_names.txt:",
                e
            )


    # --------------------------------------------------------
    # BACKUP: DATASET FOLDER
    # --------------------------------------------------------

    dataset_folder = os.path.join(
        BASE_DIR,
        "dataset"
    )


    if os.path.exists(dataset_folder):

        folders = []

        for item in os.listdir(
            dataset_folder
        ):

            path = os.path.join(
                dataset_folder,
                item
            )

            if os.path.isdir(path):

                folders.append(item)


        folders.sort()


        if len(folders) >= 2:

            class_names = folders

            print(
                "✅ Classes loaded from dataset:"
            )

            for index, name in enumerate(
                class_names
            ):

                print(
                    index,
                    "=",
                    name
                )


# ============================================================
# LOAD AI MODEL
# ============================================================

def load_ai_model():

    global model

    print()
    print(
        "=========================================="
    )

    print(
        "🌱 SMART CROP CARE AI"
    )

    print(
        "=========================================="
    )


    # --------------------------------------------------------
    # LOAD TENSORFLOW
    # --------------------------------------------------------

    try:

        import tensorflow as tf

        print(
            "✅ TensorFlow loaded"
        )

        print(
            "TensorFlow version:",
            tf.__version__
        )


    except Exception as e:

        print(
            "❌ TensorFlow could not be loaded"
        )

        print(
            "Error:",
            e
        )

        model = None

        return


    # --------------------------------------------------------
    # CHECK MODEL FILE
    # --------------------------------------------------------

    if not os.path.exists(
        MODEL_PATH
    ):

        print(
            "❌ model.h5 not found!"
        )

        print(
            "Expected location:"
        )

        print(
            MODEL_PATH
        )

        model = None

        return


    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    try:

        model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False
        )

        print(
            "✅ AI model loaded successfully"
        )

        print(
            "Model input shape:",
            model.input_shape
        )

        print(
            "Model output shape:",
            model.output_shape
        )


    except Exception as e:

        print(
            "❌ Error loading AI model:"
        )

        print(
            e
        )

        model = None

        return


    # --------------------------------------------------------
    # LOAD CLASS NAMES
    # --------------------------------------------------------

    load_class_names()


    if class_names:

        print(
            "✅ AI classes:"
        )

        for index, name in enumerate(
            class_names
        ):

            print(
                index,
                "=",
                name
            )

    else:

        print(
            "⚠️ Class names not found"
        )


    print(
        "=========================================="
    )


# ============================================================
# START AI MODEL
# ============================================================

load_ai_model()


# ============================================================
# AI PREDICTION
# ============================================================

def predict_disease(image_path):

    global model
    global class_names


    # --------------------------------------------------------
    # CHECK MODEL
    # --------------------------------------------------------

    if model is None:

        return (
            "AI Model Not Available",
            "N/A"
        )


    try:

        # ----------------------------------------------------
        # OPEN IMAGE
        # ----------------------------------------------------

        image = Image.open(
            image_path
        ).convert("RGB")


        # ----------------------------------------------------
        # GET MODEL INPUT SIZE
        # ----------------------------------------------------

        input_shape = model.input_shape


        if (
            len(input_shape) == 4
            and input_shape[1] is not None
            and input_shape[2] is not None
        ):

            height = int(
                input_shape[1]
            )

            width = int(
                input_shape[2]
            )

        else:

            height = 224
            width = 224


        # ----------------------------------------------------
        # RESIZE IMAGE
        # ----------------------------------------------------

        image = image.resize(
            (
                width,
                height
            )
        )


        # ----------------------------------------------------
        # CONVERT IMAGE TO NUMPY
        # ----------------------------------------------------

        image_array = np.array(
            image,
            dtype="float32"
        )


        # ----------------------------------------------------
        # NORMALIZE
        # ----------------------------------------------------

        image_array = (
            image_array / 255.0
        )


        # ----------------------------------------------------
        # ADD BATCH DIMENSION
        # ----------------------------------------------------

        image_array = np.expand_dims(
            image_array,
            axis=0
        )


        # ----------------------------------------------------
        # PREDICT
        # ----------------------------------------------------

        prediction = model.predict(
            image_array,
            verbose=0
        )


        prediction = np.array(
            prediction
        )


        # ----------------------------------------------------
        # MULTI-CLASS MODEL
        # ----------------------------------------------------

        if (
            prediction.ndim == 2
            and prediction.shape[1] > 1
        ):

            probabilities = prediction[0]

            predicted_index = int(
                np.argmax(
                    probabilities
                )
            )

            confidence = (
                float(
                    probabilities[
                        predicted_index
                    ]
                )
                * 100
            )


        # ----------------------------------------------------
        # BINARY MODEL
        # ----------------------------------------------------

        else:

            value = float(
                prediction.flatten()[0]
            )


            if value >= 0.5:

                predicted_index = 1

                confidence = (
                    value * 100
                )

            else:

                predicted_index = 0

                confidence = (
                    (1 - value) * 100
                )


        # ----------------------------------------------------
        # GET CLASS NAME
        # ----------------------------------------------------

        if (
            class_names
            and
            predicted_index < len(
                class_names
            )
        ):

            disease = class_names[
                predicted_index
            ]

        else:

            disease = (
                "Class "
                + str(
                    predicted_index
                )
            )


        # ----------------------------------------------------
        # RETURN RESULT
        # ----------------------------------------------------

        return (
            disease,
            f"{confidence:.2f}%"
        )


    except Exception as e:

        print(
            "❌ Prediction error:"
        )

        print(
            e
        )


        return (
            "Prediction Error",
            "N/A"
        )
    # ============================================================
# CROP HEALTH SCORE
# ============================================================

def calculate_health_score(
    disease,
    confidence
):

    try:

        confidence_value = float(
            str(confidence).replace(
                "%",
                ""
            )
        )

    except (
        ValueError,
        TypeError
    ):

        confidence_value = 0


    disease_name = str(
        disease
    ).lower()


    # --------------------------------------------------------
    # HEALTHY PLANT
    # --------------------------------------------------------

    if (
        "healthy" in disease_name
        or "normal" in disease_name
        or "class1" in disease_name
    ):

        health_score = (
            70
            + (
                confidence_value
                * 0.30
            )
        )


    # --------------------------------------------------------
    # DISEASED PLANT
    # --------------------------------------------------------

    elif (
        "disease" in disease_name
        or "diseased" in disease_name
        or "class2" in disease_name
    ):

        health_score = (
            100
            - confidence_value
        )


    # --------------------------------------------------------
    # UNKNOWN
    # --------------------------------------------------------

    else:

        health_score = 50


    health_score = max(
        0,
        min(
            100,
            round(
                health_score
            )
        )
    )


    # --------------------------------------------------------
    # RISK LEVEL
    # --------------------------------------------------------

    if health_score >= 75:

        risk_level = "Low"
        risk_icon = "🟢"


    elif health_score >= 50:

        risk_level = "Medium"
        risk_icon = "🟡"


    else:

        risk_level = "High"
        risk_icon = "🔴"


    return (
        health_score,
        risk_level,
        risk_icon
    )


# ============================================================
# SMART PLANT CARE RECOMMENDATION
# ============================================================

def get_recommendation(
    disease,
    health_score=None,
    risk_level=None
):

    disease_lower = str(
        disease
    ).lower()


    # --------------------------------------------------------
    # HEALTHY PLANT
    # --------------------------------------------------------

    if (
        "healthy" in disease_lower
        or "normal" in disease_lower
        or "class1" in disease_lower
    ):

        return (
            "🌱 Plant appears healthy. "
            "Continue proper watering, "
            "adequate sunlight and regular "
            "monitoring. Check the leaves "
            "regularly for early symptoms."
        )


    # --------------------------------------------------------
    # DISEASED PLANT
    # --------------------------------------------------------

    if (
        "disease" in disease_lower
        or "diseased" in disease_lower
        or "class2" in disease_lower
    ):

        if risk_level == "High":

            return (
                "🔴 High Risk detected. "
                "Remove severely affected "
                "leaves if appropriate, keep "
                "the affected plant area clean "
                "and monitor nearby plants. "
                "Consider expert agricultural "
                "advice for treatment."
            )


        if risk_level == "Medium":

            return (
                "🟡 Medium Risk detected. "
                "Monitor the crop carefully, "
                "avoid overwatering and remove "
                "clearly damaged leaves. "
                "Recheck the plant regularly."
            )


        return (
            "⚠️ Possible plant disease "
            "detected. Keep the crop area "
            "clean, maintain proper watering "
            "and monitor the affected leaves."
        )


    # --------------------------------------------------------
    # UNKNOWN CONDITION
    # --------------------------------------------------------

    return (
        "🔍 Plant condition could not be "
        "clearly identified. Upload a clear "
        "close-up image of the affected leaf "
        "for better AI analysis."
    )


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# SIGNUP
# ============================================================

@app.route(
    "/signup",
    methods=["GET", "POST"]
)
def signup():

    if request.method == "POST":

        fullname = request.form.get(
            "fullname",
            ""
        ).strip()

        mobile = request.form.get(
            "mobile",
            ""
        ).strip()

        email = request.form.get(
            "email",
            ""
        ).strip().lower()

        password = request.form.get(
            "password",
            ""
        )

        confirm_password = request.form.get(
            "confirm_password",
            ""
        )


        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        if not fullname:

            return (
                "❌ Please enter full name"
            )


        if not mobile:

            return (
                "❌ Please enter mobile number"
            )


        if not email:

            return (
                "❌ Please enter email"
            )


        if not password:

            return (
                "❌ Please enter password"
            )


        if password != confirm_password:

            return (
                "❌ Passwords do not match"
            )


        if len(password) < 6:

            return (
                "❌ Password must be "
                "at least 6 characters"
            )


        # ----------------------------------------------------
        # HASH PASSWORD
        # ----------------------------------------------------

        hashed_password = (
            generate_password_hash(
                password
            )
        )


        # ----------------------------------------------------
        # SAVE USER
        # ----------------------------------------------------

        conn = None

        try:

            conn = get_db()

            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO users
                (
                    fullname,
                    mobile,
                    email,
                    password
                )
                VALUES (?, ?, ?, ?)
            """, (
                fullname,
                mobile,
                email,
                hashed_password
            ))

            conn.commit()


        except sqlite3.IntegrityError:

            if conn:

                conn.rollback()

            return (
                "❌ Mobile number or "
                "Email already registered"
            )


        except sqlite3.OperationalError as e:

            if conn:

                conn.rollback()

            print(
                "❌ Database error:",
                e
            )

            return (
                "❌ Database is busy. "
                "Please stop the Flask server "
                "and start it again."
            )


        finally:

            if conn:

                conn.close()


        return redirect(
            url_for("login")
        )


    return render_template(
        "signup.html"
    )


# ============================================================
# LOGIN
# ============================================================

@app.route(
    "/login",
    methods=["GET", "POST"]
)
def login():

    if request.method == "POST":

        email = request.form.get(
            "email",
            ""
        ).strip().lower()

        password = request.form.get(
            "password",
            ""
        )


        conn = get_db()

        cursor = conn.cursor()


        cursor.execute("""
            SELECT
                id,
                fullname,
                password
            FROM users
            WHERE email = ?
        """, (
            email,
        ))


        user = cursor.fetchone()

        conn.close()


        if user:

            try:

                password_ok = (
                    check_password_hash(
                        user["password"],
                        password
                    )
                )

            except Exception:

                password_ok = False


            if password_ok:

                session["user_id"] = (
                    user["id"]
                )

                session["fullname"] = (
                    user["fullname"]
                )


                return redirect(
                    url_for(
                        "dashboard"
                    )
                )


        return (
            "❌ Invalid Email or Password"
        )


    return render_template(
        "login.html"
    )

    # ============================================================
# DASHBOARD
# ============================================================

@app.route("/dashboard")
def dashboard():

    # --------------------------------------------------------
    # LOGIN CHECK
    # --------------------------------------------------------

    if "user_id" not in session:

        return redirect(
            url_for("login")
        )


    user_id = session["user_id"]


    # --------------------------------------------------------
    # DATABASE
    # --------------------------------------------------------

    conn = get_db()

    cursor = conn.cursor()


    # --------------------------------------------------------
    # TOTAL SCANS
    # --------------------------------------------------------

    cursor.execute("""
        SELECT COUNT(*)
        FROM disease_history
        WHERE user_id = ?
    """, (
        user_id,
    ))

    total_scans = cursor.fetchone()[0]


    # --------------------------------------------------------
    # HEALTHY COUNT
    # --------------------------------------------------------

    cursor.execute("""
        SELECT COUNT(*)
        FROM disease_history
        WHERE user_id = ?
        AND (
            LOWER(disease) LIKE '%healthy%'
            OR LOWER(disease) LIKE '%normal%'
            OR LOWER(disease) LIKE '%class1%'
        )
    """, (
        user_id,
    ))

    healthy_count = cursor.fetchone()[0]


    # --------------------------------------------------------
    # DISEASED COUNT
    # --------------------------------------------------------

    cursor.execute("""
        SELECT COUNT(*)
        FROM disease_history
        WHERE user_id = ?
        AND NOT (
            LOWER(disease) LIKE '%healthy%'
            OR LOWER(disease) LIKE '%normal%'
            OR LOWER(disease) LIKE '%class1%'
        )
    """, (
        user_id,
    ))

    diseased_count = cursor.fetchone()[0]


    conn.close()


    # --------------------------------------------------------
    # DASHBOARD
    # --------------------------------------------------------

    return render_template(
        "dashboard.html",

        fullname=session.get(
            "fullname",
            "Farmer"
        ),

        total_scans=total_scans,

        healthy_count=healthy_count,

        diseased_count=diseased_count
    )


# ============================================================
# DETECT CROP DISEASE
# ============================================================

@app.route(
    "/detect",
    methods=["GET", "POST"]
)
def detect():

    # --------------------------------------------------------
    # LOGIN CHECK
    # --------------------------------------------------------

    if "user_id" not in session:

        return redirect(
            url_for("login")
        )


    # --------------------------------------------------------
    # OPEN DETECT PAGE
    # --------------------------------------------------------

    if request.method == "GET":

        return render_template(
            "detect.html"
        )


    # --------------------------------------------------------
    # CHECK IMAGE FIELD
    # --------------------------------------------------------

    if "crop_image" not in request.files:

        return (
            "❌ Please select a plant "
            "or leaf image."
        )


    file = request.files[
        "crop_image"
    ]


    # --------------------------------------------------------
    # CHECK EMPTY FILE
    # --------------------------------------------------------

    if file.filename == "":

        return (
            "❌ No image selected."
        )


    # --------------------------------------------------------
    # CHECK FILE TYPE
    # --------------------------------------------------------

    if not allowed_file(
        file.filename
    ):

        return (
            "❌ Please upload JPG, "
            "JPEG, PNG or WEBP image."
        )


    # --------------------------------------------------------
    # SECURE FILENAME
    # --------------------------------------------------------

    original_name = secure_filename(
        file.filename
    )


    name, extension = os.path.splitext(
        original_name
    )


    filename = (
        name
        + "_"
        + str(
            int(
                time.time()
            )
        )
        + extension
    )


    file_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )


    # --------------------------------------------------------
    # SAVE IMAGE
    # --------------------------------------------------------

    try:

        file.save(
            file_path
        )


        # Check that image can be opened
        test_image = Image.open(
            file_path
        )

        test_image.verify()


    except Exception as e:

        print(
            "❌ Image error:",
            e
        )


        if os.path.exists(
            file_path
        ):

            os.remove(
                file_path
            )


        return (
            "❌ Invalid or corrupted image."
        )


    # --------------------------------------------------------
    # AI PREDICTION
    # --------------------------------------------------------

    disease, confidence = (
        predict_disease(
            file_path
        )
    )


    # --------------------------------------------------------
    # CROP HEALTH SCORE
    # --------------------------------------------------------

    (
        health_score,
        risk_level,
        risk_icon
    ) = calculate_health_score(
        disease,
        confidence
    )


    print(
        "🌿 HEALTH SCORE:",
        health_score
    )

    print(
        "⚠️ RISK LEVEL:",
        risk_level
    )


    # --------------------------------------------------------
    # SMART RECOMMENDATION
    # --------------------------------------------------------

    recommendation = (
        get_recommendation(
            disease,
            health_score,
            risk_level
        )
    )


    # --------------------------------------------------------
    # SAVE ANALYSIS HISTORY
    # --------------------------------------------------------

    conn = None

    try:

        conn = get_db()

        cursor = conn.cursor()


        cursor.execute("""
            INSERT INTO disease_history
            (
                user_id,
                image_name,
                disease,
                confidence,
                health_score,
                risk_level,
                risk_icon,
                recommendation
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session["user_id"],
            filename,
            disease,
            confidence,
            health_score,
            risk_level,
            risk_icon,
            recommendation
        ))


        conn.commit()


    except sqlite3.OperationalError as e:

        if conn:

            conn.rollback()

        print(
            "❌ History database error:",
            e
        )


        # Image analysis can still be shown
        print(
            "⚠️ Analysis completed, "
            "but history could not be saved."
        )


    finally:

        if conn:

            conn.close()


    # --------------------------------------------------------
    # IMAGE URL
    # --------------------------------------------------------

    image_url = url_for(
        "uploaded_file",
        filename=filename
    )


    # --------------------------------------------------------
    # RESULT PAGE
    # --------------------------------------------------------

    return render_template(
        "result.html",

        image_path=image_url,

        message=(
            "✅ Plant image "
            "analyzed successfully!"
        ),

        disease=disease,

        confidence=confidence,

        health_score=health_score,

        risk_level=risk_level,

        risk_icon=risk_icon,

        recommendation=recommendation
    )

    # ============================================================
# HISTORY
# ============================================================

@app.route("/history")
def history():

    if "user_id" not in session:

        return redirect(
            url_for("login")
        )

    conn = get_db()

    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            image_name,
            disease,
            confidence,
            health_score,
            risk_level,
            risk_icon,
            recommendation,
            date
        FROM disease_history
        WHERE user_id = ?
        ORDER BY id DESC
    """, (
        session["user_id"],
    ))

    records = cursor.fetchall()

    conn.close()

    return render_template(
        "history.html",
        records=records
    )


# ============================================================
# UPLOADED IMAGES
# ============================================================

@app.route("/uploads/<filename>")
def uploaded_file(filename):

    return send_from_directory(
        app.config["UPLOAD_FOLDER"],
        filename
    )


# ============================================================
# PROFILE
# ============================================================

@app.route("/profile")
def profile():

    if "user_id" not in session:

        return redirect(
            url_for("login")
        )

    conn = get_db()

    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            id,
            fullname,
            mobile,
            email
        FROM users
        WHERE id = ?
    """, (
        session["user_id"],
    ))

    user = cursor.fetchone()

    conn.close()

    return render_template(
        "profile.html",
        user=user
    )


# ============================================================
# ANALYSIS REPORT
# ============================================================

@app.route("/report")
def report():

    if "user_id" not in session:

        return redirect(
            url_for("login")
        )


    user_id = session["user_id"]

    conn = get_db()

    cursor = conn.cursor()


    # --------------------------------------------------------
    # TOTAL SCANS
    # --------------------------------------------------------

    cursor.execute("""
        SELECT COUNT(*)
        FROM disease_history
        WHERE user_id = ?
    """, (
        user_id,
    ))

    total = cursor.fetchone()[0]


    # --------------------------------------------------------
    # HEALTHY SCANS
    # --------------------------------------------------------

    cursor.execute("""
        SELECT COUNT(*)
        FROM disease_history
        WHERE user_id = ?
        AND (
            LOWER(disease) LIKE '%healthy%'
            OR LOWER(disease) LIKE '%normal%'
            OR LOWER(disease) LIKE '%class1%'
        )
    """, (
        user_id,
    ))

    healthy_count = cursor.fetchone()[0]


    # --------------------------------------------------------
    # DISEASED SCANS
    # --------------------------------------------------------

    cursor.execute("""
        SELECT COUNT(*)
        FROM disease_history
        WHERE user_id = ?
        AND NOT (
            LOWER(disease) LIKE '%healthy%'
            OR LOWER(disease) LIKE '%normal%'
            OR LOWER(disease) LIKE '%class1%'
        )
    """, (
        user_id,
    ))

    diseased_count = cursor.fetchone()[0]


    # --------------------------------------------------------
    # AVERAGE HEALTH SCORE
    # --------------------------------------------------------

    cursor.execute("""
        SELECT
            AVG(health_score)
        FROM disease_history
        WHERE user_id = ?
    """, (
        user_id,
    ))

    average_score = cursor.fetchone()[0]


    if average_score is None:

        average_score = 0

    else:

        average_score = round(
            float(average_score),
            1
        )


    # --------------------------------------------------------
    # HIGH RISK COUNT
    # --------------------------------------------------------

    cursor.execute("""
        SELECT COUNT(*)
        FROM disease_history
        WHERE user_id = ?
        AND LOWER(risk_level) = 'high'
    """, (
        user_id,
    ))

    high_risk_count = cursor.fetchone()[0]


    # --------------------------------------------------------
    # MEDIUM RISK COUNT
    # --------------------------------------------------------

    cursor.execute("""
        SELECT COUNT(*)
        FROM disease_history
        WHERE user_id = ?
        AND LOWER(risk_level) = 'medium'
    """, (
        user_id,
    ))

    medium_risk_count = cursor.fetchone()[0]


    # --------------------------------------------------------
    # LOW RISK COUNT
    # --------------------------------------------------------

    cursor.execute("""
        SELECT COUNT(*)
        FROM disease_history
        WHERE user_id = ?
        AND LOWER(risk_level) = 'low'
    """, (
        user_id,
    ))

    low_risk_count = cursor.fetchone()[0]


    # --------------------------------------------------------
    # DISEASE SUMMARY
    # --------------------------------------------------------

    cursor.execute("""
        SELECT
            disease,
            COUNT(*) AS total
        FROM disease_history
        WHERE user_id = ?
        GROUP BY disease
        ORDER BY total DESC
    """, (
        user_id,
    ))

    disease_summary = cursor.fetchall()


    # --------------------------------------------------------
    # RECENT ANALYSIS
    # --------------------------------------------------------

    cursor.execute("""
        SELECT
            disease,
            confidence,
            health_score,
            risk_level,
            date
        FROM disease_history
        WHERE user_id = ?
        ORDER BY id DESC
        LIMIT 10
    """, (
        user_id,
    ))

    recent_records = cursor.fetchall()


    conn.close()


    # --------------------------------------------------------
    # REPORT PAGE
    # --------------------------------------------------------

    return render_template(
        "report.html",

        total=total,

        healthy_count=healthy_count,

        diseased_count=diseased_count,

        average_score=average_score,

        high_risk_count=high_risk_count,

        medium_risk_count=medium_risk_count,

        low_risk_count=low_risk_count,

        disease_summary=disease_summary,

        recent_records=recent_records
    )
    # ============================================================
# AI FARMING ASSISTANT
# ============================================================
# ============================================================
# AI FARMING CHATBOT
# ============================================================

# ============================================================
# FARMING ASSISTANT
# ============================================================

@app.route(
    "/chatbot",
    methods=["GET", "POST"]
)
def chatbot():

    if "user_id" not in session:

        return redirect(
            url_for("login")
        )


    answer = None
    question = None


    if request.method == "POST":

        question = request.form.get(
            "question",
            ""
        ).strip().lower()


        # ====================================================
        # WHEAT LEAVES YELLOW
        # ====================================================

        if (
            (
                "गेहूं" in question
                or "गेंहू" in question
                or "wheat" in question
            )
            and
            (
                "पीली" in question
                or "पीला" in question
                or "yellow" in question
                or "yellowing" in question
            )
            and
            (
                "पत्ती" in question
                or "पत्तियां" in question
                or "पत्ते" in question
                or "leaf" in question
                or "leaves" in question
            )
        ):

            answer = (
                "🌾 गेहूं की पत्तियां पीली होने के कई कारण "
                "हो सकते हैं। सबसे सामान्य कारण nitrogen "
                "की कमी, पानी की अधिकता या कमी, खराब drainage "
                "और कुछ रोग हो सकते हैं। यदि पुरानी पत्तियां "
                "पहले पीली हो रही हैं तो nitrogen deficiency "
                "की संभावना हो सकती है। मिट्टी की नमी जांचें "
                "और बिना जरूरत ज्यादा सिंचाई न करें। यदि "
                "पत्तियों पर धब्बे या असामान्य निशान भी दिखाई "
                "दे रहे हैं, तो disease की संभावना के लिए "
                "leaf की clear image से जांच करें।"
            )


        # ====================================================
        # GENERAL YELLOW LEAVES
        # ====================================================

        elif (
            (
                "पीली पत्ती" in question
                or "पीली पत्तियां" in question
                or "पीले पत्ते" in question
                or "yellow leaf" in question
                or "yellow leaves" in question
                or "leaves yellow" in question
                or "पत्तियां पीली" in question
            )
        ):

            answer = (
                "🌿 पत्तियां पीली होने के सामान्य कारणों में "
                "nutrient deficiency, पानी की अधिकता या कमी, "
                "खराब drainage और plant disease शामिल हो सकते "
                "हैं। मिट्टी की नमी और पत्तियों की स्थिति जांचें। "
                "अगर पत्तियों पर धब्बे या कीड़े दिखाई दे रहे हैं, "
                "तो disease की संभावना भी जांचें।"
            )


        # ====================================================
        # WATERING
        # ====================================================

        elif (
            "water" in question
            or "watering" in question
            or "irrigation" in question
            or "पानी" in question
            or "सिंचाई" in question
        ):

            answer = (
                "💧 फसल को उसकी crop और मिट्टी की जरूरत के "
                "अनुसार पानी दें। मिट्टी में लंबे समय तक पानी "
                "जमा न होने दें। बहुत अधिक या बहुत कम सिंचाई "
                "दोनों से पौधे को नुकसान हो सकता है।"
            )


        # ====================================================
        # FERTILIZER / NUTRIENTS
        # ====================================================

        elif (
            "fertilizer" in question
            or "fertiliser" in question
            or "खाद" in question
            or "उर्वरक" in question
            or "nitrogen" in question
            or "नाइट्रोजन" in question
            or "nutrient" in question
            or "पोषक" in question
        ):

            answer = (
                "🌱 Fertilizer का चुनाव crop और soil की "
                "जरूरत के अनुसार करें। Nitrogen की कमी से "
                "कई crops में पुरानी पत्तियां पीली पड़ सकती "
                "हैं। बिना soil condition जाने बहुत अधिक "
                "fertilizer डालने से बचें। संभव हो तो soil "
                "test के आधार पर fertilizer की मात्रा तय करें।"
            )


        # ====================================================
        # DISEASE
        # ====================================================

        elif (
            "disease" in question
            or "बीमारी" in question
            or "रोग" in question
            or "infection" in question
            or "संक्रमण" in question
        ):

            answer = (
                "🔎 Crop disease की सही पहचान के लिए affected "
                "leaf की clear image बहुत उपयोगी होती है। "
                "पत्तियों पर spots, रंग बदलना, सूखना या fungus "
                "जैसे लक्षण देखें। Dashboard से Crop Disease "
                "Detection खोलकर clear image upload करें।"
            )


        # ====================================================
        # PEST / INSECTS
        # ====================================================

        elif (
            "pest" in question
            or "insect" in question
            or "कीड़ा" in question
            or "कीड़े" in question
            or "कीट" in question
            or "insects" in question
        ):

            answer = (
                "🐛 अगर crop में कीड़े दिखाई दे रहे हैं तो पहले "
                "उनकी पहचान करें। पत्तियों के नीचे और नए shoots "
                "को ध्यान से देखें। बिना कीट की पहचान किए "
                "chemical pesticide का उपयोग न करें। जरूरत होने "
                "पर crop की clear image लेकर जांच करें।"
            )


        # ====================================================
        # WEATHER
        # ====================================================

        elif (
            "weather" in question
            or "मौसम" in question
            or "rain" in question
            or "बारिश" in question
            or "वर्षा" in question
        ):

            answer = (
                "🌦️ Weather crop-care planning में महत्वपूर्ण "
                "है। बारिश होने की संभावना में unnecessary "
                "irrigation से बचें। तेज हवा में spraying avoid "
                "करें और बहुत ज्यादा गर्मी में soil moisture "
                "पर नजर रखें।"
            )


        # ====================================================
        # WHEAT
        # ====================================================

        elif (
            "गेहूं" in question
            or "गेंहू" in question
            or "wheat" in question
        ):

            answer = (
                "🌾 गेहूं की अच्छी growth के लिए उचित सिंचाई, "
                "balanced nutrients, अच्छी मिट्टी और disease तथा "
                "pest monitoring जरूरी है। अगर गेहूं में कोई "
                "विशेष समस्या है तो उसके लक्षण बताएं, जैसे "
                "पत्तियां पीली होना, धब्बे, सूखना या कीड़े लगना।"
            )


        # ====================================================
        # TOMATO
        # ====================================================

        elif (
            "tomato" in question
            or "टमाटर" in question
        ):

            answer = (
                "🍅 टमाटर में पानी की सही मात्रा, अच्छी drainage "
                "और balanced nutrition जरूरी है। पत्तियों के "
                "पीले होने, spots या curling जैसे symptoms पर "
                "ध्यान दें।"
            )


        # ====================================================
        # RICE / PADDY
        # ====================================================

        elif (
            "rice" in question
            or "paddy" in question
            or "धान" in question
        ):

            answer = (
                "🌾 धान में उचित पानी management, nutrients और "
                "weed तथा disease monitoring महत्वपूर्ण है। "
                "पत्तियों का रंग बदलना या spots दिखाई देने पर "
                "समस्या की जल्दी पहचान करें।"
            )


        # ====================================================
        # HEALTH SCORE
        # ====================================================

        elif (
            "health score" in question
            or "health" in question
            or "स्वास्थ्य" in question
            or "हेल्थ स्कोर" in question
        ):

            answer = (
                "📊 Health Score crop की overall condition को "
                "समझने के लिए उपयोग किया जाता है। Score जितना "
                "अधिक होगा, सामान्यतः crop की स्थिति उतनी बेहतर "
                "मानी जाती है।"
            )


        # ====================================================
        # REPORT
        # ====================================================

        elif (
            "report" in question
            or "analysis" in question
            or "रिपोर्ट" in question
            or "विश्लेषण" in question
        ):

            answer = (
                "📈 Analysis Report में आपके scans, healthy और "
                "diseased classifications तथा उपलब्ध crop "
                "analysis information देखी जा सकती है।"
            )


        # ====================================================
        # HELP
        # ====================================================

        elif (
            "help" in question
            or "मदद" in question
            or "क्या पूछ" in question
            or "what can you do" in question
        ):

            answer = (
                "🤖 मैं farming assistant हूँ। आप crop disease, "
                "पानी, सिंचाई, fertilizer, nitrogen deficiency, "
                "पीली पत्तियां, insects, weather और crop-care "
                "से जुड़े सवाल पूछ सकते हैं।"
            )


        # ====================================================
        # DEFAULT
        # ====================================================

        else:

            answer = (
                "🤖 मैं आपके farming question को समझने की "
                "कोशिश कर रहा हूँ। कृपया crop का नाम और समस्या "
                "बताएं। उदाहरण: 'मेरी गेहूं की पत्तियां पीली "
                "क्यों हो रही हैं?' या 'टमाटर में पत्तियों पर "
                "दाग क्यों हैं?'"
            )


    return render_template(
        "chatbot.html",
        answer=answer,
        question=question
    )

# ============================================================
# WEATHER ADVICE
# ============================================================

# ============================================================
# WEATHER ADVICE
# ============================================================

@app.route(
    "/weather",
    methods=["GET", "POST"]
)
def weather():

    if "user_id" not in session:

        return redirect(
            url_for("login")
        )

    advice = None
    weather_data = None
    city = None

    if request.method == "POST":

        city = request.form.get(
            "city",
            ""
        ).strip()

        if city:

            try:

                api_url = (
                    "https://api.openweathermap.org/data/2.5/weather"
                )

                params = {
                    "q": city,
                    "appid": OPENWEATHER_API_KEY,
                    "units": "metric"
                }

                response = requests.get(
                    api_url,
                    params=params,
                    timeout=10
                )

                data = response.json()

                if response.status_code == 200:

                    weather_data = {
                        "city": data["name"],
                        "temperature": data["main"]["temp"],
                        "humidity": data["main"]["humidity"],
                        "condition": data["weather"][0]["description"],
                        "wind": data["wind"]["speed"]
                    }

                    # ------------------------------------------------
                    # TEMPERATURE ADVICE
                    # ------------------------------------------------

                    temperature = float(
                        data["main"]["temp"]
                    )

                    if temperature >= 35:

                        advice = (
                            "🌡️ Temperature high है। "
                            "Crop को adequate water दें "
                            "और बहुत गर्म समय में "
                            "unnecessary irrigation avoid करें।"
                        )

                    elif temperature <= 15:

                        advice = (
                            "❄️ Temperature low है। "
                            "Sensitive crops को cold "
                            "conditions से protect करें।"
                        )

                    else:

                        advice = (
                            "🌱 Temperature moderate है। "
                            "Normal crop care continue करें।"
                        )

                    # ------------------------------------------------
                    # WEATHER CONDITION ADVICE
                    # ------------------------------------------------

                    condition = data["weather"][0]["description"]

                    condition_lower = condition.lower()

                    if "rain" in condition_lower:

                        advice += (
                            " 🌧️ Rain की संभावना में "
                            "extra irrigation avoid करें।"
                        )

                    elif (
                        "sunny" in condition_lower
                        or "clear" in condition_lower
                    ):

                        advice += (
                            " ☀️ Sunny weather में "
                            "soil moisture monitor करें।"
                        )

                    elif "cloud" in condition_lower:

                        advice += (
                            " ☁️ Cloudy conditions में "
                            "soil moisture देखकर "
                            "irrigation करें।"
                        )

                    elif "storm" in condition_lower:

                        advice += (
                            " ⛈️ Storm conditions में "
                            "spraying avoid करें और "
                            "plants को support दें।"
                        )

                    elif "snow" in condition_lower:

                        advice += (
                            " ❄️ Cold conditions में "
                            "sensitive crops को protect करें।"
                        )

                else:

                    advice = (
                        "❌ City नहीं मिली। "
                        "City का सही नाम डालें।"
                    )

            except requests.RequestException:

                advice = (
                    "❌ Weather service से connection "
                    "नहीं हो पाया।"
                )

            except (KeyError, ValueError):

                advice = (
                    "❌ Weather data सही format में "
                    "नहीं मिला।"
                )

        else:

            advice = (
                "❌ Please enter a city name."
            )

    return render_template(
        "weather.html",
        advice=advice,
        weather_data=weather_data,
        city=city
    )

# ============================================================
# LANGUAGE SWITCH
# ============================================================

@app.route("/set-language/<language>")
def set_language(language):

    if language not in ["en", "hi"]:
        language = "en"

    session["language"] = language

    return redirect(
        request.referrer
        or url_for("dashboard")
    )
# ============================================================
# LOGOUT
# ============================================================

@app.route("/logout")
def logout():

    session.clear()

    return redirect(
        url_for("home")
    )


# ============================================================
# APPLICATION ERROR HANDLER
# ============================================================

@app.errorhandler(413)
def file_too_large(error):

    return (
        "❌ Image is too large. "
        "Please upload an image smaller "
        "than 10 MB."
    ), 413


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    print()

    print(
        "=========================================="
    )

    print(
        "🌱 SMART CROP CARE AI"
    )

    print(
        "=========================================="
    )

    print(
        "🚀 Starting Flask application..."
    )

    print(
        "📊 Dashboard: /dashboard"
    )

    print(
        "🔍 Disease Detection: /detect"
    )

    print(
        "📈 Analysis Report: /report"
    )

    print(
        "📋 Disease History: /history"
    )

    print(
        "🤖 AI Assistant: /chatbot"
    )

    print(
        "🌦️ Weather Advice: /weather"
    )

    print(
        "=========================================="
    )


    app.run(
        debug=True
    )
