from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timezone
import os
import requests
import json
import google.generativeai as genai

app = Flask(__name__)
app.config['SECRET_KEY'] = 'thendo-fitness-tracker-secret-key-2025'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///running_tracker.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Configure Gemini AI
GEMINI_API_KEY = 'AIzaSyCdj4LSp8G7EAlXysrKDYpXMl48VYSmcgg'
genai.configure(api_key=GEMINI_API_KEY)

db = SQLAlchemy(app)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access your fitness tracker.'
login_manager.login_message_category = 'info'

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    first_name = db.Column(db.String(50), nullable=False)
    last_name = db.Column(db.String(50), nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    
    # Relationships
    runs = db.relationship('Run', backref='user', lazy=True)
    gym_sessions = db.relationship('GymSession', backref='user', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f'<User {self.username}>'
class Run(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    date = db.Column(db.Date, nullable=False, default=lambda: datetime.now(timezone.utc).date())
    distance = db.Column(db.Float, nullable=False)
    duration = db.Column(db.Integer)  # in minutes
    notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    def __repr__(self):
        return f'<Run {self.id}: {self.distance}km on {self.date}>'

    @property
    def pace(self):
        """Calculate pace in minutes per kilometer"""
        if self.duration and self.distance:
            return round(self.duration / self.distance, 2)
        return None

class GymSession(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    date = db.Column(db.Date, nullable=False, default=lambda: datetime.now(timezone.utc).date())
    exercise_name = db.Column(db.String(100), nullable=False)
    exercise_type = db.Column(db.String(50), nullable=False)  # strength, cardio, flexibility
    sets = db.Column(db.Integer)
    reps = db.Column(db.String(100))  # can be "12, 10, 8" for multiple sets
    weight = db.Column(db.Float)  # in kg
    duration = db.Column(db.Integer)  # in minutes for cardio exercises
    notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    def __repr__(self):
        return f'<GymSession {self.id}: {self.exercise_name} on {self.date}>'

    @property
    def formatted_reps(self):
        """Format reps for display"""
        if self.reps:
            return self.reps
        return "N/A"

# Weather function
def get_weather_data():
    """Get weather data from OpenWeatherMap API"""
    try:
        # You can get a free API key from openweathermap.org
        # For now, we'll use a mock response for demonstration
        # Replace this with actual API call when you have an API key
        
        # Mock weather data (replace with actual API call)
        weather_data = {
            'temperature': 22,
            'description': 'Partly Cloudy',
            'icon': 'partly-cloudy',
            'humidity': 65,
            'wind_speed': 12,
            'city': 'Your City'
        }
        
        # Uncomment this when you have an API key:
        api_key = "600da4aa9340fc8d8b6093c528f3c95e"
        city = "Polokwane"  # Change to your city
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            weather_data = {
                'temperature': round(data['main']['temp']),
                'description': data['weather'][0]['description'].title(),
                'icon': data['weather'][0]['main'].lower(),
                'humidity': data['main']['humidity'],
                'wind_speed': round(data['wind']['speed'] * 3.6),  # Convert m/s to km/h
                'city': data['name']
            }
        
        return weather_data
    except Exception as e:
        # Return default data if API fails
        return {
            'temperature': 20,
            'description': 'Weather Unavailable',
            'icon': 'default',
            'humidity': 50,
            'wind_speed': 10,
            'city': 'Your City'
        }

# Authentication Routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        remember = bool(request.form.get('remember'))
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user, remember=remember)
            next_page = request.args.get('next')
            flash(f'Welcome back, {user.first_name}!', 'success')
            return redirect(next_page) if next_page else redirect(url_for('index'))
        else:
            flash('Invalid username or password. Please try again.', 'error')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        first_name = request.form.get('first_name')
        last_name = request.form.get('last_name')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        # Validation
        if password != confirm_password:
            flash('Passwords do not match.', 'error')
            return render_template('register.html')
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists. Please choose a different one.', 'error')
            return render_template('register.html')
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered. Please use a different email.', 'error')
            return render_template('register.html')
        
        # Create new user
        user = User(
            username=username,
            email=email,
            first_name=first_name,
            last_name=last_name
        )
        user.set_password(password)
        
        try:
            db.session.add(user)
            db.session.commit()
            flash('Registration successful! You can now log in.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            flash('Registration failed. Please try again.', 'error')
            db.session.rollback()
    
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out successfully.', 'info')
    return redirect(url_for('login'))

# Main Routes
@app.route('/')
@login_required
def index():
    runs = Run.query.filter_by(user_id=current_user.id).order_by(Run.date.desc()).all()
    total_distance = sum(run.distance for run in runs)
    total_runs = len(runs)
    avg_distance = round(total_distance / total_runs, 2) if total_runs > 0 else 0
    
    # Get last 10 runs for quick chart
    recent_runs = runs[:10]
    recent_chart_data = {
        'dates': [run.date.strftime('%m/%d') for run in reversed(recent_runs)],
        'distances': [float(run.distance) for run in reversed(recent_runs)]
    }
    
    # Get weather data
    weather = get_weather_data()
    
    return render_template('index.html', 
                         runs=runs, 
                         total_distance=total_distance,
                         total_runs=total_runs,
                         avg_distance=avg_distance,
                         recent_chart_data=recent_chart_data,
                         weather=weather)

@app.route('/add_run', methods=['GET', 'POST'])
@login_required
def add_run():
    if request.method == 'POST':
        distance = request.form.get('distance')
        duration = request.form.get('duration')
        notes = request.form.get('notes')
        date = request.form.get('date')
        
        try:
            distance = float(distance)
            duration = int(duration) if duration else None
            run_date = datetime.strptime(date, '%Y-%m-%d').date() if date else datetime.now(timezone.utc).date()
            
            new_run = Run(
                user_id=current_user.id,
                distance=distance,
                duration=duration,
                notes=notes,
                date=run_date
            )
            
            db.session.add(new_run)
            db.session.commit()
            flash(f'Successfully logged {distance}km run!', 'success')
            return redirect(url_for('index'))
            
        except ValueError:
            flash('Please enter valid numbers for distance and duration.', 'error')
        except Exception as e:
            flash(f'Error adding run: {str(e)}', 'error')
    
    # Pass today's date to the template
    today = datetime.now(timezone.utc).date().strftime('%Y-%m-%d')
    return render_template('add_run.html', today=today)

@app.route('/delete_run/<int:run_id>')
@login_required
def delete_run(run_id):
    run = Run.query.filter_by(id=run_id, user_id=current_user.id).first_or_404()
    try:
        db.session.delete(run)
        db.session.commit()
        flash('Run deleted successfully!', 'success')
    except Exception as e:
        flash(f'Error deleting run: {str(e)}', 'error')
    
    return redirect(url_for('index'))

@app.route('/gym')
@login_required
def gym():
    gym_sessions = GymSession.query.filter_by(user_id=current_user.id).order_by(GymSession.date.desc()).all()
    total_sessions = len(gym_sessions)
    
    # Calculate statistics by exercise type
    strength_sessions = [s for s in gym_sessions if s.exercise_type == 'strength']
    cardio_sessions = [s for s in gym_sessions if s.exercise_type == 'cardio']
    flexibility_sessions = [s for s in gym_sessions if s.exercise_type == 'flexibility']
    
    # Get recent sessions for chart
    recent_sessions = gym_sessions[:10]
    recent_chart_data = {
        'dates': [session.date.strftime('%m/%d') for session in reversed(recent_sessions)],
        'types': [session.exercise_type for session in reversed(recent_sessions)]
    }
    
    # Prepare comprehensive chart data
    chart_data = {
        'workout_distribution': {
            'labels': ['Strength', 'Cardio', 'Flexibility'],
            'data': [len(strength_sessions), len(cardio_sessions), len(flexibility_sessions)],
            'colors': ['#dc3545', '#28a745', '#17a2b8']
        },
        'workout_frequency': {
            'dates': [],
            'strength_count': [],
            'cardio_count': [],
            'flexibility_count': []
        },
        'weight_progression': {
            'dates': [],
            'weights': [],
            'exercises': []
        },
        'duration_trends': {
            'dates': [],
            'durations': [],
            'types': []
        },
        'popular_exercises': {
            'names': [],
            'counts': []
        }
    }
    
    if gym_sessions:
        # Calculate monthly workout frequency
        from collections import defaultdict
        monthly_data = defaultdict(lambda: {'strength': 0, 'cardio': 0, 'flexibility': 0})
        
        for session in gym_sessions:
            month_key = f"{session.date.year}-{session.date.month:02d}"
            monthly_data[month_key][session.exercise_type] += 1
        
        sorted_months = sorted(monthly_data.keys())
        chart_data['workout_frequency']['dates'] = sorted_months
        chart_data['workout_frequency']['strength_count'] = [monthly_data[month]['strength'] for month in sorted_months]
        chart_data['workout_frequency']['cardio_count'] = [monthly_data[month]['cardio'] for month in sorted_months]
        chart_data['workout_frequency']['flexibility_count'] = [monthly_data[month]['flexibility'] for month in sorted_months]
        
        # Weight progression for strength exercises
        strength_with_weight = [s for s in strength_sessions if s.weight and s.weight > 0]
        if strength_with_weight:
            # Sort by date and take recent 20 entries
            strength_with_weight.sort(key=lambda x: x.date)
            recent_weight_sessions = strength_with_weight[-20:]
            chart_data['weight_progression']['dates'] = [s.date.strftime('%m/%d') for s in recent_weight_sessions]
            chart_data['weight_progression']['weights'] = [float(s.weight) for s in recent_weight_sessions]
            chart_data['weight_progression']['exercises'] = [s.exercise_name for s in recent_weight_sessions]
        
        # Duration trends for cardio and flexibility
        duration_sessions = [s for s in gym_sessions if s.duration and s.duration > 0]
        if duration_sessions:
            duration_sessions.sort(key=lambda x: x.date)
            recent_duration_sessions = duration_sessions[-15:]
            chart_data['duration_trends']['dates'] = [s.date.strftime('%m/%d') for s in recent_duration_sessions]
            chart_data['duration_trends']['durations'] = [s.duration for s in recent_duration_sessions]
            chart_data['duration_trends']['types'] = [s.exercise_type for s in recent_duration_sessions]
        
        # Popular exercises
        exercise_counts = defaultdict(int)
        for session in gym_sessions:
            exercise_counts[session.exercise_name] += 1
        
        # Get top 8 most popular exercises
        popular = sorted(exercise_counts.items(), key=lambda x: x[1], reverse=True)[:8]
        chart_data['popular_exercises']['names'] = [ex[0] for ex in popular]
        chart_data['popular_exercises']['counts'] = [ex[1] for ex in popular]
    
    return render_template('gym.html', 
                         gym_sessions=gym_sessions,
                         total_sessions=total_sessions,
                         strength_count=len(strength_sessions),
                         cardio_count=len(cardio_sessions),
                         flexibility_count=len(flexibility_sessions),
                         recent_chart_data=recent_chart_data,
                         chart_data=chart_data)

@app.route('/add_gym', methods=['GET', 'POST'])
@login_required
def add_gym():
    if request.method == 'POST':
        exercise_name = request.form.get('exercise_name')
        exercise_type = request.form.get('exercise_type')
        sets = request.form.get('sets')
        reps = request.form.get('reps')
        weight = request.form.get('weight')
        duration = request.form.get('duration')
        notes = request.form.get('notes')
        date = request.form.get('date')
        
        try:
            sets_int = int(sets) if sets else None
            weight_float = float(weight) if weight else None
            duration_int = int(duration) if duration else None
            session_date = datetime.strptime(date, '%Y-%m-%d').date() if date else datetime.now(timezone.utc).date()
            
            new_session = GymSession(
                user_id=current_user.id,
                exercise_name=exercise_name,
                exercise_type=exercise_type,
                sets=sets_int,
                reps=reps,
                weight=weight_float,
                duration=duration_int,
                notes=notes,
                date=session_date
            )
            
            db.session.add(new_session)
            db.session.commit()
            flash(f'Successfully logged {exercise_name} workout!', 'success')
            return redirect(url_for('gym'))
            
        except ValueError:
            flash('Please enter valid numbers for sets, weight, and duration.', 'error')
        except Exception as e:
            flash(f'Error adding gym session: {str(e)}', 'error')
    
    # Pass today's date to the template
    today = datetime.now(timezone.utc).date().strftime('%Y-%m-%d')
    return render_template('add_gym.html', today=today)

@app.route('/delete_gym/<int:session_id>')
@login_required
def delete_gym(session_id):
    session = GymSession.query.filter_by(id=session_id, user_id=current_user.id).first_or_404()
    try:
        db.session.delete(session)
        db.session.commit()
        flash('Gym session deleted successfully!', 'success')
    except Exception as e:
        flash(f'Error deleting gym session: {str(e)}', 'error')
    
    return redirect(url_for('gym'))

@app.route('/statistics')
@login_required
def statistics():
    runs = Run.query.filter_by(user_id=current_user.id).order_by(Run.date.asc()).all()
    
    if not runs:
        return render_template('statistics.html', stats={}, chart_data={})
    
    total_distance = sum(run.distance for run in runs)
    total_runs = len(runs)
    total_duration = sum(run.duration for run in runs if run.duration)
    
    # Calculate monthly statistics
    monthly_stats = {}
    for run in runs:
        month_key = f"{run.date.year}-{run.date.month:02d}"
        if month_key not in monthly_stats:
            monthly_stats[month_key] = {'distance': 0, 'runs': 0, 'durations': []}
        monthly_stats[month_key]['distance'] += run.distance
        monthly_stats[month_key]['runs'] += 1
        if run.duration:
            monthly_stats[month_key]['durations'].append(run.duration)
    
    # Prepare chart data
    chart_data = {
        'distance_over_time': {
            'dates': [run.date.strftime('%Y-%m-%d') for run in runs],
            'distances': [float(run.distance) for run in runs],
            'cumulative_distances': []
        },
        'monthly_averages': {
            'months': list(monthly_stats.keys()),
            'avg_distances': [],
            'avg_durations': []
        }
    }
    
    # Calculate cumulative distance
    cumulative = 0
    for distance in chart_data['distance_over_time']['distances']:
        cumulative += distance
        chart_data['distance_over_time']['cumulative_distances'].append(cumulative)
    
    # Calculate monthly averages
    for month_data in monthly_stats.values():
        chart_data['monthly_averages']['avg_distances'].append(
            round(month_data['distance'] / month_data['runs'], 2)
        )
        if month_data['durations']:
            avg_duration = sum(month_data['durations']) / len(month_data['durations'])
            chart_data['monthly_averages']['avg_durations'].append(round(avg_duration, 2))
        else:
            chart_data['monthly_averages']['avg_durations'].append(0)
    
    stats = {
        'total_distance': total_distance,
        'total_runs': total_runs,
        'avg_distance': round(total_distance / total_runs, 2) if total_runs > 0 else 0,
        'total_duration': total_duration,
        'avg_duration': round(total_duration / total_runs, 2) if total_runs > 0 and total_duration else 0,
        'monthly_stats': monthly_stats,
        'longest_run': max(runs, key=lambda x: x.distance) if runs else None,
        'fastest_pace': min([run for run in runs if run.pace], key=lambda x: x.pace) if any(run.pace for run in runs) else None
    }
    
    return render_template('statistics.html', stats=stats, chart_data=chart_data)

@app.route('/map', methods=['GET', 'POST'])
@login_required
def map_page():
    city = request.args.get('city', 'Polokwane')  # Default to Polokwane
    
    # Get city coordinates and gyms
    city_data = get_city_gyms(city)
    
    return render_template('map.html', 
                         gyms=city_data['gyms'], 
                         center=city_data['center'],
                         current_city=city)

def get_city_coordinates(city_name):
    """Get coordinates for a city using OpenWeatherMap geocoding API"""
    try:
        api_key = "600da4aa9340fc8d8b6093c528f3c95e"
        url = f"http://api.openweathermap.org/geo/1.0/direct?q={city_name}&limit=1&appid={api_key}"
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            if data:
                return {'lat': data[0]['lat'], 'lng': data[0]['lon']}
    except Exception as e:
        print(f"Error getting coordinates for {city_name}: {e}")
    
    # Default fallback coordinates (Polokwane)
    return {'lat': -23.9045, 'lng': 29.4689}

def get_city_gyms(city_name):
    """Get gym data for a specific city"""
    center = get_city_coordinates(city_name)
    
    # Gym database organized by city
    city_gyms = {
        'polokwane': [
            {
                'name': 'Virgin Active Polokwane',
                'address': 'Mall of the North, Polokwane',
                'lat': -23.8756,
                'lng': 29.4449,
                'phone': '015 297 4000',
                'type': 'Premium Gym',
                'amenities': ['Pool', 'Sauna', 'Group Classes', 'Personal Training']
            },
            {
                'name': 'Planet Fitness Polokwane',
                'address': 'Cycad Centre, Polokwane',
                'lat': -23.9089,
                'lng': 29.4567,
                'phone': '015 295 5555',
                'type': 'Budget Gym',
                'amenities': ['24/7 Access', 'Cardio Equipment', 'Weight Training']
            },
            {
                'name': 'Fitness First Polokwane',
                'address': 'Savannah Mall, Polokwane',
                'lat': -23.8923,
                'lng': 29.4712,
                'phone': '015 291 2000',
                'type': 'Fitness Center',
                'amenities': ['Group Classes', 'Personal Training', 'Functional Training']
            },
            {
                'name': 'Anytime Fitness Polokwane',
                'address': 'Bendor Park, Polokwane',
                'lat': -23.8845,
                'lng': 29.4634,
                'phone': '015 297 8888',
                'type': '24/7 Gym',
                'amenities': ['24/7 Access', 'Personal Training', 'Small Group Training']
            },
            {
                'name': 'Curves Polokwane',
                'address': 'Limpopo Mall, Polokwane',
                'lat': -23.9123,
                'lng': 29.4598,
                'phone': '015 295 7777',
                'type': 'Women Only',
                'amenities': ['Circuit Training', 'Women Only', 'Nutrition Coaching']
            },
            {
                'name': 'Iron Fitness Gym',
                'address': 'Ivy Park, Polokwane',
                'lat': -23.8967,
                'lng': 29.4723,
                'phone': '015 296 3333',
                'type': 'Bodybuilding Gym',
                'amenities': ['Heavy Weights', 'Powerlifting', 'Bodybuilding']
            },
            {
                'name': 'Fitness Express',
                'address': 'Westenburg, Polokwane',
                'lat': -23.9001,
                'lng': 29.4512,
                'phone': '015 297 1111',
                'type': 'Local Gym',
                'amenities': ['Affordable Rates', 'Basic Equipment', 'Friendly Staff']
            },
            {
                'name': 'CrossFit Polokwane',
                'address': 'Bendor, Polokwane',
                'lat': -23.8789,
                'lng': 29.4656,
                'phone': '015 298 9999',
                'type': 'CrossFit Box',
                'amenities': ['CrossFit Classes', 'Olympic Lifting', 'Functional Fitness']
            },
            {
                'name': 'Gym Company Polokwane',
                'address': 'Peter Mokaba Stadium, Polokwane',
                'lat': -23.8912,
                'lng': 29.4445,
                'phone': '015 297 2222',
                'type': 'Premium Gym',
                'amenities': ['Pool', 'Squash Courts', 'Group Classes', 'Personal Training', 'Steam Room']
            },
            {
                'name': 'Snap Fitness Polokwane',
                'address': 'Platinum Square, Polokwane',
                'lat': -23.8823,
                'lng': 29.4578,
                'phone': '015 296 7777',
                'type': '24/7 Gym',
                'amenities': ['24/7 Access', 'HIIT Classes', 'Personal Training', 'Modern Equipment']
            },
            {
                'name': 'Body Zone Gym',
                'address': 'Fauna Park, Polokwane',
                'lat': -23.9156,
                'lng': 29.4623,
                'phone': '015 295 4444',
                'type': 'Local Gym',
                'amenities': ['Boxing Ring', 'Functional Training', 'Weight Training', 'Cardio']
            },
            {
                'name': 'Ultimate Fitness',
                'address': 'Turfloop, Polokwane',
                'lat': -23.8834,
                'lng': 29.4712,
                'phone': '015 268 5555',
                'type': 'Student Gym',
                'amenities': ['Student Rates', 'Group Classes', 'Basic Equipment', 'Study Area']
            },
            {
                'name': 'Ladies Only Fitness',
                'address': 'Landdros Mare, Polokwane',
                'lat': -23.9045,
                'lng': 29.4534,
                'phone': '015 297 6666',
                'type': 'Women Only',
                'amenities': ['Women Only', 'Pilates', 'Yoga Classes', 'Childcare']
            },
            {
                'name': 'PowerHouse Gym',
                'address': 'Welgelegen, Polokwane',
                'lat': -23.8967,
                'lng': 29.4789,
                'phone': '015 296 8888',
                'type': 'Bodybuilding Gym',
                'amenities': ['Heavy Weights', 'Powerlifting Platform', 'Supplements', 'Posing Room']
            },
            {
                'name': 'FitZone Polokwane',
                'address': 'Seshego, Polokwane',
                'lat': -23.8456,
                'lng': 29.3789,
                'phone': '015 223 4444',
                'type': 'Local Gym',
                'amenities': ['Community Focused', 'Affordable', 'Group Classes', 'Weight Training']
            },
            {
                'name': 'Elite Fitness Centre',
                'address': 'Polokwane Central, Polokwane',
                'lat': -23.9012,
                'lng': 29.4567,
                'phone': '015 297 9999',
                'type': 'Premium Gym',
                'amenities': ['Luxury Facilities', 'Personal Training', 'Nutrition Bar', 'Recovery Zone']
            },
            {
                'name': 'Combat Fitness',
                'address': 'Annadale, Polokwane',
                'lat': -23.9123,
                'lng': 29.4234,
                'phone': '015 296 1111',
                'type': 'Martial Arts Gym',
                'amenities': ['MMA Classes', 'Boxing', 'Kickboxing', 'Self Defense']
            },
            {
                'name': 'Yoga & Wellness Studio',
                'address': 'Hammanskraal, Polokwane',
                'lat': -23.8876,
                'lng': 29.4123,
                'phone': '015 295 2222',
                'type': 'Yoga Studio',
                'amenities': ['Hot Yoga', 'Meditation', 'Wellness Classes', 'Holistic Health']
            },
            {
                'name': 'Sports Science Institute',
                'address': 'University of Limpopo, Polokwane',
                'lat': -23.8834,
                'lng': 29.4712,
                'phone': '015 268 7777',
                'type': 'Research Gym',
                'amenities': ['Sports Performance', 'Research Facilities', 'High-Tech Equipment', 'Student Access']
            }
        ],
        'johannesburg': [
            {
                'name': 'Virgin Active Sandton',
                'address': 'Sandton City, Johannesburg',
                'lat': -26.1076,
                'lng': 28.0567,
                'phone': '011 784 8000',
                'type': 'Premium Gym',
                'amenities': ['Pool', 'Sauna', 'Group Classes', 'Personal Training', 'Spa']
            },
            {
                'name': 'Planet Fitness Rosebank',
                'address': 'Rosebank Mall, Johannesburg',
                'lat': -26.1467,
                'lng': 28.0408,
                'phone': '011 447 9000',
                'type': 'Budget Gym',
                'amenities': ['24/7 Access', 'Cardio Equipment', 'Weight Training']
            },
            {
                'name': 'Anytime Fitness Melville',
                'address': 'Melville, Johannesburg',
                'lat': -26.1886,
                'lng': 28.0097,
                'phone': '011 482 1234',
                'type': '24/7 Gym',
                'amenities': ['24/7 Access', 'Personal Training', 'Small Group Training']
            },
            {
                'name': 'CrossFit Bryanston',
                'address': 'Bryanston, Johannesburg',
                'lat': -26.0689,
                'lng': 28.0206,
                'phone': '011 463 7777',
                'type': 'CrossFit Box',
                'amenities': ['CrossFit Classes', 'Olympic Lifting', 'Functional Fitness']
            },
            {
                'name': 'Virgin Active Hyde Park',
                'address': 'Hyde Park Corner, Johannesburg',
                'lat': -26.1234,
                'lng': 28.0398,
                'phone': '011 325 8000',
                'type': 'Premium Gym',
                'amenities': ['Pool', 'Tennis Courts', 'Spa', 'Group Classes', 'Personal Training']
            },
            {
                'name': 'Life Style Fitness',
                'address': 'Fourways Mall, Johannesburg',
                'lat': -26.0123,
                'lng': 28.0089,
                'phone': '011 465 1234',
                'type': 'Premium Gym',
                'amenities': ['Pool', 'Spinning Studio', 'Personal Training', 'Nutrition Consulting']
            },
            {
                'name': 'Planet Fitness Eastgate',
                'address': 'Eastgate Shopping Centre, Johannesburg',
                'lat': -26.2345,
                'lng': 28.1234,
                'phone': '011 823 5000',
                'type': 'Budget Gym',
                'amenities': ['24/7 Access', 'Cardio Equipment', 'Weight Training', 'Group Classes']
            },
            {
                'name': 'Fitness Xpress',
                'address': 'Randburg, Johannesburg',
                'lat': -26.0945,
                'lng': 28.0067,
                'phone': '011 789 3333',
                'type': 'Local Gym',
                'amenities': ['Boxing Classes', 'Functional Training', 'Weight Training', 'HIIT']
            },
            {
                'name': 'Gold\'s Gym Sandton',
                'address': 'Sandton CBD, Johannesburg',
                'lat': -26.1076,
                'lng': 28.0567,
                'phone': '011 784 9999',
                'type': 'Bodybuilding Gym',
                'amenities': ['Heavy Weights', 'Bodybuilding Focus', 'Supplements', 'Competition Prep']
            },
            {
                'name': 'F45 Parkhurst',
                'address': 'Parkhurst, Johannesburg',
                'lat': -26.1234,
                'lng': 28.0234,
                'phone': '011 447 5555',
                'type': 'Functional Training',
                'amenities': ['HIIT Classes', 'Functional Training', 'Heart Rate Monitoring', 'Nutrition Coaching']
            },
            {
                'name': 'Virgin Active Illovo',
                'address': 'Illovo Boulevard, Johannesburg',
                'lat': -26.1298,
                'lng': 28.0412,
                'phone': '011 268 8000',
                'type': 'Premium Gym',
                'amenities': ['Pool', 'Luxury Spa', 'Personal Training', 'Executive Classes']
            },
            {
                'name': 'CrossFit Johannesburg CBD',
                'address': 'Marshalltown, Johannesburg',
                'lat': -26.2034,
                'lng': 28.0412,
                'phone': '011 834 7777',
                'type': 'CrossFit Box',
                'amenities': ['CrossFit Classes', 'Urban Setting', 'Community Driven', 'Open Gym']
            },
            {
                'name': 'The Fitness Club',
                'address': 'Braamfontein, Johannesburg',
                'lat': -26.1923,
                'lng': 28.0345,
                'phone': '011 403 1234',
                'type': 'Budget Gym',
                'amenities': ['Student Friendly', '24/7 Access', 'Basic Equipment', 'Affordable']
            },
            {
                'name': 'BodyTech Midrand',
                'address': 'Midrand, Johannesburg',
                'lat': -25.9876,
                'lng': 28.1123,
                'phone': '011 315 9999',
                'type': 'Premium Gym',
                'amenities': ['Pool', 'Squash Courts', 'Tennis', 'Personal Training', 'Kids Club']
            },
            {
                'name': 'Combat Zone MMA',
                'address': 'Newtown, Johannesburg',
                'lat': -26.2034,
                'lng': 28.0345,
                'phone': '011 838 5555',
                'type': 'Martial Arts Gym',
                'amenities': ['MMA Training', 'Brazilian Jiu-Jitsu', 'Muay Thai', 'Boxing']
            },
            {
                'name': 'Yoga Flow Studio',
                'address': 'Greenside, Johannesburg',
                'lat': -26.1456,
                'lng': 28.0123,
                'phone': '011 486 7777',
                'type': 'Yoga Studio',
                'amenities': ['Vinyasa Flow', 'Hatha Yoga', 'Meditation', 'Wellness Workshops']
            },
            {
                'name': 'WITS Sport Centre',
                'address': 'University of Witwatersrand, Johannesburg',
                'lat': -26.1912,
                'lng': 28.0304,
                'phone': '011 717 1234',
                'type': 'University Gym',
                'amenities': ['Student Rates', 'Olympic Pool', 'Climbing Wall', 'Sports Science Lab']
            }
        ],
        'cape town': [
            {
                'name': 'Virgin Active V&A Waterfront',
                'address': 'V&A Waterfront, Cape Town',
                'lat': -33.9038,
                'lng': 18.4191,
                'phone': '021 419 0600',
                'type': 'Premium Gym',
                'amenities': ['Pool', 'Sauna', 'Group Classes', 'Personal Training', 'Ocean Views']
            },
            {
                'name': 'Planet Fitness Cavendish',
                'address': 'Cavendish Square, Cape Town',
                'lat': -33.9648,
                'lng': 18.4648,
                'phone': '021 674 8000',
                'type': 'Budget Gym',
                'amenities': ['24/7 Access', 'Cardio Equipment', 'Weight Training']
            },
            {
                'name': 'Anytime Fitness Green Point',
                'address': 'Green Point, Cape Town',
                'lat': -33.9069,
                'lng': 18.4105,
                'phone': '021 439 8888',
                'type': '24/7 Gym',
                'amenities': ['24/7 Access', 'Personal Training', 'Small Group Training']
            },
            {
                'name': 'Virgin Active Claremont',
                'address': 'Cavendish Square, Cape Town',
                'lat': -33.9798,
                'lng': 18.4698,
                'phone': '021 683 3000',
                'type': 'Premium Gym',
                'amenities': ['Pool', 'Squash Courts', 'Group Classes', 'Personal Training', 'Spa']
            },
            {
                'name': 'CrossFit Cape Town',
                'address': 'Woodstock, Cape Town',
                'lat': -33.9245,
                'lng': 18.4456,
                'phone': '021 448 7777',
                'type': 'CrossFit Box',
                'amenities': ['CrossFit Classes', 'Olympic Lifting', 'Open Gym', 'Nutrition Coaching']
            },
            {
                'name': 'Life Style Fitness Tygervalley',
                'address': 'Tygervalley Centre, Cape Town',
                'lat': -33.8456,
                'lng': 18.5789,
                'phone': '021 914 5000',
                'type': 'Premium Gym',
                'amenities': ['Pool', 'Tennis Courts', 'Personal Training', 'Pilates Studio']
            },
            {
                'name': 'Planet Fitness Century City',
                'address': 'Century City, Cape Town',
                'lat': -33.8912,
                'lng': 18.5123,
                'phone': '021 555 8000',
                'type': 'Budget Gym',
                'amenities': ['24/7 Access', 'Cardio Equipment', 'Weight Training', 'Spinning']
            },
            {
                'name': 'UCT Sports Centre',
                'address': 'University of Cape Town, Cape Town',
                'lat': -33.9567,
                'lng': 18.4612,
                'phone': '021 650 2151',
                'type': 'University Gym',
                'amenities': ['Student Rates', 'Pool', 'Climbing Wall', 'Sports Courts']
            },
            {
                'name': 'SSA Stellenbosch',
                'address': 'Stellenbosch, Cape Town',
                'lat': -33.9321,
                'lng': 18.8602,
                'phone': '021 883 9000',
                'type': 'Premium Gym',
                'amenities': ['Wine Country Views', 'Pool', 'Personal Training', 'Wellness Centre']
            },
            {
                'name': 'Virgin Active Constantia',
                'address': 'Constantia Village, Cape Town',
                'lat': -34.0234,
                'lng': 18.4234,
                'phone': '021 794 8000',
                'type': 'Premium Gym',
                'amenities': ['Pool', 'Tennis Courts', 'Spa', 'Mountain Views', 'Personal Training']
            },
            {
                'name': 'Planet Fitness Bellville',
                'address': 'Bellville Mall, Cape Town',
                'lat': -33.8765,
                'lng': 18.6345,
                'phone': '021 948 5000',
                'type': 'Budget Gym',
                'amenities': ['24/7 Access', 'Cardio Equipment', 'Weight Training', 'Group Classes']
            },
            {
                'name': 'CrossFit Observatory',
                'address': 'Observatory, Cape Town',
                'lat': -33.9402,
                'lng': 18.4678,
                'phone': '021 447 8888',
                'type': 'CrossFit Box',
                'amenities': ['CrossFit Classes', 'Olympic Lifting', 'Community Focused', 'Nutrition Coaching']
            },
            {
                'name': 'The Yoga Room',
                'address': 'Sea Point, Cape Town',
                'lat': -33.9189,
                'lng': 18.3876,
                'phone': '021 434 7777',
                'type': 'Yoga Studio',
                'amenities': ['Bikram Yoga', 'Vinyasa', 'Meditation', 'Wellness Retreats']
            },
            {
                'name': 'Combat Fitness Academy',
                'address': 'Parow, Cape Town',
                'lat': -33.9034,
                'lng': 18.5678,
                'phone': '021 939 5555',
                'type': 'Martial Arts Gym',
                'amenities': ['Karate', 'Taekwondo', 'Judo', 'Self Defense Classes']
            },
            {
                'name': 'F45 Camps Bay',
                'address': 'Camps Bay, Cape Town',
                'lat': -33.9512,
                'lng': 18.3767,
                'phone': '021 438 9999',
                'type': 'Functional Training',
                'amenities': ['HIIT Classes', 'Beach Views', 'Functional Training', 'Nutrition Coaching']
            },
            {
                'name': 'Cape Peninsula University Sports',
                'address': 'University of Cape Town, Cape Town',
                'lat': -33.9567,
                'lng': 18.4612,
                'phone': '021 650 3000',
                'type': 'University Gym',
                'amenities': ['Student Rates', 'Rock Climbing', 'Swimming Pool', 'Athletics Track']
            }
        ],
        'durban': [
            {
                'name': 'Virgin Active Gateway',
                'address': 'Gateway Theatre, Durban',
                'lat': -29.7263,
                'lng': 31.0672,
                'phone': '031 566 0000',
                'type': 'Premium Gym',
                'amenities': ['Pool', 'Sauna', 'Group Classes', 'Personal Training']
            },
            {
                'name': 'Planet Fitness Pavilion',
                'address': 'Pavilion Shopping Centre, Durban',
                'lat': -29.8175,
                'lng': 30.9796,
                'phone': '031 265 0000',
                'type': 'Budget Gym',
                'amenities': ['24/7 Access', 'Cardio Equipment', 'Weight Training']
            },
            {
                'name': 'Virgin Active Umhlanga',
                'address': 'Gateway Theatre, Durban',
                'lat': -29.7189,
                'lng': 31.0456,
                'phone': '031 566 1000',
                'type': 'Premium Gym',
                'amenities': ['Pool', 'Beachfront Views', 'Group Classes', 'Personal Training', 'Spa']
            },
            {
                'name': 'Life Style Fitness La Lucia',
                'address': 'La Lucia Mall, Durban',
                'lat': -29.7456,
                'lng': 31.0234,
                'phone': '031 572 8000',
                'type': 'Premium Gym',
                'amenities': ['Pool', 'Tennis Courts', 'Personal Training', 'Nutrition Consulting']
            },
            {
                'name': 'Anytime Fitness Westville',
                'address': 'Westville, Durban',
                'lat': -29.8234,
                'lng': 30.9345,
                'phone': '031 266 7777',
                'type': '24/7 Gym',
                'amenities': ['24/7 Access', 'Personal Training', 'Small Group Training', 'Modern Equipment']
            },
            {
                'name': 'CrossFit Durban',
                'address': 'Morningside, Durban',
                'lat': -29.8123,
                'lng': 31.0012,
                'phone': '031 312 5555',
                'type': 'CrossFit Box',
                'amenities': ['CrossFit Classes', 'Olympic Lifting', 'Functional Fitness', 'Beach Workouts']
            },
            {
                'name': 'UKZN Sports Centre',
                'address': 'University of KwaZulu-Natal, Durban',
                'lat': -29.8689,
                'lng': 30.9823,
                'phone': '031 260 1234',
                'type': 'University Gym',
                'amenities': ['Student Rates', 'Pool', 'Athletics Track', 'Sports Courts']
            },
            {
                'name': 'Virgin Active Florida Road',
                'address': 'Florida Road, Durban',
                'lat': -29.8345,
                'lng': 31.0123,
                'phone': '031 303 8000',
                'type': 'Premium Gym',
                'amenities': ['Pool', 'Squash Courts', 'Group Classes', 'Personal Training', 'Spa']
            },
            {
                'name': 'Planet Fitness Musgrave',
                'address': 'Musgrave Centre, Durban',
                'lat': -29.8434,
                'lng': 31.0345,
                'phone': '031 201 5000',
                'type': 'Budget Gym',
                'amenities': ['24/7 Access', 'Cardio Equipment', 'Weight Training', 'Modern Facilities']
            },
            {
                'name': 'CrossFit Durban North',
                'address': 'Durban North, Durban',
                'lat': -29.7789,
                'lng': 31.0567,
                'phone': '031 563 7777',
                'type': 'CrossFit Box',
                'amenities': ['CrossFit Classes', 'Olympic Lifting', 'Functional Fitness', 'Beach Training']
            },
            {
                'name': 'The Fitness Factory',
                'address': 'Pinetown, Durban',
                'lat': -29.8234,
                'lng': 30.8765,
                'phone': '031 701 1234',
                'type': 'Local Gym',
                'amenities': ['Community Gym', 'Affordable Rates', 'Group Classes', 'Weight Training']
            },
            {
                'name': 'Yoga & Pilates Studio',
                'address': 'Glenwood, Durban',
                'lat': -29.8567,
                'lng': 31.0234,
                'phone': '031 202 7777',
                'type': 'Yoga Studio',
                'amenities': ['Hatha Yoga', 'Pilates', 'Meditation', 'Wellness Classes']
            },
            {
                'name': 'Combat Sports Academy',
                'address': 'Overport, Durban',
                'lat': -29.9123,
                'lng': 30.9876,
                'phone': '031 207 5555',
                'type': 'Martial Arts Gym',
                'amenities': ['Boxing', 'Kickboxing', 'Mixed Martial Arts', 'Fitness Training']
            }
        ],
        'pretoria': [
            {
                'name': 'Virgin Active Menlyn',
                'address': 'Menlyn Park, Pretoria',
                'lat': -25.7842,
                'lng': 28.2775,
                'phone': '012 348 8000',
                'type': 'Premium Gym',
                'amenities': ['Pool', 'Sauna', 'Group Classes', 'Personal Training']
            },
            {
                'name': 'Planet Fitness Brooklyn',
                'address': 'Brooklyn Mall, Pretoria',
                'lat': -25.7615,
                'lng': 28.2362,
                'phone': '012 346 5000',
                'type': 'Budget Gym',
                'amenities': ['24/7 Access', 'Cardio Equipment', 'Weight Training']
            },
            {
                'name': 'Virgin Active Centurion',
                'address': 'Centurion Mall, Pretoria',
                'lat': -25.8567,
                'lng': 28.1890,
                'phone': '012 663 8000',
                'type': 'Premium Gym',
                'amenities': ['Pool', 'Squash Courts', 'Group Classes', 'Personal Training', 'Kids Club']
            },
            {
                'name': 'Life Style Fitness Hatfield',
                'address': 'Hatfield Plaza, Pretoria',
                'lat': -25.7489,
                'lng': 28.2345,
                'phone': '012 362 7000',
                'type': 'Premium Gym',
                'amenities': ['Pool', 'Student Friendly', 'Personal Training', 'Group Classes']
            },
            {
                'name': 'Anytime Fitness Waterkloof',
                'address': 'Waterkloof, Pretoria',
                'lat': -25.7834,
                'lng': 28.2456,
                'phone': '012 460 8888',
                'type': '24/7 Gym',
                'amenities': ['24/7 Access', 'Personal Training', 'Small Group Training', 'Executive Clientele']
            },
            {
                'name': 'UP Sport Centre',
                'address': 'University of Pretoria, Pretoria',
                'lat': -25.7545,
                'lng': 28.2314,
                'phone': '012 420 4555',
                'type': 'University Gym',
                'amenities': ['Student Rates', 'Olympic Pool', 'Athletics Stadium', 'High Performance Centre']
            },
            {
                'name': 'CrossFit Pretoria East',
                'address': 'Faerie Glen, Pretoria',
                'lat': -25.7678,
                'lng': 28.3123,
                'phone': '012 991 7777',
                'type': 'CrossFit Box',
                'amenities': ['CrossFit Classes', 'Olympic Lifting', 'Functional Fitness', 'Outdoor Training']
            },
            {
                'name': 'Virgin Active Brooklyn',
                'address': 'Brooklyn Bridge, Pretoria',
                'lat': -25.7654,
                'lng': 28.2345,
                'phone': '012 346 8000',
                'type': 'Premium Gym',
                'amenities': ['Pool', 'Squash Courts', 'Group Classes', 'Personal Training', 'Kids Club']
            },
            {
                'name': 'Planet Fitness Wonderboom',
                'address': 'Wonderboom Junction, Pretoria',
                'lat': -25.6789,
                'lng': 28.1987,
                'phone': '012 543 5000',
                'type': 'Budget Gym',
                'amenities': ['24/7 Access', 'Cardio Equipment', 'Weight Training', 'Group Classes']
            },
            {
                'name': 'The Wellness Centre',
                'address': 'Lynnwood, Pretoria',
                'lat': -25.7654,
                'lng': 28.2789,
                'phone': '012 348 7777',
                'type': 'Wellness Gym',
                'amenities': ['Yoga Classes', 'Pilates', 'Meditation', 'Holistic Health', 'Nutrition']
            },
            {
                'name': 'Combat Fitness Pretoria',
                'address': 'Silverton, Pretoria',
                'lat': -25.7345,
                'lng': 28.3123,
                'phone': '012 804 5555',
                'type': 'Martial Arts Gym',
                'amenities': ['MMA Training', 'Brazilian Jiu-Jitsu', 'Boxing', 'Self Defense']
            },
            {
                'name': 'F45 Pretoria North',
                'address': 'Pretoria North, Pretoria',
                'lat': -25.6789,
                'lng': 28.1567,
                'phone': '012 546 9999',
                'type': 'Functional Training',
                'amenities': ['HIIT Classes', 'Functional Training', 'Heart Rate Monitoring', 'Nutrition Coaching']
            },
            {
                'name': 'Tuks Sports Centre',
                'address': 'University of Pretoria, Pretoria',
                'lat': -25.7545,
                'lng': 28.2314,
                'phone': '012 420 6000',
                'type': 'University Gym',
                'amenities': ['Student Rates', 'High Performance Centre', 'Sports Science', 'Olympic Facilities']
            }
        ]
    }
    
    # Get gyms for the specified city (case insensitive)
    city_key = city_name.lower().strip()
    gyms = city_gyms.get(city_key, [])
    
    # If no gyms found for the city, try to create generic gym data
    if not gyms:
        gyms = create_generic_gyms(city_name, center)
    
    return {
        'center': center,
        'gyms': gyms
    }

def create_generic_gyms(city_name, center):
    """Create generic gym data for cities not in our database"""
    lat_offset = 0.01  # Small offset for gym locations
    lng_offset = 0.01
    
    generic_gyms = [
        {
            'name': f'Fitness Center {city_name}',
            'address': f'City Center, {city_name}',
            'lat': center['lat'] + lat_offset,
            'lng': center['lng'] + lng_offset,
            'phone': 'Contact Local Directory',
            'type': 'Local Gym',
            'amenities': ['Basic Equipment', 'Cardio', 'Weight Training']
        },
        {
            'name': f'{city_name} Gym',
            'address': f'Main Street, {city_name}',
            'lat': center['lat'] - lat_offset,
            'lng': center['lng'] - lng_offset,
            'phone': 'Contact Local Directory',
            'type': 'Budget Gym',
            'amenities': ['Affordable Rates', 'Basic Equipment']
        },
        {
            'name': f'24/7 Fitness {city_name}',
            'address': f'Commercial District, {city_name}',
            'lat': center['lat'] + lat_offset/2,
            'lng': center['lng'] - lng_offset/2,
            'phone': 'Contact Local Directory',
            'type': '24/7 Gym',
            'amenities': ['24/7 Access', 'Security', 'Modern Equipment']
        }
    ]
    
    return generic_gyms

@app.route('/ai-assistant')
@login_required
def ai_assistant():
    """AI Assistant page for fitness, diet, and mental health advice"""
    return render_template('ai_assistant.html')

@app.route('/ask-ai', methods=['POST'])
@login_required
def ask_ai():
    """Handle AI questions via API"""
    try:
        user_question = request.json.get('question', '').strip()
        
        if not user_question:
            return jsonify({'error': 'Please provide a question'}), 400
        
        # Create a fitness-focused prompt
        system_prompt = """You are a helpful fitness, nutrition, and mental health assistant. 
        Provide practical, evidence-based advice on diet, exercise, and mental wellness. 
        Keep responses concise but informative (2-3 paragraphs max). 
        Always encourage consulting healthcare professionals for serious concerns."""
        
        full_prompt = f"{system_prompt}\n\nUser question: {user_question}"
        
        # Generate response using Gemini
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(full_prompt)
        
        return jsonify({
            'response': response.text,
            'success': True
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Sorry, I encountered an error: {str(e)}',
            'success': False
        }), 500

# Initialize database
def create_tables():
    with app.app_context():
        db.create_all()

if __name__ == '__main__':
    create_tables()
    app.run(debug=True)