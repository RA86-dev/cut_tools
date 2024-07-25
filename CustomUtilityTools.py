import webbrowser
import secrets
import getpass
from Bio.Align import PairwiseAligner
import string
import smtplib
import yaml
from bs4 import BeautifulSoup
from ultralytics import YOLO
import scapy
import librosa
import librosa
import soundfile as sf
from Bio.Align import substitution_matrices
import numpy as np
from sklearn import datasets
import speech_recognition as sr
from gtts import gTTS 
import nmap
import cryptography
from cryptography.fernet import Fernet
import yagmail
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import zlib
import gzip
import bz2
import lzma
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
import librosa.display
import numpy as np
import matplotlib.pyplot
import string,json
import Bio
import random
import cv2
from Bio import SeqIO, Entrez, AlignIO, Phylo
from Bio.Seq import Seq
from Bio.SeqUtils import  molecular_weight, ProtParam
from Bio.Blast import NCBIWWW, NCBIXML
from Bio.PDB import PDBList, PDBParser
from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor
import matplotlib.pyplot as plt
import io
import openai
import os
import ffmpeg
import nltk
import PyPDF2
import random
from Bio import SeqIO
from Bio.Seq import Seq
import pygame
import numpy as np
from web3 import Web3
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC
import numpy as nps
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from urllib.parse import quote
import io
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import yt_dlp as ytDLP
import numpy as np
from PIL import Image, ImageFilter
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from heapq import nlargest
import geocoder
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding
import os
import base64
import requests
import platform
from gtts import gTTS
import mpmath as mp
from deep_translator import GoogleTranslator
import cpuinfo
from fastapi import FastAPI
import math
import requests
import socket
from typing import Dict, List, Optional
from urllib.parse import urlencode
import psutil
import subprocess
from pathlib import Path
import os
import socketserver
from fastapi.responses import HTMLResponse
import http
import statistics
import subprocess
import hashlib
from PIL import Image
import pytz
import geopy.distance
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import sqlite3
import logging
import cProfile
import nltk
import spacy
from textblob import TextBlob
from gensim.summarization import summarize
from gensim.models.ldamodel import LdaModel
from gensim.corpora import Dictionary
import pytesseract
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from fbprophet import Prophet
from stable_baselines3 import PPO
import networkx as nx
from pyspark.sql import SparkSession
import dask.dataframe as dd
from lime.lime_text import LimeTextExplainer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense, LSTM
import tensorflow as tf
from pennylane import qml
from diffprivlib import models as dp_models
from federated_learning_utils import FederatedLearning
def clear():
    os.system('clear')
class CustomUtilityTools:
    def __init__(self,AppName=False):
        print('Finishing Compiling...')
        print('Generating Key Identifier')
        if AppName:
            self.appname = AppName
        else:
            self.appname = None
     
    
    class Servertools():
        def __init__(self):
            pass

        @staticmethod
        def get_ip():
            IPaddr = socket.gethostbyname(socket.gethostname())
            return IPaddr
        @staticmethod
        def get_hostname():
            hostname = socket.gethostname()
            return hostname

        @staticmethod
        def create_app_serverAPI():
            app = FastAPI()
            return app

        @staticmethod
        def add_html_endpoint(app, html_file_path,endpoint_path='/'):
            @app.get(endpoint_path)
            async def read_html():
                with open(html_file_path, 'r') as file:
                    return HTMLResponse(content=file.read(), status_code=200)

        @staticmethod
        def start_simple_http_server(port=8000):
            """
            Starts a simple HTTP server to serve files from the current directory.

            Args:
                port (int): Port number to start the server on. Default is 8000.
            """
            handler = http.server.SimpleHTTPRequestHandler
            httpd = socketserver.TCPServer(("", port), handler)
            print(f"Serving HTTP on port {port} (http://{socket.gethostbyname(socket.gethostbyname)}:{port}/)...")
            httpd.serve_forever()

        @staticmethod
        def check_server_status(url):
            """
            Checks if a server is running at the given URL.

            Args:
                url (str): The URL of the server to check.

            Returns:
                bool: True if the server is running, False otherwise.
            """
            try:
                response = requests.get(url)
                return response.status_code == 200
            except requests.ConnectionError:
                return False

    class PCINFOUtils():
        def __init__(self):
            pass
        @staticmethod
        def get_logs():
            logs = {}
            if psutil.MACOS:
                logs = {}
                macOS_log_path = '/var/log/system.log'
                if os.path.exists(macOS_log_path):
                    with open(macOS_log_path,'r') as file:
                        logs['System Logs'] = file.read()
                install_log = '/var/log/install.log'
                if os.path.exists(install_log):
                    with open(install_log,'r') as file:
                        logs['Install Logs'] = file.read()
                diag_reports_path = '/Library/Logs/DiagnosticReports/'
                if os.path.exists(diag_reports_path):
                    logs['Diagnostic Reports'] = []
                    for filename in os.listdir(diag_reports_path):
                        if filename.endswith('.crash') or filename.endswith('.diagnostic'):
                            with open(Path(diag_reports_path) / filename, 'r') as file:
                                logs['Diagnostic Reports'].append({filename: file.read()})
                return logs
            elif psutil.LINUX:
                logs = {}

                # System Log
                syslog_path = '/var/log/syslog'
                if os.path.exists(syslog_path):
                    with open(syslog_path, 'r') as file:
                        logs['Syslog'] = file.read()

                # Kernel Log
                kern_log_path = '/var/log/kern.log'
                if os.path.exists(kern_log_path):
                    with open(kern_log_path, 'r') as file:
                        logs['Kernel Log'] = file.read()

                # Authentication Log
                auth_log_path = '/var/log/auth.log'
                if os.path.exists(auth_log_path):
                    with open(auth_log_path, 'r') as file:
                        logs['Authentication Log'] = file.read()

                return logs
            elif psutil.WINDOWS:

                # Event Logs
                event_logs = ['Application', 'System', 'Security']
                for log_name in event_logs:
                    try:
                        result = subprocess.run(['powershell', '-Command', f'Get-EventLog -LogName {log_name} -Newest 10'], capture_output=True, text=True)
                        logs[f'{log_name} Log'] = result.stdout
                    except Exception as e:
                        logs[f'{log_name} Log'] = f"Failed to retrieve {log_name} log: {str(e)}"

                # CBS Log
                cbs_log_path = r'C:\Windows\Logs\CBS\CBS.log'
                if os.path.exists(cbs_log_path):
                    with open(cbs_log_path, 'r') as file:
                        logs['CBS Log'] = file.read()

                return logs
        @staticmethod
        def get_cpuInfo():
            return cpuinfo.get_cpu_info_json()
        @staticmethod
        def get_ram_information():
            information = {

            }
            information['used'] = psutil.virtual_memory().used
            information['avaliable'] = psutil.virtual_memory().available
            information['total'] = psutil.virtual_memory().total
            information['percentage'] = psutil.virtual_memory().percent
            return information
        @staticmethod
        def get_swap():
            info = {}
            info['Used'] = psutil.swap_memory().used
            info['Avaliable'] = psutil.swap_memory().free
            info['Total'] = psutil.swap_memory().total
            info['Percentage'] = psutil.swap_memory().percent
            return info

    class WeatherTools:
        def __init__(self):
            pass

        @staticmethod
        def wmo4677_json() -> Dict[int, str]:
            """
            Returns a dictionary of WMO4677 weather codes and descriptions.
            """
            return  {0: "Cloud development not observed or not observable.",1: "Clouds generally dissolving or becoming less developed",2: "State of sky on the whole unchanged",3: "Clouds generally forming or developing",4: "Visibility reduced by smoke, e.g., veldt or forest fires, industrial smoke or volcanic ashes",5: "Haze",6: "Widespread dust in suspension in the air, not raised by wind at or near the station at the time of observation",7: "Dust or sand raised by wind at or near the station at the time of observation, but no well developed dust whirl(s), and no duststorm or sandstorm seen.",8: "Well developed dust whirl(s) or sand whirl(s) seen at or near the station during the preceding hour or at the time of observation, but no duststorm or sandstorm.",9: "Duststorm or sandstorm within sight at the time of observation, or at the station during the preceding hour.",10: "Mist",11: "Patches of fog",12: "More or less continuous fog",13: "Lightning visible, no thunder heard",14: "Precipitation within sight, not reaching the ground or the surface of the sea.",15: "Precipitation within sight, reaching the ground or the surface of the sea, but distant, i.e., estimated to be more than 5km from the station",16: "Precipitation within sight, reaching the ground or the surface of the sea, near to, but not at the station",17: "Thunderstorm, but no precipitation at the time of observation",18: "Squalls at or within sight of the station during the preceding hour or at the time of observation",19: "Funnel clouds at or within sight of the station during the preceding hour or at the time of observation.",20: "Drizzle (not freezing) or snow grains not falling as shower(s)",21: "Rain (not freezing) not falling as shower(s)",22: "Snow",23: "Rain and snow or ice pellets",24: "Freezing drizzle or freezing rain",25: "Shower(s) of rain",26: "Shower(s) of snow, or of rain and snow",27: "Shower(s) of hail, or of rain and hail",28: "Fog or ice fog",29: "Thunderstorm (with or without precipitation)",30: "Slight or moderate duststorm or sandstorm (Has decreased during the preceding hour)",31: "Slight or moderate duststorm or sandstorm (No appreciable change during the preceding hour)",32: "Slight or moderate duststorm or sandstorm (Has begun or has increased during the preceding hour)",33: "Severe duststorm or sandstorm (Has decreased during the preceding hour)",34: "Severe duststorm or sandstorm (No appreciable change during the preceding hour)",35: "Severe duststorm or sandstorm (Has begun or has increased during the preceding hour)",36: "Slight or moderate blowing snow (Generally low, below eye level)",37: "Heavy drifting snow (Generally low, below eye level)",38: "Slight or moderate blowing snow (Generally high, above eye level)",39: "Heavy drifting snow (Generally high, above eye level)",40: "Fog at a distance",41: "Fog in patches",42: "Fog in the vicinity",43: "Fog, sky visible",44: "Fog, sky not visible",45: "Fog, depositing rime",46: "Fog, partial",47: "Fog, with smoke",48: "Fog, visibility reducing",49: "Fog, no change in visibility",50: "Drizzle, not freezing, itermittent (Slight at the time of observation)",51:"Drizzle, not freezing, continuous, slight at the time of observation",52:"Drizzle, not freezing, intermittent (Moderate at the time of observation)",53:"Drizzle, not freezing, continuous (Moderate at the time of observation)",54:"Drizzle, not freezing, intermittent (Heavy at the time of observation)",55:"Drizzle, not freezing, continuous (Heavy at the time of observation)",56: "Rain, freezing, slight",57: "Rain, not freezing, moderate or heavy (dence)",58:"Drizzle and rain, slight",59:"Drizzle and rain, moderate or heavy",60:"Rain,not freezing, intermittent (slight at the time of observation)",61:"Rain,not freezing, continuous (slight at the time of observation)",62:"Rain,not freezing, intermittent (Moderate at the time of observation)",63:"Rain,not freezing, continuous (Moderate at the time of observation)",64:"Rain,not freezing, intermittent (Heavy at the time of observation)",65:"Rain,not freezing, continuous (Heavy at the time of observation)",66:"Rain,freezing, slight",67:"Rain, freezing,moderate or heavy (dence)",68:"Rain or drizzle and snow, slight",69:"Rain or drizzle and snow, moderate or heavy",70:"Intermittent fall of snowflakes (slight at the time of observation)",71:"Continuous fall of snowflakes (slight at the time of observation)",72:"Intermittent fall of snowflakes (moderate at the time of observation)",73:"Continuous fall of snowflakes (moderate at the time of observation)",74:"Intermittent fall of snowflakes (heavy at the time of observation)",75:"Continuous fall of snowflakes (heavy at the time of observation)",76:"Diamond Dust (with or without fog)",77:"Snow grains (with or without fog)",78:"Isolated star-like snow crystals (with or without fog)",79:"Ice pellets",80:"Rain shower(s), slight",81:"Rain shower(s),moderate or heavy",82:"Rain shower(s), violent",83:"Shower(s) of rain and snow mixed,slight",84:"Shower(s) of rain and snow mixed, heavy or moderate",85:"Snow shower(s),slight",86:"Snow shower(s),moderate or heavy",87:"Shower(s) of snow pellets or smail hail, with or without rain or rain and snow mixed (slight)",88:"Shower(s) of snow pellets or smail hail, with or without rain or rain and snow mixed (moderate or heavy)",89:"Shower(s) of hail,with or without rain or rain and snow mixed,not associated with tunder (slight)",90:"Shower(s) of hail,with or without rain or rain and snow mixed,not associated with tunder (moderate or heavy)",91:"Slight rain at time of observation (Thunderstorm during the preceding hour but not at time of observation)",92:"Moderate or heavy rain at time of observation (Thunderstorm during the preceding hour but not at time of observation)",93:"Slight snow, or rain and snow mixed or hail - at time of observation (Thunderstorm during the preceding hour but not at time of observation)",94:"Moderate or heavy snow, or rain and snow mixed or hail - at time of observation (Thunderstorm during the preceding hour but not at time of observation)",95:"Thunderstorm, slight or moderate, without hail",96:"Thunderstorm, slight or moderate, with hail",97:"Thunderstorm, heavy, without hail - but with rain and/or snow at time of observation",98:"Thunderstorm combined with duststorm or sandstorm at time of observation",99:"Thunderstorm, heavy, with hail at time of observation"}

        @staticmethod
        def fetch_openweathermap_api(api_key: str, lat: Optional[float] = None, lon: Optional[float] = None, city: Optional[str] = None) -> Dict:
            """
            Fetches weather data from OpenWeatherMap API.

            Args:
                api_key (str): OpenWeatherMap API key
                lat (float, optional): Latitude
                lon (float, optional): Longitude
                city (str, optional): City name

            Returns:
                Dict: JSON response from the API
            """
            base_url = "http://api.openweathermap.org/data/2.5/weather"
            base_params = {
                "appid": api_key,
                "units": "metric"
            }

            if city:
                base_params['q'] = city
            elif lat is not None and lon is not None:
                base_params['lat'] = lat
                base_params['lon'] = lon
            else:
                return {"error": "Please provide city name or latitude and longitude"}

            try:
                response = requests.get(base_url, params=base_params)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                return {"error": str(e)}

        @staticmethod
        def fetch_open_meteo(lat: float, lon: float, api_key: Optional[str] = None,
                             params: Dict[str, Optional[List[str]]] = None) -> Dict:
            """
            Fetches weather data from Open-Meteo API.

            Args:
                lat (float): Latitude
                lon (float): Longitude
                api_key (str, optional): API key for commercial use
                params (Dict[str, Optional[List[str]]], optional): Parameters for the API request

            Returns:
                Dict: JSON response from the API
            """
            if params is None:
                params = {}

            base_url = "https://api.open-meteo.com/v1/forecast" if api_key is None else "https://customer-api.open-meteo.com/v1/forecast"

            query_params = {
                "latitude": lat,
                "longitude": lon
            }

            if api_key:
                query_params["api_key"] = api_key

            for param_type in ["current", "hourly", "daily", "minutely_15"]:
                if param_type in params and params[param_type]:
                    query_params[param_type] = ",".join(params[param_type])

            url = f"{base_url}?{urlencode(query_params)}"

            try:
                response = requests.get(url)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                return {"error": str(e)}
    class ConversionTools():
        def __init__(self):
            pass
        @staticmethod
        def convert_CtoF(celsius:float) -> float:
            return (celsius * (9/5)) + 32
        @staticmethod
        def convert_FtoC(fahrenheit:float) -> float:
            return (fahrenheit -32) * (5/9)
        @staticmethod
        def metric_conversion(value:float, from_unit:str, to_unit:str):

            # Conversion factors for length, mass, volume, and temperature
            conversion_factors = {
                'length': {
                    'mm': 0.001,
                    'cm': 0.01,
                    'm': 1,
                    'km': 1000,
                },
                'mass': {
                    'mg': 1e-6,
                    'g': 0.001,
                    'kg': 1,
                    't': 1000,
                },
                'volume': {
                    'mL': 1e-6,
                    'L': 0.001,
                    'm^3': 1,
                },
                'temperature': {
                    'C': 'C',
                    'F': 'F',
                    'K': 'K',
                }
            }

            # Function for temperature conversion
            def convert_temperature(value, from_unit, to_unit):
                if from_unit == to_unit:
                    return value

                if from_unit == 'C':
                    if to_unit == 'F':
                        return (value * 9/5) + 32
                    elif to_unit == 'K':
                        return value + 273.15
                elif from_unit == 'F':
                    if to_unit == 'C':
                        return (value - 32) * 5/9
                    elif to_unit == 'K':
                        return ((value - 32) * 5/9) + 273.15
                elif from_unit == 'K':
                    if to_unit == 'C':
                        return value - 273.15
                    elif to_unit == 'F':
                        return ((value - 273.15) * 9/5) + 32
                return None

            # Check if the units belong to the same category
            def same_category(unit1, unit2):
                for category, units in conversion_factors.items():
                    if unit1 in units and unit2 in units:
                        return category
                return None

            # Convert units
            category = same_category(from_unit, to_unit)

            if category:
                if category == 'temperature':
                    return convert_temperature(value, from_unit, to_unit)
                else:
                    factor_from = conversion_factors[category][from_unit]
                    factor_to = conversion_factors[category][to_unit]
                    return value * (factor_from / factor_to)
            else:
                raise ValueError(f"Conversion from {from_unit} to {to_unit} is not supported.")
        @staticmethod
        def imperial_conversion(value, from_unit, to_unit):
            """
            Convert between Imperial units and Metric units.

            Parameters:
            - value (float): The value to be converted.
            - from_unit (str): The unit to convert from (e.g., 'in', 'ft', 'lb', 'gal', 'F').
            - to_unit (str): The unit to convert to (e.g., 'in', 'ft', 'lb', 'gal', 'F').

            Returns:
            - float: The converted value.
            """

            # Conversion factors between Imperial and Metric units
            conversion_factors = {
                'length': {
                    'in': 0.0254,
                    'ft': 0.3048,
                    'yd': 0.9144,
                    'mi': 1609.34,
                    'm': 1,
                    'cm': 0.01,
                    'km': 1000,
                    'mm': 0.001,
                },
                'weight': {
                    'oz': 0.0283495,
                    'lb': 0.453592,
                    'st': 6.35029,
                    'kg': 1,
                    'g': 0.001,
                    'mg': 1e-6,
                    't': 1000,
                },
                'volume': {
                    'fl_oz': 0.0295735,
                    'cup': 0.236588,
                    'pt': 0.473176,
                    'qt': 0.946353,
                    'gal': 3.78541,
                    'L': 1,
                    'mL': 0.001,
                    'm^3': 1000,
                },
                'temperature': {
                    'F': 'F',
                    'C': 'C',
                    'K': 'K',
                }
            }

            # Function for temperature conversion
            def convert_temperature(value, from_unit, to_unit):
                if from_unit == to_unit:
                    return value

                if from_unit == 'C':
                    if to_unit == 'F':
                        return (value * 9/5) + 32
                    elif to_unit == 'K':
                        return value + 273.15
                elif from_unit == 'F':
                    if to_unit == 'C':
                        return (value - 32) * 5/9
                    elif to_unit == 'K':
                        return ((value - 32) * 5/9) + 273.15
                elif from_unit == 'K':
                    if to_unit == 'C':
                        return value - 273.15
                    elif to_unit == 'F':
                        return ((value - 273.15) * 9/5) + 32
                return None

            # Check if the units belong to the same category
            def same_category(unit1, unit2):
                for category, units in conversion_factors.items():
                    if unit1 in units and unit2 in units:
                        return category
                return None

            # Convert units
            category = same_category(from_unit, to_unit)

            if category:
                if category == 'temperature':
                    return convert_temperature(value, from_unit, to_unit)
                else:
                    factor_from = conversion_factors[category][from_unit]
                    factor_to = conversion_factors[category][to_unit]
                    return value * (factor_from / factor_to)
            else:
                raise ValueError(f"Conversion from {from_unit} to {to_unit} is not supported.")
    class UtilTools():
        def __init__(self):
            pass
        @staticmethod
        def run_terminal(command):

            s = subprocess.run(command)
            return s.stdout,s.stderr
        @staticmethod
        def get_request_info_json(url,params):
            response = requests.get(url=url,params=params)
            if response.status_code == 200:
                try:
                    output = response.json()
                except Exception as e:
                    output = response.text()
            else:
                return {'error':'UtilTools Handler could not fetch information.'}
        @staticmethod
        def translate(start_text, end_lang):
            try:
                translator = GoogleTranslator(source='auto', target=end_lang)
                return translator.translate(str(start_text))
            except Exception as e:
                return f"Translation error: {str(e)}"
        @staticmethod
        def TextToSpeech(fp,text):
            gtts = gTTS(text)
            gtts.save(str(fp))
            return True
    class MathTools():
        def __init__(self):
            pass
        @staticmethod
        def pythagorean_triplets(limit):
            """Generates Pythagorean triplets up to a given limit.

            Args:
                limit: The maximum value for any side of the triangle.

            Returns:
                A list of tuples representing Pythagorean triplets.
            """

            triplets = []

            for m in range(2, int(limit**0.5) + 1):
                for n in range(1, m):
                    a = m**2 - n**2
                    b = 2 * m * n
                    c = m**2 + n**2

                    if c > limit:
                        break

                    triplets.append((a, b, c))

            return triplets
        @staticmethod
        def pythagorean_theorem(a_leg,b_leg):
            n1 = a_leg^2
            n2 = b_leg^2
            t0 = n1 + n2
            result = math.sqrt(t0)
            return result
        
        @staticmethod
        def find_missing_side(number_of_sides, sides_angle):
            if number_of_sides < 3:
                return "Polygons must have at least 3 sides"

            if len(sides_angle) != number_of_sides - 1:
                return f"For a {number_of_sides}-sided polygon, please provide {number_of_sides - 1} side-angle pairs"

            known_angles_sum = sum(angle for _, angle in sides_angle if angle is not None)

            missing_angle = (number_of_sides - 2) * 180 - known_angles_sum
            missing_angle_rad = math.radians(missing_angle)

            reference_side, reference_angle = next((side, angle) for side, angle in sides_angle if side is not None and angle is not None)
            reference_angle_rad = math.radians(reference_angle)

            missing_side = (reference_side * math.sin(missing_angle_rad)) / math.sin(reference_angle_rad)

            return round(missing_side, 2)
        @staticmethod
        def generate_pi(number_of_digits):
            mp.dps = number_of_digits + 1
            pi = mp.pi
            return str(pi)
        @staticmethod
        def radians(angle):
            """
            Converts degrees to
            RADIANS.
            """
            return math.radians(angle)
        @staticmethod
        def get_interior_angles(number_of_sides):
            e1 = number_of_sides - 2
            e2 = (e1 * 180) / number_of_sides
            return e2

    class DataProcessingTools:
        @staticmethod
        def basic_statistics(data):
            return {
                "mean": statistics.mean(data),
                "median": statistics.median(data),
                "mode": statistics.mode(data),
                "std_dev": statistics.stdev(data)
            }

        @staticmethod
        def filter_data(data, condition):
            return [item for item in data if condition(item)]

        @staticmethod
        def sort_data(data, key=None, reverse=False):
            return sorted(data, key=key, reverse=reverse)

    class NetworkTools:
        @staticmethod
        def ping(host):
            param = '-n' if os.name == 'nt' else '-c'
            command = ['ping', param, '1', host]
            return subprocess.call(command) == 0

        @staticmethod
        def traceroute(host):
            param = '-n' if os.name == 'nt' else '-c'
            command = ['tracert' if os.name == 'nt' else 'traceroute', param, '1', host]
            return subprocess.check_output(command).decode()
        @staticmethod

        def wifi_list():
            os_name = platform.system()
            try:
                if os_name == "Windows":
                    list_networks_command = 'netsh wlan show networks'
                    output = subprocess.check_output(list_networks_command, shell=True, text=True)
                elif os_name == "Linux":
                    list_networks_command = "nmcli device wifi list"
                    output = subprocess.check_output(list_networks_command, shell=True, text=True)
                elif os_name == "Darwin":  # macOS
                    list_networks_command = "/System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport -s"
                    output = subprocess.check_output(list_networks_command, shell=True, text=True)
                else:
                    return "Unsupported operating system"
                return output
            except subprocess.CalledProcessError:
                return "Failed to retrieve Wi-Fi networks. Make sure you have the necessary permissions and Wi-Fi is turned on."
            except FileNotFoundError:
                return "Command not found. Make sure you have the necessary tools installed."
    class CryptographyTools:
        @staticmethod
        def hash_string(string, algorithm='sha256'):
            return hashlib.new(algorithm, string.encode()).hexdigest()
        def encrypt_string(self, text, encryption_method="fernet"):
            enc_type= encryption_method.lower()
            if enc_type == "fernet":
                key = Fernet.generate_key()
                f = Fernet(key)
                encrypted_text = f.encrypt(text.encode('utf-8'))
                return encrypted_text, key
            elif enc_type == "b64":
                return base64.b64encode(text)
            elif enc_type == "a85":
                return base64.a85encode(text)
            elif enc_type == "b32":
                return base64.b32encode(text)
            elif enc_type == "b16":
                return base64.b16encode(text)
            else:
                raise ValueError("Unsupported encryption method")

        def decrypt_string(self, token, key=None, encryption_method="fernet"):
            enc_type = encryption_method.lower()
            if enc_type == "fernet":
                f = Fernet(key)
                decrypted_text = f.decrypt(token).decode('utf-8')
                return decrypted_text
            elif enc_type == "b64":
                return base64.b64decode(token)
            elif enc_type == "a85":
                return base64.a85decode(token)
            elif enc_type == "b32":
                return base64.b32decode(token)
            elif enc_type == "b16":
                return base64.b16decode(token)
            else:
                raise ValueError("Unsupported encryption method")
        def encrypt_file(self, filename, encryption_method="fernet"):
            with open(filename,'r') as file:
                text = file.read()
            enc_type= encryption_method.lower()
            if enc_type == "fernet":
                key = Fernet.generate_key()
                f = Fernet(key)
                encrypted_text = f.encrypt(text.encode('utf-8'))
                return encrypted_text, key
            elif enc_type == "b64":
                return base64.b64encode(text)
            elif enc_type == "a85":
                return base64.a85encode(text)
            elif enc_type == "b32":
                return base64.b32encode(text)
            elif enc_type == "b16":
                return base64.b16encode(text)
            else:
                raise ValueError("Unsupported encryption method")

        def decrypt_string(self,filename, key=None, encryption_method="fernet"):
            with open(filename,'r') as file:
                token = file.read()
            enc_type = encryption_method.lower()
            if enc_type == "fernet":
                f = Fernet(key)
                decrypted_text = f.decrypt(token).decode('utf-8')
                return decrypted_text
            elif enc_type == "b64":
                return base64.b64decode(token)
            elif enc_type == "a85":
                return base64.a85decode(token)
            elif enc_type == "b32":
                return base64.b32decode(token)
            elif enc_type == "b16":
                return base64.b16decode(token)
            
            else:
                raise ValueError("Unsupported encryption method")

        @staticmethod
        def simple_caesar_cipher(text, shift):
            result = ""
            for char in text:
                if char.isalpha():
                    ascii_offset = 65 if char.isupper() else 97
                    result += chr((ord(char) - ascii_offset + shift) % 26 + ascii_offset)
                else:
                    result += char
            return result
        @staticmethod
        def generate_random_key(l=128):
            length = l - 10
            def generate_chart(width=500, height=500):
                chart = []
                for y in range(height):
                    row = []
                    for x in range(width):
                        choice = random.choice(['number', 'char', 'char_low', 'hex','oct','special'])
                        if choice == 'number':
                            v = random.randint(0, 1000)
                            row.append(v)  # Random number between 0 and 1000
                        elif choice == "char":
                            v = random.choice(string.ascii_uppercase)
                            row.append(v)  # Random uppercase letter
                        elif choice == "char_low":
                            v = random.choice(string.ascii_lowercase)
                            row.append(v)  # Random lowercase letter
                        elif choice == "hex":
                            v = random.choice(string.hexdigits)
                            row.append(v)  # Random hex digit
                        elif choice == "oct":
                            v = random.choice(string.octdigits)
                            row.append(v)
                        elif choice == "special":
                            v = random.choice(string.punctuation)
                            row.append(v)  # Random special character

                    chart.append(row)
                return chart

            c = ""
            for i in tqdm(range(length), desc="Generating key"):
                chart = generate_chart()
                y = secrets.randbelow(500)
                x = secrets.randbelow(500)
                char = chart[y][x]
                c += str(char)
            seed = secrets.token_bytes(250)
            hash_obj = hashlib.sha256(seed)
            key = hash_obj.hexdigest()[:length]
            c += str(key)
            return c

            
        @staticmethod
        def aes_encode(data,key):
            key = key.encode('utf-8')
            key = key.ljust(32,b'\0')[:32]
            iv = os.urandom(16)
            cipher = Cipher(algorithms.AES(key),modes.CBC(iv),backend=default_backend())
            encryptor = cipher.encryptor()
            padder = padding.PKCS7(128).padder()
            padded_data = padder.update(data.encode('utf-8')) + padder.finalize()
            encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
            return base64.b64encode(iv + encrypted_data).decode('utf-8')
        @staticmethod
        def aes_decode(encrypted_data, key):
            encrypted_data = base64.b64decode(encrypted_data)

            key = key.encode('utf-8')
            key = key.ljust(32, b'\0')[:32]

            iv = encrypted_data[:16]
            ciphertext = encrypted_data[16:]

            cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
            decryptor = cipher.decryptor()

            decrypted_padded_data = decryptor.update(ciphertext) + decryptor.finalize()


            unpadder = padding.PKCS7(128).unpadder()
            decrypted_data = unpadder.update(decrypted_padded_data) + unpadder.finalize()

            return decrypted_data.decode('utf-8')
        @staticmethod
        def aes_encode_file(filename,key):
            with open(filename,'r') as file:
                data = file.read()

            key = key.encode('utf-8')
            key = key.ljust(32,b'\0')[:32]
            iv = os.urandom(16)
            cipher = Cipher(algorithms.AES(key),modes.CBC(iv),backend=default_backend())
            encryptor = cipher.encryptor()
            padder = padding.PKCS7(128).padder()
            padded_data = padder.update(data.encode('utf-8')) + padder.finalize()
            encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
            return base64.b64encode(iv + encrypted_data).decode('utf-8')
        @staticmethod
        def aes_decode_file(filename, key):
            with open(filename,'r') as file:
                encrypted_data = file.read()
            encrypted_data = base64.b64decode(encrypted_data)

            key = key.encode('utf-8')
            key = key.ljust(32, b'\0')[:32]

            iv = encrypted_data[:16]
            ciphertext = encrypted_data[16:]

            cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
            decryptor = cipher.decryptor()

            decrypted_padded_data = decryptor.update(ciphertext) + decryptor.finalize()


            unpadder = padding.PKCS7(128).unpadder()
            decrypted_data = unpadder.update(decrypted_padded_data) + unpadder.finalize()

            return decrypted_data.decode('utf-8')
    class ImageProcessingTools:
        @staticmethod
        def resize_image(image_path, output_path, size):
            with Image.open(image_path) as img:
                img.thumbnail(size)
                img.save(output_path)

        @staticmethod
        def rotate_image(image_path, output_path, degrees):
            with Image.open(image_path) as img:
                rotated = img.rotate(degrees)
                rotated.save(output_path)

        @staticmethod
        def apply_filter(image_path, output_path, filter_type='blur'):
            with Image.open(image_path) as img:
                if filter_type == 'blur':
                    filtered = img.filter(ImageFilter.BLUR)
                elif filter_type == 'sharpen':
                    filtered = img.filter(ImageFilter.SHARPEN)
                elif filter_type == 'edge_enhance':
                    filtered = img.filter(ImageFilter.EDGE_ENHANCE)
                else:
                    raise ValueError("Unsupported filter type")
                filtered.save(output_path)

        @staticmethod
        def detect_edges(image_path, output_path):
            img = cv2.imread(image_path, 0)
            edges = cv2.Canny(img, 100, 200)
            cv2.imwrite(output_path, edges)

        @staticmethod
        def image_to_grayscale(image_path, output_path):
            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(output_path, gray)
    class TimeAndDateTools:
        @staticmethod
        def convert_timezone(dt, from_tz, to_tz):
            from_zone = pytz.timezone(from_tz)
            to_zone = pytz.timezone(to_tz)
            return dt.replace(tzinfo=from_zone).astimezone(to_zone)

        @staticmethod
        def date_difference(date1, date2):
            return abs((date2 - date1).days)

    class GeolocationTools:
        @staticmethod
        def distance_between_coords(lat1, lon1, lat2, lon2):
            return geopy.distance.distance((lat1, lon1), (lat2, lon2)).km
        @staticmethod
        def get_current_location():
            g = geocoder.ip('me')
            if g.latlng is not None:
                return g.latng
            else:
                return None
    class TextProcessingTools:
        @staticmethod
        def tokenize_text(text):
            return word_tokenize(text)

        @staticmethod
        def remove_stopwords(text):
            stop_words = set(stopwords.words('english'))
            word_tokens = word_tokenize(text)
            return [w for w in word_tokens if not w.lower() in stop_words]
        @staticmethod
        def split_words(text):
            return text.split(' ')

    class DatabaseTools:
        @staticmethod
        def create_sqlite_connection(db_file):
            conn = None
            try:
                conn = sqlite3.connect(db_file)
                return conn
            except sqlite3.Error as e:
                print(e)
            return conn

        @staticmethod
        def execute_query(connection, query):
            try:
                c = connection.cursor()
                c.execute(query)
                connection.commit()
                return True
            except sqlite3.Error as e:
                print(e)
                return False

    class LoggingAndDebuggingTools:
        @staticmethod
        def setup_logger(name, log_file, level=logging.INFO):
            formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
            handler = logging.FileHandler(log_file)
            handler.setFormatter(formatter)

            logger = logging.getLogger(name)
            logger.setLevel(level)
            logger.addHandler(handler)

            return logger

        @staticmethod
        def profile_function(func):
            def wrapper(*args, **kwargs):
                profile = cProfile.Profile()
                try:
                    return profile.runcall(func, *args, **kwargs)
                finally:
                    profile.print_stats(sort='cumulative')
            return wrapper

    class RaspberryPiSenseHAT:
        def __init__(self):
            print('Only runs on a raspberry pi.')
            from sense_hat import SenseHat
            self.sense = SenseHat()
            self.sense.clear()

        def get_temperature(self):
            return self.sense.get_temperature()

        def get_humidity(self):
            return self.sense.get_humidity()

        def get_pressure(self):
            return self.sense.get_pressure()

        def get_orientation(self):
            return self.sense.get_orientation()

        def get_accelerometer(self):
            return self.sense.get_accelerometer_raw()

        def get_gyroscope(self):
            return self.sense.get_gyroscope_raw()

        def set_pixel(self, x, y, color):
            self.sense.set_pixel(x, y, color)

        def display_message(self, message, text_color, bg_color, scroll_speed=0.1):
            self.sense.show_message(message, text_colour=text_color, back_colour=bg_color, scroll_speed=scroll_speed)

        def clear_display(self):
            self.sense.clear()

        def set_rotation(self, angle):
            self.sense.set_rotation(angle)

        def get_joystick_events(self):
            return self.sense.stick.get_events()

        def set_led_matrix(self, pixel_list):
            self.sense.set_pixels(pixel_list)

        def get_compass(self):
            return self.sense.get_compass()

    class NaturalLanguageProcessingTools:
        def __init__(self):
                print('Downloading nltk info...')
                nltk.download('punkt')
                nltk.download('averaged_perceptron_tagger')
                nltk.download('maxent_ne_chunker')
                nltk.download('words')
                nltk.download('stopwords')
                nltk.download('wordnet')
                nltk.download('vader_lexicon')
                self.stop_words = set(stopwords.words('english'))
                self.lemmatizer = WordNetLemmatizer()
                self.sentiment_analyzer = SentimentIntensityAnalyzer()
        def tokenize_text(self, text):
            return word_tokenize(text)

        def sentence_tokenize(self, text):
            return sent_tokenize(text)

        def remove_stopwords(self, tokens):
            return [token for token in tokens if token.lower() not in self.stop_words]

        def lemmatize_tokens(self, tokens):
            return [self.lemmatizer.lemmatize(token) for token in tokens]

        def named_entity_recognition(self, text):
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            named_entities = ne_chunk(pos_tags)
            return [(ne.label(), ' '.join(word for word, tag in ne.leaves()))
                    for ne in named_entities if isinstance(ne, nltk.Tree)]

        def sentiment_analysis(self, text):
            scores = self.sentiment_analyzer.polarity_scores(text)
            if scores['compound'] >= 0.05:
                return 'Positive'
            elif scores['compound'] <= -0.05:
                return 'Negative'
            else:
                return 'Neutral'

        def text_summarization(self, text, num_sentences=3):
            sentences = self.sentence_tokenize(text)
            words = self.tokenize_text(text.lower())
            words = self.remove_stopwords(words)

            word_freq = FreqDist(words)
            ranking = {}

            for i, sentence in enumerate(sentences):
                for word in self.tokenize_text(sentence.lower()):
                    if word in word_freq:
                        if i in ranking:
                            ranking[i] += word_freq[word]
                        else:
                            ranking[i] = word_freq[word]

            top_sentence_indices = nlargest(num_sentences, ranking, key=ranking.get)
            top_sentences = [sentences[i] for i in sorted(top_sentence_indices)]

            return ' '.join(top_sentences)

        def keyword_extraction(self, text, num_keywords=5):
            words = self.tokenize_text(text.lower())
            words = self.remove_stopwords(words)
            words = self.lemmatize_tokens(words)

            word_freq = FreqDist(words)
            return [word for word, _ in word_freq.most_common(num_keywords)]

        def part_of_speech_tagging(self, text):
            tokens = self.tokenize_text(text)
            return pos_tag(tokens)

        def calculate_lexical_diversity(self, text):
            tokens = self.tokenize_text(text.lower())
            return len(set(tokens)) / len(tokens)

        def extract_ngrams(self, text, n=2):
            tokens = self.tokenize_text(text.lower())
            ngrams = nltk.ngrams(tokens, n)
            return list(ngrams)
    class PDFTools:
        @staticmethod
        def create_pdf(filename, content):
            """Create a simple PDF file with given content."""
            pdf_buffer = io.BytesIO()
            c = canvas.Canvas(pdf_buffer, pagesize=letter)
            textobject = c.beginText(40, 750)
            for line in content.split('\n'):
                textobject.textLine(line)
            c.drawText(textobject)
            c.showPage()
            c.save()

            with open(filename, 'wb') as f:
                f.write(pdf_buffer.getvalue())

        @staticmethod
        def read_pdf(filename):
            """Read content from a PDF file."""
            with open(filename, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            return text

        @staticmethod
        def merge_pdfs(filenames, output_filename):
            """Merge multiple PDF files into one."""
            merger = PyPDF2.PdfMerger()
            for filename in filenames:
                merger.append(filename)
            merger.write(output_filename)
            merger.close()

    class DataVisualizationTools:
        @staticmethod
        def line_plot(x, y, title='Line Plot', xlabel='X-axis', ylabel='Y-axis'):
            plt.figure(figsize=(10, 6))
            plt.plot(x, y)
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.show()

        @staticmethod
        def scatter_plot(x, y, title='Scatter Plot', xlabel='X-axis', ylabel='Y-axis'):
            plt.figure(figsize=(10, 6))
            plt.scatter(x, y)
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.show()

        @staticmethod
        def bar_plot(x, y, title='Bar Plot', xlabel='X-axis', ylabel='Y-axis'):
            plt.figure(figsize=(10, 6))
            plt.bar(x, y)
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.show()

        @staticmethod
        def heatmap(data, title='Heatmap'):
            plt.figure(figsize=(10, 8))
            sns.heatmap(data, annot=True, cmap='coolwarm')
            plt.title(title)
            plt.show()

    class MachineLearningTools:
        @staticmethod
        def AskLocalAI(text,models):
            try:
                from ollama import generate
            except Exception as e:
                return e
            out = generate(model=models,prompt=str(text))
            return out
        @staticmethod
        def avaliable_ai_models():
            import ollama
            return ollama.ps()
        @staticmethod
        def download_ai_model(model_download):
            import ollama
            ollama.pull(model_download)
            return True
        
        @staticmethod
        def linear_regression(X, y):
            """Perform simple linear regression."""
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LinearRegression()
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            return model, score

        @staticmethod
        def kmeans_clustering(X, n_clusters=3):
            """Perform K-means clustering."""
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans.fit(X_scaled)
            return kmeans, scaler
    class Stocks():
        def __init__(self):
            pass
        def fetch_ticker(self,ticker):
            tickerOBJ = yf.Ticker(ticker)
            return tickerOBJ.get_info()
        def fetch_news(self,ticker):
            tickerOBJ = yf.Ticker(ticker)
            total_list = []
            for news in tickerOBJ.get_news():
                obj = {}
                obj['UUID'] = news['uuid']
                obj['Title'] = news['title']
                obj['Publisher'] = news['publisher']
                obj['links'] = news['link']
                obj['Type'] = news['type']
                obj['img_link'] = news['thumbnail']['resolutions'][0]['url']
                total_list.append(obj)
            return total_list
        def fetch_multiple_tickers(self,tickers):
            alv = []
            for ticker in tickers:
                objectInfo = yf.Ticker(ticker)
                alv.append(objectInfo.get_info)
            return alv
    class VideoTools():
        def __init__(self):
            self.ydl_opts = {
                'format':'bestaudio/best',
                'outtmpl': '%(title)s.%(ext)s'
            }
        def DownloadYoutubeURLVideo(self,url):

            with ytDLP.YoutubeDL(self.ydl_opts) as ydl:
                ydl.download([url])
        def InfoVideo(self,url):
            with ytDLP.YoutubeDL(self.ydl_opts) as ydl:
                data = ydl.extract_info(url,download=False)
                return data

        def trim_video(input_file, output_file, start_time, duration):
            stream = ffmpeg.input(input_file, ss=start_time, t=duration)
            stream = ffmpeg.output(stream, output_file)
            ffmpeg.run(stream)

        def extract_audio(input_file, output_file):
            stream = ffmpeg.input(input_file)
            stream = ffmpeg.output(stream, output_file, acodec='libmp3lame')
            ffmpeg.run(stream)

        def resize_video(input_file, output_file, width, height):
            stream = ffmpeg.input(input_file)
            stream = ffmpeg.filter(stream, 'scale', width=width, height=height)
            stream = ffmpeg.output(stream, output_file)
            ffmpeg.run(stream)

        def add_watermark(input_file, watermark_file, output_file):
            main = ffmpeg.input(input_file)
            logo = ffmpeg.input(watermark_file)
            stream = ffmpeg.filter([main, logo], 'overlay', 'W-w-10:H-h-10')
            stream = ffmpeg.output(stream, output_file)
            ffmpeg.run(stream)

        def convert_video_format(input_file, output_file):
            stream = ffmpeg.input(input_file)
            stream = ffmpeg.output(stream, output_file)
            ffmpeg.run(stream)

        def adjust_volume(input_file, output_file, volume):
            stream = ffmpeg.input(input_file)
            stream = ffmpeg.filter(stream, 'volume', volume=volume)
            stream = ffmpeg.output(stream, output_file)
            ffmpeg.run(stream)

        def create_thumbnail(input_file, output_file, time):
            stream = ffmpeg.input(input_file, ss=time)
            stream = ffmpeg.filter(stream, 'scale', 300, -1)
            stream = ffmpeg.output(stream, output_file, vframes=1)
            ffmpeg.run(stream)

        def concat_videos(input_files, output_file):
            inputs = [ffmpeg.input(file) for file in input_files]
            stream = ffmpeg.concat(*inputs)
            stream = ffmpeg.output(stream, output_file)
            ffmpeg.run(stream)

        def add_subtitles(input_file, subtitle_file, output_file):
            stream = ffmpeg.input(input_file)
            stream = ffmpeg.filter(stream, 'subtitles', subtitle_file)
            stream = ffmpeg.output(stream, output_file)
            ffmpeg.run(stream)

        def rotate_video(input_file, output_file, angle):
            stream = ffmpeg.input(input_file)
            stream = ffmpeg.filter(stream, 'rotate', angle=angle)
            stream = ffmpeg.output(stream, output_file)

            ffmpeg.run(stream)
    class APIIntegrations():
        def __init__(self):
            pass
        class ChatGPT():
            def __init__(self,model,APIKEY):
                self.model_engine = model
                self.client = openai.Client(api_key=APIKEY)
            def call_text_api(self,message):
                response = openai.chat.completions.create(
                    model=self.model_engine,
                    messages=[
                        {
                            "role":"user","content":str(message)
                        }
                    ]
                )
                return response.choices[0].message
            def models(self):
                return [
                    'gpt-4o',
                    'gpt-4o-mini',
                    'gpt-4',
                    'gpt-3.5-turbo',
                    'gpt-3.5',
                    'dall-e-2',
                    'dall-e-3'
                ]
            def call_dalle_api(self,prompt,quality='standard',open_in_browser=False):
                response = self.client.images.generate(
                    model=self.model_engine,
                    prompt=str(prompt),
                    size="1024x1024",
                    quality=str(quality),
                    n=1
                )
                if open_in_browser:
                    webbrowser.open(response.data[0].url)
                return {
                    "url":response.data[0].url,

                }
        class GeminiAPI():
            def __init__(self,APIKEY):
                self.APIKEY = APIKEY
            def generate_content(self, text):
                url = f"https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent?key={self.APIKEY}"
                headers = {'Content-Type': 'application/json'}
                data = {
                    "contents": [
                        {
                            "role": "user",
                            "parts": [{"text": text}]
                        }
                    ]
                }
                response = requests.post(url, headers=headers, json=data)
                text_content = response.json()['candidates'][0]['content']['parts'][0]['text']
                return text_content
            def provide_text_and_image_input(self,text,image_fp):
                url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={self.APIKEY}"
                image_path = Path(image_fp)
                prepared_image = base64.b64encode(image_path.read_bytes()).decode('utf-8')

                payload = {
                    "contents": [
                        {
                            "parts": [
                                {"text": "Do these look store-bought or homemade?"},
                                {
                                    "inlineData": {
                                        "mimeType": "image/png",
                                        "data": prepared_image
                                    }
                                }
                            ]
                        }
                    ]
                }
                headers = {'Content-Type':'application/json'}
                response = requests.post(url,json=payload,headers=headers)
                return response.json()
        class BingAPI():
            def __init__(self,api_key):
                self.api_key = api_key
            def call_bing_api(self,text,count=10):
                headers = {
                "Ocp-Apim-Subscription-Key": self.api_key
                }
                params = {
                    'q':str(text),
                    'count':int(count),
                    'offset':0,
                    "mkt": "en-us",
                    "safesearch": "Moderate"

                }
                response = requests.get("https://api.cognitive.microsoft.com/bing/v7.0/search", headers=headers, params=params)
                if response.status_code == 200:
                    return response.json()
                else:
                    info = {}
                    info['Error'] = "Status code is not 200."
                    info['Response Info'] = response.json()
                    return info
        class WorldNewsAPI():
            def __init__(self,api_key):
                self.api_key = api_key
            def search_worldnewsapi(self,search_query):
                url = f"https://api.worldnewsapi.com/search-news?api-key={self.api_key}&text={str(search_query)}"
                response = requests.get(url)
                if response.status_code == 200:
                    return response.json()
                else:
                    return {
                        "error":f"Status code is {response.status_code}",
                        "json":response.json()

                    }
            def top_news(self,source_country="us",language="en",date="2024-05-29"):
                url = f"https://api.worldnewsapi.com/top-news?source-country={source_country}&language={language}&date={date}"
                api_key = self.api_key

                headers = {
                    'x-api-key': api_key
                }

                response = requests.get(url, headers=headers)

                if response.status_code == 200:
                    return response.json()
                else:
                    return f"Error: {response.status_code}"
        class FreeDictionaryAPI():
            def __init__(self):
                pass
            def search_word(self,word):
                url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}"
                response = requests.get(url)
                if response.status_code == 200:
                    return response.json()
                else:
                    return {
                        "error!":"not it"
                    }
        class TemporaryEmail():
            def __init__(self):

                BASE_URL = "https://api.mail.gw"
                def get_domains():
                    response = requests.get(f"{BASE_URL}/domains")
                    return json.loads(response.text)

                def generate_random_email(self,domains):
                    username = ''.join(random.choices(string.ascii_lowercase + string.digits, k=10))
                    domain = random.choice(domains['hydra:member'])['domain']
                    return f"{username}@{domain}"

                def create_account(self,address, password):
                    data = {
                        "address": address,
                        "password": password
                    }
                    response = requests.post(f"{BASE_URL}/accounts", json=data)
                    return json.loads(response.text)

                def get_token(self,address, password):
                    data = {
                        "address": address,
                        "password": password
                    }
                    response = requests.post(f"{BASE_URL}/token", json=data)
                    return json.loads(response.text)
            
            def return_temp_email(self):

                return {
                    "temp_email":self.temp_email,
                    "temp_password":self.temp_password

                }
        class OpenSkyAPI:
            BASE_URL = "https://opensky-network.org/api"

            def __init__(self, username=None, password=None):
                self.auth = (username, password) if username and password else None

            def get_flights_all(self, begin, end):
                """Get state vectors for all flights"""
                url = f"{self.BASE_URL}/flights/all"
                params = {
                    'begin': begin,
                    'end': end
                }
                response = requests.get(url, params=params, auth=self.auth)
                return response.json()

            def get_flights_by_aircraft(self, icao24, begin, end):
                """Get flights for a specific aircraft"""
                url = f"{self.BASE_URL}/flights/aircraft"
                params = {
                    'icao24': icao24,
                    'begin': begin,
                    'end': end
                }
                response = requests.get(url, params=params, auth=self.auth)
                return response.json()

            def get_flights_by_airport(self, airport, begin, end):
                """Get flights for a specific airport"""
                url = f"{self.BASE_URL}/flights/arrival"
                params = {
                    'airport': airport,
                    'begin': begin,
                    'end': end
                }
                response = requests.get(url, params=params, auth=self.auth)
                return response.json()

            def get_states(self, time=0, icao24=None):
                """Get state vectors (all or filtered by ICAO24)"""
                url = f"{self.BASE_URL}/states/all"
                params = {'time': time}
                if icao24:
                    params['icao24'] = icao24
                response = requests.get(url, params=params, auth=self.auth)
                return response.json()

            def get_airports(self):
                """Get a list of all airports"""
                url = f"{self.BASE_URL}/airports"
                response = requests.get(url, auth=self.auth)
                return response.json()

        class TwitterAPI:
            def __init__(self, consumer_key, consumer_secret, access_token, access_token_secret):
                import tweepy
                auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
                auth.set_access_token(access_token, access_token_secret)
                self.api = tweepy.API(auth)

            def get_user_tweets(self, username, count=10):
                tweets = self.api.user_timeline(screen_name=username, count=count)
                return [{'text': tweet.text, 'created_at': tweet.created_at} for tweet in tweets]

            def search_tweets(self, query, count=10):
                tweets = self.api.search_tweets(q=query, count=count)
                return [{'text': tweet.text, 'created_at': tweet.created_at} for tweet in tweets]

        class RedditAPI:
            def __init__(self, client_id, client_secret, user_agent):
                import praw
                self.reddit = praw.Reddit(client_id=client_id,
                                        client_secret=client_secret,
                                        user_agent=user_agent)

            def get_subreddit_posts(self, subreddit_name, limit=10):
                subreddit = self.reddit.subreddit(subreddit_name)
                return [{'title': post.title, 'url': post.url, 'score': post.score} 
                        for post in subreddit.hot(limit=limit)]

            def search_reddit(self, query, limit=10):
                return [{'title': post.title, 'url': post.url, 'score': post.score} 
                        for post in self.reddit.subreddit('all').search(query, limit=limit)]
    class BioinformaticsTools:
        @staticmethod
        def analyze_dna_sequence(sequence):
            seq = Seq(sequence)

            return {
                'length': len(seq),
                'gc_content': seq.count('G') + seq.count('C'),
                'complement': str(seq.complement()),
                'reverse_complement': str(seq.reverse_complement()),
                'transcription': str(seq.transcribe()),
                'complement_rna': str(seq.complement_rna()),
                "reverse_complement_rna": str(seq.reverse_complement_rna()),
                'translation': str(seq.translate()),
                'molecular_weight': molecular_weight(seq),
                'nucleotide_counts': {
                    'A': seq.count('A'),
                    'T': seq.count('T'),
                    'G': seq.count('G'),
                    'C': seq.count('C')
                }
            }

        @staticmethod
        def analyze_protein_sequence(sequence):
            seq = Seq(sequence)
            param = ProtParam.ProteinAnalysis(str(seq))

            return {
                'length': len(seq),
                'molecular_weight': param.molecular_weight(),
                'isoelectric_point': param.isoelectric_point(),
                'aromaticity': param.aromaticity(),
                'instability_index': param.instability_index(),
                'gravy': param.gravy(),
                'secondary_structure_fraction': param.secondary_structure_fraction(),
                'amino_acid_counts': param.count_amino_acids()
            }

        @staticmethod
        def read_fasta_file(file_path):
            sequences = []
            for record in SeqIO.parse(file_path, "fasta"):
                sequences.append({
                    'id': record.id,
                    'sequence': str(record.seq),
                    'description': record.description
                })
            return sequences

        @staticmethod
        def write_fasta_file(sequences, file_path):
            records = [SeqIO.SeqRecord(Seq(seq['sequence']), id=seq['id'], description=seq['description'])
                    for seq in sequences]
            SeqIO.write(records, file_path, "fasta")

        @staticmethod
        def fetch_sequence_from_ncbi(accession, email):
            Entrez.email = email
            with Entrez.efetch(db="nucleotide", id=accession, rettype="gb", retmode="text") as handle:
                record = SeqIO.read(handle, "genbank")
            return {
                'id': record.id,
                'sequence': str(record.seq),
                'description': record.description,
                'features': [{'type': f.type, 'location': str(f.location)} for f in record.features]
            }

        @staticmethod
        def perform_blast(sequence, database="nt", program="blastn"):
            result_handle = NCBIWWW.qblast(program, database, sequence)
            blast_records = NCBIXML.parse(result_handle)
            results = []
            for record in blast_records:
                for alignment in record.alignments:
                    for hsp in alignment.hsps:
                        results.append({
                            'title': alignment.title,
                            'length': alignment.length,
                            'e_value': hsp.expect,
                            'score': hsp.score,
                            'identities': hsp.identities,
                            'gaps': hsp.gaps
                        })
            return results

        @staticmethod
        def align_sequences(sequences, algorithm="clustalw"):
            from io import StringIO
            fasta_io = StringIO()
            SeqIO.write(sequences, fasta_io, "fasta")
            fasta_io.seek(0)
            alignment = AlignIO.read(fasta_io, "fasta")
            return alignment

        @staticmethod
        def create_phylogenetic_tree(aligned_sequences):
            calculator = DistanceCalculator('identity')
            dm = calculator.get_distance(aligned_sequences)
            constructor = DistanceTreeConstructor(calculator)
            tree = constructor.build_tree(aligned_sequences)
            return tree

        @staticmethod
        def visualize_phylogenetic_tree(tree):
            fig, ax = plt.subplots(figsize=(10, 8))
            Phylo.draw(tree, axes=ax)
            plt.show()

        @staticmethod
        def fetch_pdb_structure(pdb_id):
            pdbl = PDBList()
            parser = PDBParser()
            pdb_file = pdbl.retrieve_pdb_file(pdb_id, file_format='pdb', pdir='.')
            structure = parser.get_structure(pdb_id, pdb_file)
            return structure

        @staticmethod
        def find_open_reading_frames(sequence, min_length=100):
            seq = Seq(sequence)
            orfs = []
            for strand, nuc in [(+1, seq), (-1, seq.reverse_complement())]:
                for frame in range(3):
                    for pro in nuc[frame:].translate().split("*"):
                        if len(pro) >= min_length/3:
                            orfs.append({
                                'strand': strand,
                                'frame': frame,
                                'protein': str(pro),
                                'start': frame + 3 * strand * pro.start(),
                                'end': frame + 3 * strand * (pro.start() + len(pro))
                            })
            return orfs
        @staticmethod
        def predict_protein_localization(sequence):
            matrix = substitution_matrices.load("BLOSUM62")
            signal_peptide = Seq.Seq("MKKLTALSLALVLLAFTVFA")
            
            aligner = PairwiseAligner()
            aligner.substitution_matrix = matrix
            aligner.open_gap_score = -11
            aligner.extend_gap_score = -1
            
            alignments = aligner.align(sequence, signal_peptide)
            best_alignment = max(alignments, key=lambda a: a.score)
            
            score = best_alignment.score
            if score > 50:
                return "Likely secreted (contains signal peptide)"
            else:
                return "Likely cytoplasmic (no signal peptide detected)"
    class GameDevelopmentUtils:
        def __init__(self):
            pygame.init()

        @staticmethod
        def create_game_window(width, height, title):
            screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption(title)
            return screen

        @staticmethod
        def game_loop(screen, game_logic_func):
            running = True
            clock = pygame.time.Clock()
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False

                game_logic_func(screen)
                pygame.display.flip()
                clock.tick(60)

            pygame.quit()

        @staticmethod
        def draw_rectangle(screen, color, x, y, width, height):
            pygame.draw.rect(screen, color, (x, y, width, height))

    class OptimizationAlgorithms:
        @staticmethod
        def genetic_algorithm(fitness_func, gene_length, population_size=100, generations=100):
            def create_individual():
                return [random.randint(0, 1) for _ in range(gene_length)]

            population = [create_individual() for _ in range(population_size)]

            for _ in range(generations):
                population = sorted(population, key=fitness_func, reverse=True)
                new_population = population[:population_size // 2]

                while len(new_population) < population_size:
                    parent1, parent2 = random.sample(population[:population_size // 2], 2)
                    crossover_point = random.randint(1, gene_length - 1)
                    child = parent1[:crossover_point] + parent2[crossover_point:]
                    if random.random() < 0.1:  # mutation probability
                        mutation_point = random.randint(0, gene_length - 1)
                        child[mutation_point] = 1 - child[mutation_point]
                    new_population.append(child)

                population = new_population

            return max(population, key=fitness_func)

        @staticmethod
        def simulated_annealing(cost_func, initial_state, temperature=10000.0, cooling_rate=0.995, iterations=1000):
            current_state = initial_state
            current_cost = cost_func(current_state)
            best_state = current_state
            best_cost = current_cost

            for _ in range(iterations):
                neighbor = current_state + np.random.normal(0, 1, len(current_state))
                neighbor_cost = cost_func(neighbor)

                if neighbor_cost < current_cost or random.random() < np.exp((current_cost - neighbor_cost) / temperature):
                    current_state = neighbor
                    current_cost = neighbor_cost

                if current_cost < best_cost:
                    best_state = current_state
                    best_cost = current_cost

                temperature *= cooling_rate

            return best_state, best_cost

    class BlockchainCryptoTools:
        def __init__(self, provider_url):
            self.w3 = Web3(Web3.HTTPProvider(provider_url))

        def get_ethereum_balance(self, address):
            balance_wei = self.w3.eth.get_balance(address)
            balance_eth = self.w3.from_wei(balance_wei, 'ether')
            return balance_eth

        def get_latest_block(self):
            return self.w3.eth.get_block('latest')

        @staticmethod
        def generate_ethereum_wallet():
            account = Web3().eth.account.create()
            return {
                'address': account.address,
                'private_key': account.privateKey.hex()
            }

    class AudioProcessingTools:
        def __init__(self):
            pass

        @staticmethod
        def load_audio(file_path, sr=22050):
            """
            Load an audio file.

            :param file_path: Path to the audio file
            :param sr: Sample rate (default: 22050 Hz)
            :return: Audio time series and sampling rate
            """
            return librosa.load(file_path, sr=sr)

        @staticmethod
        def display_waveform(y, sr):
            """
            Display the waveform of an audio signal.

            :param y: Audio time series
            :param sr: Sampling rate
            """
            plt.figure(figsize=(12, 4))
            librosa.display.waveshow(y, sr=sr)
            plt.title('Audio Waveform')
            plt.xlabel('Time')
            plt.ylabel('Amplitude')
            plt.show()

        @staticmethod
        def compute_mel_spectrogram(y, sr):
            """
            Compute and display the Mel spectrogram of an audio signal.

            :param y: Audio time series
            :param sr: Sampling rate
            """
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

            plt.figure(figsize=(12, 4))
            librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel')
            plt.colorbar(format='%+2.0f dB')
            plt.title('Mel Spectrogram')
            plt.show()

        @staticmethod
        def extract_mfcc(y, sr, n_mfcc=13):
            """
            Extract Mel-frequency cepstral coefficients (MFCCs) from an audio signal.

            :param y: Audio time series
            :param sr: Sampling rate
            :param n_mfcc: Number of MFCCs to return
            :return: MFCCs
            """
            return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

        @staticmethod
        def detect_onset(y, sr):
            """
            Detect onsets in an audio signal.

            :param y: Audio time series
            :param sr: Sampling rate
            :return: Array of detected onset frames
            """
            return librosa.onset.onset_detect(y=y, sr=sr)

        @staticmethod
        def estimate_tempo(y, sr):
            """
            Estimate the tempo of an audio signal.

            :param y: Audio time series
            :param sr: Sampling rate
            :return: Estimated tempo in beats per minute
            """
            return librosa.beat.tempo(y=y, sr=sr)[0]

        @staticmethod
        def pitch_shift(y, sr, n_steps):
            """
            Shift the pitch of an audio signal.

            :param y: Audio time series
            :param sr: Sampling rate
            :param n_steps: Number of semitones to shift
            :return: Pitch-shifted audio time series
            """
            return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)

        @staticmethod
        def time_stretch(y, rate):
            """
            Time-stretch an audio signal without changing its pitch.

            :param y: Audio time series
            :param rate: Stretch factor (1.0 = no stretch)
            :return: Time-stretched audio time series
            """
            return librosa.effects.time_stretch(y=y, rate=rate)

        @staticmethod
        def harmonic_percussive_separation(y):
            """
            Separate an audio signal into harmonic and percussive components.

            :param y: Audio time series
            :return: Tuple of harmonic and percussive components
            """
            return librosa.effects.hpss(y)
        @staticmethod
        def text_to_speech(text, output_file):
            tts = gTTS(text=text, lang='en')
            tts.save(output_file)

        @staticmethod
        def speech_to_text(audio_file):
            recognizer = sr.Recognizer()
            with sr.AudioFile(audio_file) as source:
                audio = recognizer.record(source)
            try:
                text = recognizer.recognize_google(audio)
                return text
            except sr.UnknownValueError:
                return "Speech recognition could not understand audio"
            except sr.RequestError as e:
                return f"Could not request results from speech recognition service; {e}"

    class NetworkSecurityTools:
        @staticmethod
        def port_scan(target_ip):
            nm = nmap.PortScanner()
            nm.scan(target_ip, arguments='-p-')
            open_ports = nm[target_ip]['tcp'].keys()
            return list(open_ports)

        @staticmethod
        def encrypt_message(message, key):
            f = Fernet(key)
            encrypted_message = f.encrypt(message.encode())
            return encrypted_message

        @staticmethod
        def decrypt_message(encrypted_message, key):
            f = Fernet(key)
            decrypted_message = f.decrypt(encrypted_message).decode()
            return decrypted_message

        @staticmethod
        def generate_encryption_key():
            return Fernet.generate_key()

        @staticmethod
        def network_traffic_analysis(interface, duration):
            packets = scapy.sniff(iface=interface, timeout=duration)
            return packets.summary()

    class DataCompressionTools:
        @staticmethod
        def compress_string(data, algorithm='zlib'):
            if algorithm == 'zlib':
                return zlib.compress(data.encode())
            elif algorithm == 'gzip':
                return gzip.compress(data.encode())
            elif algorithm == 'bz2':
                return bz2.compress(data.encode())
            elif algorithm == 'lzma':
                return lzma.compress(data.encode())
            else:
                raise ValueError("Unsupported compression algorithm")

        @staticmethod
        def decompress_string(compressed_data, algorithm='zlib'):
            if algorithm == 'zlib':
                return zlib.decompress(compressed_data).decode()
            elif algorithm == 'gzip':
                return gzip.decompress(compressed_data).decode()
            elif algorithm == 'bz2':
                return bz2.decompress(compressed_data).decode()
            elif algorithm == 'lzma':
                return lzma.decompress(compressed_data).decode()
            else:
                raise ValueError("Unsupported decompression algorithm")

        @staticmethod
        def compress_file(self,input_file, output_file, algorithm='gzip'):
            with open(input_file, 'rb') as f_in:
                data = f_in.read()
            compressed_data = self.compress_string(data.decode(), algorithm)
            with open(output_file, 'wb') as f_out: 
                f_out.write(compressed_data)

        @staticmethod
        def decompress_file(self,input_file, output_file, algorithm='gzip'):
            with open(input_file, 'rb') as f_in:
                compressed_data = f_in.read()
            decompressed_data = self.decompress_string(compressed_data, algorithm)
            with open(output_file, 'w') as f_out:
                f_out.write(decompressed_data)

        @staticmethod
        def analyze_compression(self,data, algorithms=['zlib', 'gzip', 'bz2', 'lzma']):
            original_size = len(data)
            results = {}
            for algorithm in algorithms:
                compressed = self.compress_string(data, algorithm)
                compressed_size = len(compressed)
                ratio = compressed_size / original_size
                results[algorithm] = {
                    'original_size': original_size,
                    'compressed_size': compressed_size,
                    'compression_ratio': ratio,
                    'space_saving': 1 - ratio
                }
            return results

    class FinancialAnalysisTools:
        @staticmethod
        def get_stock_data(ticker, start_date, end_date):
            stock = yf.Ticker(ticker)
            data = stock.history(start=start_date, end=end_date)
            return data

        @staticmethod
        def plot_stock_price(data, title):
            plt.figure(figsize=(12, 6))
            plt.plot(data.index, data['Close'])
            plt.title(title)
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.grid(True)
            plt.show()

        @staticmethod
        def calculate_returns(data):
            return data['Close'].pct_change()

        @staticmethod
        def calculate_volatility(returns):
            return returns.std() * np.sqrt(252)  # Annualized volatility

        @staticmethod
        def calculate_sharpe_ratio(self,returns, risk_free_rate=0.01):
            volatility = self.calculate_volatility(returns)
            excess_returns = returns.mean() * 252 - risk_free_rate
            return excess_returns / volatility

        @staticmethod
        def portfolio_optimization(self,tickers, start_date, end_date, num_portfolios=10000):
            data = pd.DataFrame()
            for ticker in tickers:
                stock_data = self.get_stock_data(ticker, start_date, end_date)
                data[ticker] = stock_data['Close']

            returns = data.pct_change()
            mean_returns = returns.mean()
            cov_matrix = returns.cov()

            results = np.zeros((3, num_portfolios))
            weights_record = []

            for i in range(num_portfolios):
                weights = np.random.random(len(tickers))
                weights /= np.sum(weights)
                portfolio_return = np.sum(mean_returns * weights) * 252
                portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
                results[0,i] = portfolio_std_dev
                results[1,i] = portfolio_return
                results[2,i] = (portfolio_return - 0.01) / portfolio_std_dev
                weights_record.append(weights)

            return results, weights_record

        @staticmethod
        def stock_price_prediction(data, days_to_predict=30):
            df = data['Close'].resample('D').mean().dropna()
            model = ARIMA(df, order=(1,1,1))
            results = model.fit()
            forecast = results.forecast(steps=days_to_predict)
            return forecast
    class DatasetVisualizer:
        def __init__(self):
            self.datasets = {
                'iris': datasets.load_iris(),
                'digits': datasets.load_digits(),
                'wine': datasets.load_wine(),
                'breast_cancer': datasets.load_breast_cancer()
            }

        def visualize_dataset(self, dataset_name, feature_indices=(0, 1)):
            if dataset_name not in self.datasets:
                raise ValueError(f"Dataset '{dataset_name}' not found. Available datasets: {list(self.datasets.keys())}")

            dataset = self.datasets[dataset_name]
            X = dataset.data
            y = dataset.target

            plt.figure(figsize=(10, 6))
            scatter = plt.scatter(X[:, feature_indices[0]], X[:, feature_indices[1]], c=y, cmap=plt.cm.Set1)
            plt.xlabel(dataset.feature_names[feature_indices[0]])
            plt.ylabel(dataset.feature_names[feature_indices[1]])
            plt.title(f'{dataset_name.capitalize()} Dataset - {dataset.feature_names[feature_indices[0]]} vs {dataset.feature_names[feature_indices[1]]}')
            plt.colorbar(scatter)
            plt.show()

        def visualize_digits(self, num_samples=5):
            digits = self.datasets['digits']
            _, axes = plt.subplots(1, num_samples, figsize=(10, 3))
            for ax, image, label in zip(axes, digits.images, digits.target):
                ax.set_axis_off()
                ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
                ax.set_title(f'Digit: {label}')
            plt.show()

        def plot_feature_importance(self, dataset_name):
            if dataset_name not in self.datasets:
                raise ValueError(f"Dataset '{dataset_name}' not found. Available datasets: {list(self.datasets.keys())}")

            dataset = self.datasets[dataset_name]
            X = dataset.data
            y = dataset.target

            from sklearn.ensemble import RandomForestClassifier
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(X, y)

            importances = clf.feature_importances_
            indices = np.argsort(importances)[::-1]

            plt.figure(figsize=(10, 6))
            plt.title(f"Feature Importances - {dataset_name.capitalize()} Dataset")
            plt.bar(range(X.shape[1]), importances[indices])
            plt.xticks(range(X.shape[1]), [dataset.feature_names[i] for i in indices], rotation=90)
            plt.tight_layout()
            plt.show()
    class AdvancedMachineLearning:
        def __init__(self):
            import numpy as np
            import pandas as pd
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import accuracy_score, mean_squared_error
            import xgboost as xgb
            import lightgbm as lgb
            import shap

            self.np = np
            self.pd = pd
            self.train_test_split = train_test_split
            self.StandardScaler = StandardScaler
            self.accuracy_score = accuracy_score
            self.mean_squared_error = mean_squared_error
            self.xgb = xgb
            self.lgb = lgb
            self.shap = shap

        def prepare_data(self, X, y, test_size=0.2, random_state=42):
            X_train, X_test, y_train, y_test = self.train_test_split(X, y, test_size=test_size, random_state=random_state)
            scaler = self.StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            return X_train_scaled, X_test_scaled, y_train, y_test

        def train_xgboost(self, X_train, y_train, X_test, y_test, params=None):
            if params is None:
                params = {'max_depth': 3, 'learning_rate': 0.1, 'n_estimators': 100}
            model = self.xgb.XGBClassifier(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = self.accuracy_score(y_test, y_pred)
            return model, accuracy

        def train_lightgbm(self, X_train, y_train, X_test, y_test, params=None):
            if params is None:
                params = {'num_leaves': 31, 'learning_rate': 0.05, 'n_estimators': 100}
            model = self.lgb.LGBMClassifier(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = self.accuracy_score(y_test, y_pred)
            return model, accuracy

        def explain_model(self, model, X):
            explainer = self.shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            self.shap.summary_plot(shap_values, X)

        def train_neural_network(self, X_train, y_train, X_test, y_test):
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense

            model = Sequential([
                Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
                Dense(32, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=0)
            _, accuracy = model.evaluate(X_test, y_test, verbose=0)
            return model, accuracy

    class AIUtilities:
        def __init__(self):
            self.nlp = spacy.load("en_core_web_sm")

        # Natural Language Processing (NLP) Enhancements
        def named_entity_recognition(self, text):
            doc = self.nlp(text)
            return [(ent.text, ent.label_) for ent in doc.ents]

        def sentiment_analysis(self, text):
            analysis = TextBlob(text)
            return analysis.sentiment

        def topic_modeling(self, texts):
            dictionary = Dictionary(texts)
            corpus = [dictionary.doc2bow(text) for text in texts]
            lda = LdaModel(corpus, num_topics=5, id2word=dictionary)
            return lda.print_topics()

        def text_summarization(self, text):
            return summarize(text)

        def object_detection(self, image_path, model_name="yolov8m.pt",output_file=False,outputFilename="output.jpg"):
            model = YOLO(model_name)
            results = model.predict(image_path)
            result = results[0]
            
            number_of_identified_obj = len(result.boxes)
            output_json = {
                "number_of_identifiedOBJ":number_of_identified_obj,
                
            }
            c = []
            for box in result.boxes:
                obj = {}
                coords = box.xyxy[0].tolist()
                object_type = box.cls[0].item()
                probability = box.conf[0].item()
                obj_str_name = result.names[box.cls[0].item()]
                obj['Coordinates'] = coords
                obj['Object Type (#)'] = object_type
                obj['Probability'] = probability
                obj['Object Type (str)'] = obj_str_name
                c.append(obj)
            output_json['objects'] = c

            if output_file:
                img_obj = Image.fromarray(result.plot[:,:,::-1])
                
                img_obj.save(outputFilename)
            return output_json
            


        def face_recognition(self, image_path,faces_database):
            from deepface import DeepFace
            jsf = {}
            recognition = DeepFace.find(img_path=str(image_path),db_path=str(faces_database))
            # grab the most accurate (less distance)
            analysis = DeepFace.analyze(img_path=str(image_path),actions=["age","gender","emotion","race"])
            emotion_information = jsf['emotion']
            emotion_information['angry'] = analysis['emotion']['angry']
            emotion_information['disgust'] = analysis['emotion']['disgust']
            emotion_information['fear'] = analysis['emotion']['fear']
            emotion_information['happy'] = analysis['emotion']['happy']
            emotion_information['sad'] = analysis['emotion']['sad']
            emotion_information['suprise'] = analysis['emotion']['suprise']
            emotion_information['neutral'] = analysis['emotion']['neutral']
            emotion_information['dominant_emotion'] = analysis['dominant_emotion']
            jsf['age'] = analysis['age']
            jsf['gender'] = analysis['gender']
            race_a = analysis['race']
            race = jsf['race']
            jsf['analysis'] = recognition
            race = race_a
            jsf['dominant_race'] = analysis['dominant_race']
            return jsf
            




        def optical_character_recognition(self, image_path):
            image = cv2.imread(image_path)
            text = pytesseract.image_to_string(image)
            return text

        # Time Series Analysis
        def forecasting(self, data):
            model = SARIMAX(data, order=(1, 1, 1))
            results = model.fit()
            return results

        def anomaly_detection(self, data):
            # Placeholder for anomaly detection logic
            return data

        # Reinforcement Learning utilities
        def basic_rl_algorithm(self, env):
            model = PPO('MlpPolicy', env, verbose=1)
            model.learn(total_timesteps=10000)
            return model

        # Graph Analysis tools
        def network_analysis(self, graph):
            G = nx.Graph(graph)
            return nx.info(G)

        def graph_visualization(self, graph):
            G = nx.Graph(graph)
            nx.draw(G)
            return G

        # Big Data processing tools
        def big_data_processing_with_spark(self, data):
            spark = SparkSession.builder.appName("BigDataProcessing").getOrCreate()
            df = spark.createDataFrame(data)
            return df

        def big_data_processing_with_dask(self, data):
            df = dd.from_pandas(pd.DataFrame(data), npartitions=4)
            return df

      
        # Explainable AI (XAI) methods
        def explain_with_lime(self, model, data):
            explainer = LimeTextExplainer()
            explanation = explainer.explain_instance(data, model.predict_proba, num_features=6)
            return explanation


        # AutoML capabilities
        def automated_feature_selection(self, data, target):
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data)
            return scaled_data

        def hyperparameter_tuning(self, model, params, data, target):
            grid = GridSearchCV(model, params)
            grid.fit(data, target)
            return grid.best_params_

        # Deep Learning model architectures
        def create_cnn(self, input_shape):
            model = Sequential()
            model.add(Dense(64, activation='relu', input_shape=input_shape))
            model.add(Dense(10, activation='softmax'))
            return model

        def create_rnn(self, input_shape):
            model = Sequential()
            model.add(LSTM(50, input_shape=input_shape))
            model.add(Dense(1))
            return model


        # Quantum Computing utilities
        def quantum_circuits(self):
            dev = qml.device("default.qubit", wires=1)

            @qml.qnode(dev)
            def circuit(params):
                qml.RX(params[0], wires=0)
                qml.RY(params[1], wires=0)
                return qml.expval(qml.PauliZ(0))
            
            return circuit

        # Privacy-preserving ML techniques
        def differential_privacy_model(self, data, target):
            model = dp_models.LogisticRegression()
            model.fit(data, target)
            return model

        def federated_learning(self, data, target):
            fl = FederatedLearning(data, target)
            return fl
    class EmailFunctions():
        def __init__(self):
            pass
        @staticmethod
        def gmail_send_email(email,password,recipient,subject,body,imagePath=None):
            smtpOBJ = yagmail.SMTP(email,password)
            if imagePath is not None:
                content = [
                    str(body),
                    {str(imagePath),"Sent Image"}
                ]
                smtpOBJ.send(to=recipient,subject=subject,contents=str(content))
            else:
                smtpOBJ.send(to=recipient,subject=str(subject),body=str(body))
            return True
        
        def send_email(sender_email, sender_password, recipient_email, subject, body,smtp_server='smtp.gmail.com',port=587):
            # Set up the MIME
            message = MIMEMultipart()
            message['From'] = sender_email
            message['To'] = recipient_email
            message['Subject'] = subject

            # Add body to email
            message.attach(MIMEText(body, 'plain'))

            # Create SMTP session
            try:
                with smtplib.SMTP(smtp_server, port) as server:
                    server.starttls()  # Enable security
                    # Login to the server
                    server.login(sender_email, sender_password)
                    
                    # Send the email
                    server.send_message(message)
                    print("Email sent successfully!")
            except Exception as e:
                print(f"An error occurred: {e}")
    class read_write_json_yaml():
        def __init__(self):
            pass
        def read_yaml(self,filename):
            return yaml.load(filename)

        def read_json(self, filename):
            with open(filename, 'r') as file:
                return json.load(file)
        def dump_json(self,filename,data):
            return json.dump(data,filename)
        def dump_yaml(self,filename,data):
            return yaml.dump(data,filename)
    class BeautifulSoup():
        def __init__(self,html_content=None,url=None,parser="html.parser"):
            if url:
                response = requests.get(str(url))
                response.raise_for_status()
                html_content = response.text
            elif html_content is None:
                raise ValueError('Either html_content or url is not filled.')
        
            self.soup = BeautifulSoup(html_content)
        def get_text(self,selector):
            element = self.soup.select_one(selector)
            return element.get_text(strip=True) if element else None
        
        def get_all_texts(self, selector):
            """Extract and return texts from all elements matching the selector."""
            elements = self.soup.select(selector)
            return [element.get_text(strip=True) for element in elements]
        
        def get_attribute(self, selector, attribute):
            """Extract and return attribute value from elements matching the selector."""
            element = self.soup.select_one(selector)
            return element.get(attribute) if element else None
        
        def get_all_attributes(self, selector, attribute):
            """Extract and return attribute values from all elements matching the selector."""
            elements = self.soup.select(selector)
            return [element.get(attribute) for element in elements]
        
        def find_all(self, tag, **kwargs):
            """Find all tags matching given tag and keyword arguments."""
            return self.soup.find_all(tag, **kwargs)
        
        def find_one(self, tag, **kwargs):
            """Find a single tag matching given tag and keyword arguments."""
            return self.soup.find(tag, **kwargs)
    class LangChain():
        def __init__(self,APIKEY,AILanguageModel="openai",model="gpt-3.5-turbo",azure_endpoint="",azure_openai_api_version="",AZURE_OPENAI_DEPLOYMENT_NAME=""):
            param = AILanguageModel.lower()
            if param == "openai":
                from langchain_openai import ChatOpenAI
                os.environ["OPENAI_API_KEY"] =APIKEY
                self.model = ChatOpenAI(
                    model=str(model)
                )
            elif param == "anthropic":
                from langchain_anthropic import ChatAnthropic
                os.environ["ANTHROPIC_API_KEY"] =APIKEY
                self.model = ChatAnthropic(
                    model=str(model)
                )

            elif param == "azure":
                from langchain_openai import AzureChatOpenAI
                os.environ["AZURE_OPENAI_API_KEY"] =APIKEY
                self.model = AzureChatOpenAI(
                    azure_endpoint=azure_endpoint,
                    azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME,
                    openai_api_version=azure_openai_api_version

                )
            elif param == "google":
                from langchain_google_vertexai import ChatVertexAI
                os.environ["GOOGLE_API_KEY"] =APIKEY
                self.model = ChatVertexAI(
                    model=str(model)
                )
            elif param == "cohere":
                from langchain_cohere import ChatCohere
                os.environ["COHERE_API_KEY"] =APIKEY
                self.model = ChatCohere(
                    model=str(model)
                )

            elif param == "fireworks":
                from langchain_fireworks import ChatFireworks
                os.environ["FIREWORKS_API_KEY"] =APIKEY
                self.model = ChatFireworks(
                    model=str(model)
                )

            elif param == "groq":
                from langchain_groq import ChatGroq
                os.environ["GROQ_API_KEY"] =APIKEY
                self.model = ChatGroq(
                    model=str(model)
                )

            elif param == "mistral":
                from langchain_mistralai import ChatMistralAI
                os.environ["MISTRAL_API_KEY"] =APIKEY
                self.model = ChatMistralAI(
                    model=str(model)
                )

            elif param == "together":

                from langchain_openai import ChatOpenAI
                os.environ["TOGETHER_API_KEY"] =APIKEY
                self.model = ChatOpenAI(
                base_url="https://api.together.xyz/v1",
                api_key=os.environ["TOGETHER_API_KEY"],
                model="mistralai/Mixtral-8x7B-Instruct-v0.1",
                )
            from langchain_core.messages import HumanMessage,AIMessage
            from langchain_core.runnables.history import RunnableWithMessageHistory
            from langchain_core.chat_history import (
                BaseChatMessageHistory,
                InMemoryChatMessageHistory,
            )
            self.store = {

            }

            self.with_message_history = RunnableWithMessageHistory(model, get_session_history)
        def get_session_history(self,session_id: str) -> BaseChatMessageHistory:
            if session_id not in store:
                store[session_id] = InMemoryChatMessageHistory()
            return store[session_id]

        def sendOneMessage(self,messageText,language="English"):
            out = self.model.invoke([HumanMessage(content=str(messageText))])
            response_metadata = out.response_metadata
            token_usage = response_metadata['token_usage']

            content = {
                "output":out.content,
                "completion_tokens":token_usage['completion_tokens'],
                "prompt_tokens":token_usage['prompt_tokens'],
                "total_tokens":token_usage['total_tokens']
                "modelname":response_metadata['model_name'],
                "system_fingerprint":response_metadata['system_fingerprint']

            }
            return content
        def send_multiple_messages(self,list_of_message_text):
            content = []
            for i in range(len(list_of_message_text)):
                message = list_of_message_text[i]
                AIResponse = self.sendOneMessage(message)
                content.append(HumanMessage(content=str(message)))
                content.append(AIMessage(content=AIResponse["output"]))
            out = self.model.invoke(
                content
            )
            response_metadata = out.response_metadata
            token_usage = response_metadata['token_usage']

            content = {
                "output":out.content,
                "completion_tokens":token_usage['completion_tokens'],
                "prompt_tokens":token_usage['prompt_tokens'],
                "total_tokens":token_usage['total_tokens']
                "modelname":response_metadata['model_name'],
                "system_fingerprint":response_metadata['system_fingerprint']

            }
            return content
        def run_with_history(self,text,sessionID):
            config = {"configurable": {"session_id":sessionID}}
            out = self.with_message_history.invoke(
                [HumanMessage(content=str(text))],
                config=config
            )
             content = {
                "output":out.content,
                "completion_tokens":token_usage['completion_tokens'],
                "prompt_tokens":token_usage['prompt_tokens'],
                "total_tokens":token_usage['total_tokens']
                "modelname":response_metadata['model_name'],
                "system_fingerprint":response_metadata['system_fingerprint']

            }
            return content


            
            

