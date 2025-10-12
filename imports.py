from __future__ import annotations
import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd, numpy as np
import requests
from dateutil import parser as dtp
from matplotlib.ticker import PercentFormatter
import time, json
import matplotlib.dates as mdates
from datetime import *
from config import *
from utils import *
from typing import Iterable, Literal,Union, Dict, Tuple, List, Optional, Any, Callable, Mapping
from fetch.filtering import *
import csv
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter