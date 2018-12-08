from math import radians, atan, tan, sin, acos, cos
import os

#
PROJECT_DIR = '/home/bp/GitRepos/dc_destination_prediction/dc_destination_prediction/'
##
ANALYSIS_DIR = os.path.join(PROJECT_DIR, 'analysis')
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
MODEL_DIR = os.path.join(PROJECT_DIR, 'model')
SRC_DIR = os.path.join(PROJECT_DIR, 'src')
VISIUAL_DIR = os.path.join(PROJECT_DIR, 'visiual')

###
EXTRACTED_DIR = os.path.join(DATA_DIR, 'extracted')
FILTER_DIR = os.path.join(DATA_DIR, 'filter')
GENERATED_DIR = os.path.join(DATA_DIR, 'generated')

MODEL_DBSCAN_DIR = os.path.join(MODEL_DIR,'dbscan')
MODEL_RF_DIR = os.path.join(MODEL_DIR,'random_forest')


def getDistance(latA, lonA, latB, lonB):
    ra = 6378140  # radius of equator: meter
    rb = 6356755  # radius of polar: meter
    flatten = (ra - rb) / ra  # Partial rate of the earth
    # change angle to radians
    radLatA = radians(latA)
    radLonA = radians(lonA)
    radLatB = radians(latB)
    radLonB = radians(lonB)

    try:
        pA = atan(rb / ra * tan(radLatA))
        pB = atan(rb / ra * tan(radLatB))
        x = acos(sin(pA) * sin(pB) + cos(pA) * cos(pB) * cos(radLonA - radLonB))
        c1 = (sin(x) - x) * (sin(pA) + sin(pB)) ** 2 / cos(x / 2) ** 2
        c2 = (sin(x) + x) * (sin(pA) - sin(pB)) ** 2 / sin(x / 2) ** 2
        dr = flatten / 8 * (c1 - c2)
        distance = ra * (x + dr)
        return distance  # meter
    except:
        return 0.0000001


def getDistance_dbscan(pos1, pos2):
    latA, lonA = pos1
    latB, lonB = pos2
    ra = 6378140  # radius of equator: meter
    rb = 6356755  # radius of polar: meter
    flatten = (ra - rb) / ra  # Partial rate of the earth
    # change angle to radians
    radLatA = radians(latA)
    radLonA = radians(lonA)
    radLatB = radians(latB)
    radLonB = radians(lonB)

    try:
        pA = atan(rb / ra * tan(radLatA))
        pB = atan(rb / ra * tan(radLatB))
        x = acos(sin(pA) * sin(pB) + cos(pA) * cos(pB) * cos(radLonA - radLonB))
        c1 = (sin(x) - x) * (sin(pA) + sin(pB)) ** 2 / cos(x / 2) ** 2
        c2 = (sin(x) + x) * (sin(pA) - sin(pB)) ** 2 / sin(x / 2) ** 2
        dr = flatten / 8 * (c1 - c2)
        distance = ra * (x + dr)
    except:
        distance = 0.0000001
    return distance  # meter


getDistance(28.758534, 104.634470, 28.759908, 104.637853)
