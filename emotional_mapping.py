import math
import numpy as np
# from : https://www.geeksforgeeks.org/check-whether-point-exists-circle-sector-not/

def getCatgoryFromSector(sector):
    switcher = {
        0: "Amusement",
        1: "Excitement",
        2: "Awe", 
        3: "Fear",
        4: "Anger",
        5: "Disgust",
        6: "Sadness",
        7: "Sadness",
        8: "Sadness",
        9: "Contentment",
        10: "Contentment",
        11: "Contentment",
    }
    return switcher.get(sector, "No emotion found for that sector")

def getEmotionFromAngle(angle):
    if angle >0 and angle <=30:
        return {"category": "Amusement", "class": 6}
    elif angle >30 and angle <=60:
        return {"category": "Excitement", "class": 0}
    elif angle >60 and angle <=90:
        return {"category": "Awe", "class": 1}
    elif angle >90 and angle <=120:
        return {"category": "Fear", "class": 2}
    elif angle >120 and angle <=150:
        return {"category": "Anger", "class": 5}
    elif angle >150 and angle <=180:
        return {"category": "Disgust", "class": 7}
    elif angle >180 and angle <=270:
        return {"category": "Sadness", "class": 3}
    elif angle >270 and angle <=360:
        return {"category": "Contentment", "class": 4}
    else:
        return f'Unvalid angle {angle} given'
   

def angle_between(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return np.rad2deg((ang1 - ang2) % (2 * np.pi))


    

def getEmotionFromPoint(valence, arousal):
    angle = angle_between((valence, arousal), (0,0))
    emotion = getEmotionFromAngle(angle)
    return emotion

def main():
    valence = -0.01
    arousal =1
    print(getEmotionFromPoint(valence, arousal))
    

if __name__ == "__main__":
    main()