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

def getEmotionFromAngle(angle, category=0):
    if angle >0 and angle <=30:
        return "Amusement" if category else 6
    elif angle >30 and angle <=60:
        return "Excitement" if category else 0
    elif angle >60 and angle <=90:
        return "Awe" if category else 1
    elif angle >90 and angle <=120:
        return "Fear" if category else 2
    elif angle >120 and angle <=150:
        return "Anger" if category else 5
    elif angle >150 and angle <=180:
        return "Disgust" if category else 7
    elif angle >180 and angle <=270:
        return "Sadness" if category else 3
    elif angle >270 and angle <=360:
        return "Contentment" if category else 4
    else:
        return f'Unvalid angle {angle} given'
   

def angle_between(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return np.rad2deg((ang1 - ang2) % (2 * np.pi))



def getEmotionFromPoint(valence, arousal, cat=0):
    angle = angle_between((valence, arousal), (0,0))
    emotion = getEmotionFromAngle(angle, cat)
    return emotion

def getEmotionListFromPointList(arousal_values, valence_values):
    assert len(arousal_values) == len(valence_values), "Valence and arousal lists must be same lenght with index one-to-one"
    emotion_list = []
    for a,v in zip(arousal_values, valence_values):
        emotion = getEmotionFromPoint(v,a, cat=0)
        emotion_list.append(emotion)
    return emotion_list

def getEmotionDistribution(emotion_list, normalize=0):
    distribution = np.zeros(8,  dtype=np.float32)

    if normalize:
        amin, amax = min(distribution), max(distribution)
        for i, val in enumerate(distribution):
            distribution[i] = (val-amin) / (amax-amin)
        return distribution.tolist()
    
    for l in emotion_list:
        distribution[l] += 1
    total = sum(distribution)
    print(total)
    dist = []
    for element in distribution:
        percentage = element / total
        dist.append(percentage)
    return dist

  


def main():
    valence = 1
    arousal = 0.5
    # print(getEmotionFromPoint(valence, arousal, 1))
    a_list = [-0.5, -0.5, 0.6, -0.1, 0.6, 0.6, -0.7, 0.7, -0.5, -0.5, 0.6, -0.1, 0.6, 0.6, -0.7, 0.7]
    v_list = [0, 0.5, 0.6, 0.6, 1, 0.6, 0.1, 0.7, -0.5, -0.5, 0.6, -0.1, 0.6, 0.6, -0.7, 0.7]
    

    emotion_lsit = getEmotionListFromPointList(a_list, v_list)

    print(emotion_lsit)

    dist = getEmotionDistribution(emotion_lsit, normalize=0)
    print(dist)




if __name__ == "__main__":
    main()