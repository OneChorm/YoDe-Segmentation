def get_keys(dic, value):
    max_label = max(dic, key=dic.get)
    # Create an empty dictionary to store the ratio of each pixel
    temp = {}
    # Stores the pixel ratio in the temp dictionary
    for key in dic.keys():
        temp[key] = (dic.get(key)) / (dic.get(max_label))
    # Find all keys in the dictionary whose values are within a range and return them as a list
    return [k for k, v in temp.items() if v > value]
