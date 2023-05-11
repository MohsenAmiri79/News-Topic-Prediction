def keep_keys(dictionary, keys):
    d = {}
    for key in keys:
        d[key] = dictionary[key]
    return d
