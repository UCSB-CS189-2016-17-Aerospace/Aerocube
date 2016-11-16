import pickle

def store(location,pickled):
    pickle.dump(pickled, open(location, "wb"))

def retrieve(location):
    return pickle.load(open(location,"rb"))

def _pickle_json(json):
    pass

def _unpickle_json(pickled):
    pass

def store_from_json(location, json):
    pickled=_pickle_json(json=json)
    _store(location=location,pickled=pickled)

def retrieve_as_json(location):
    pickled=_retrieve(location=location)
    return _unpickle_json(pickled)