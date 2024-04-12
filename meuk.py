import pandas as pd
import pickle

class Ourmodel():
    """
    leg uit :)
    """

    def predict(self, filename):
        df = pd.read_csv(filename) 

        # alle voorbewerking stappen

        X = data.iloc[:,:-1]

        import pkg_resources
        modelfile = pkg_resources.resource_filename(__name__, "model.pkl")

        with open(modelfile, "rb") as file:
            model = pickle.load(file)

        yhat = model.predict(X)

        return yhat
    
