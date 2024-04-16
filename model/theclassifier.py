import pandas as pd
import pickle

class Ourmodel():
    """
    leg uit :)
    """
    def predict(self, filename):
        """
        Creates the predictions 
        This method always returns a vector of all False or all True, dependent on
        the value of self.true_or_false.

        Parameters
        ----------
        filename : string, default='competition_test.csv'
            Location of the filename to be used in the prediction. 
            This can be either a relative of an absolute path.

        Returns
        -------
        y_pred: ndarray of shape (n_samples)
            Vector containing the class labels for each sample.
        """
        #get dataframe
        df = pd.read_csv(filename) 

        #delete uitkomstkolom
        if "prognose10jaar" in df.columns.tolist():
            df.drop(["prognose10jaar"], axis=1, inplace=True)

        #delete ID
        df.drop(["Individu-ID", ], axis=1, inplace=True)

        # delete all collumns with a questionmark
        for col in df.columns.tolist():
            df = df.drop(df[df[col] == "?"].index)

        #veranderen van type naar een int
        to_int = ["opleidingsniveau", "cholesterol", "glucose", "BMI", "hartslag", "cigaretten_per_dag", "slaapscore", "onderdruk", "bovendruk"]
        df[to_int] = df[to_int].astype(int)

        #vervangen van waarden
        df_meh = df
        df_meh = df_meh.replace("CHD+", 1.0)
        df_meh = df_meh.replace("CHD-", 0.0)
        df_meh = df_meh.replace("M", 0.0)
        df_meh = df_meh.replace("V", 1.0)
        df_meh = df_meh.replace("-", 0.0)
        df_meh = df_meh.replace("+", 1.0)

        X = df_meh.iloc[:,:14]

        import pkg_resources
        modelfile = pkg_resources.resource_filename(__name__, "model.pkl")
        with open(modelfile, "rb") as file:
            model = pickle.load(file)

        yhat = model.predict(X)

        return yhat
