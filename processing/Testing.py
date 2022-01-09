import typing
import pandas as pd
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression, SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error

import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno



class GingivalFluidPredictor:

    def __init__(self):
        # TODO: Pytanie odnośnie domyślnego formatu pliku - miało być CSV ale separatorem jest tabulator
        self.data = pd.read_csv("../2021.12.21_project_data.csv", sep="\t")

        self.gender_map = {
            "m": 0,
            "k": 1
        }

        self.booleans = ["zgrzytanie", "zaciskanie", "sztywnosc", "ograniczone otwieranie", "bol miesni", "przygryzanie", "cwiczenia", "szyna", "starcie-przednie", "starcie-boczne", "ubytki klinowe", "pekniecia szkliwa", "impresje jezyka", "linea alba", "przerost zwaczy", "tkliwosc miesni"]
        self.interleukinas = ["Unnamed: 0",
                                "Interleukina – 11B",
                                "Interleukina – 11P",
                                "Interleukina – 16B",
                                "Interleukina – 16P",
                                "Interleukina – 24B",
                                "Interleukina – 24P",
                                "Interleukina – 31B",
                                "Interleukina – 31P",
                                "Interleukina – 36B",
                                "Interleukina – 36P",
                                "Interleukina – 44B",
                                "Interleukina – 44P"]
        # drop incomplete columns
        self.data.drop(columns=self.interleukinas, axis=1, inplace=True)
        # self.data.dropna(inplace=True)
        print(self.data.describe())
        print(self.data.info())
        # change str percentage columns to floats
        self.data["API"] = self.data["API"].apply(lambda x: float(x.replace("%", "")))
        self.data["SBI"] = self.data["SBI"].apply(lambda x: float(x.replace("%", "")))

        # change gender data to numbers
        self.data["plec"] = self.data["plec"].apply(lambda x: self.gender_map[x])

        # self.data.drop(columns=self.booleans, axis=1, inplace=True)

        var = 'TWI - 16 suma'
        data = pd.concat([self.data["16-B"], self.data[var]], axis=1)
        data.plot.scatter(x=var, y='16-B')




        self.y_cols = ["16-B", "16-P", "11-B", "11-P", "24-B", "24-P", "36-B", "36-P", "31-B", "31-P", "44-B", "44-P"]
        self.y = self.data["16-B"]
        self.x_16_b = self.data[["API", "SBI", "PI - 16", "GI - 16", "PPD - 16", "PPD - 16 B", "TWI - 16 suma"]]
        corrmat = self.x_16_b.corr()
        sns.heatmap(corrmat, vmax=.5, square=True, annot=True)
        # plt.show()
        self.data.drop(columns=self.y_cols, axis=1, inplace=True)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.x_16_b, self.y, test_size=0.2, random_state=42)

        self.clasifiers()

    def test_cls(self, clf, name: str):
        clf.fit(self.X_train, self.y_train)
        pred = clf.predict(self.X_test)
        print(f"{name} score: ", mean_absolute_error(self.y_test, pred))
        print([(x, y) for x, y in zip(self.y_test, pred)])

    def clasifiers(self):
        self.test_cls(LinearRegression(), "linear")

        self.test_cls(Lasso(alpha=0.01), "lasso")

        self.test_cls(Ridge(alpha=5e0), "ridge")

        self.test_cls(SVR(kernel="rbf"), "SVR")

        # self.test_cls(Pipeline([
        #     ('poly', PolynomialFeatures(degree=2)),
        #     ('line', LinearRegression())
        # ]), "pipe")

        self.test_cls(make_pipeline(StandardScaler(),
                            SGDRegressor(max_iter=1000, tol=1e-3)), "sdg")

        # self.test_cls(KNeighborsRegressor(n_neighbors=100), "knn")

        # self.test_cls(tree.DecisionTreeRegressor(), "decision tree")



if __name__ == '__main__':
    predictor = GingivalFluidPredictor()
