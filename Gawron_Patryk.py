import typing
import pandas as pd


class GingivalFluidPredictor:

    def __init__(self):
        # TODO: Pytanie odnośnie domyślnego formatu pliku - miało być CSV ale separatorem jest tabulator
        self.data = pd.read_csv("data/2021.12.21_project_data.csv", sep="\t")

        self.gender_map = {
            "m": 0,
            "k": 1
        }

        # drop incomplete columns
        self.data.drop(columns=["Unnamed: 0",
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
                                "Interleukina – 44P"], axis=1, inplace=True)

        # change str percentage columns to floats
        self.data["API"] = self.data["API"].apply(lambda x: float(x.replace("%", "")))
        self.data["SBI"] = self.data["SBI"].apply(lambda x: float(x.replace("%", "")))

        # change gender data to numbers
        self.data["plec"] = self.data["plec"].apply(lambda x: self.gender_map[x])

        print(self.data["API"])
        print(self.data["SBI"])
        print(self.data["plec"])
        print(self.data.info())


if __name__ == '__main__':
    predictor = GingivalFluidPredictor()
