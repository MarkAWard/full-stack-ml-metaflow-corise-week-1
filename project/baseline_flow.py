from metaflow import (
    FlowSpec,
    step,
    Flow,
    current,
    Parameter,
    IncludeFile,
    card,
    current,
)
from metaflow.cards import Table, Markdown, Artifact

# TODO move your labeling function from earlier in the notebook here
def labeling_function(row):
    return int(row["rating"] == 5 or (row["rating"] == 4 and row["recommended_ind"]))


class BaselineNLPFlow(FlowSpec):
    # We can define input parameters to a Flow using Parameters
    # More info can be found here https://docs.metaflow.org/metaflow/basics#how-to-define-parameters-for-flows
    split_size = Parameter("split-sz", default=0.2)
    # In order to use a file as an input parameter for a particular Flow we can use IncludeFile
    # More information can be found here https://docs.metaflow.org/api/flowspec#includefile
    data = IncludeFile("data", default="../data/Womens Clothing E-Commerce Reviews.csv")

    @step
    def start(self):
        # Step-level dependencies are loaded within a Step, instead of loading them
        # from the top of the file. This helps us isolate dependencies in a tight scope.
        import pandas as pd
        import io
        from sklearn.model_selection import train_test_split

        # load dataset packaged with the flow.
        # this technique is convenient when working with small datasets that need to move to remove tasks.
        df = pd.read_csv(io.StringIO(self.data), index_col=0)

        # filter down to reviews and labels
        df.columns = ["_".join(name.lower().strip().split()) for name in df.columns]
        df["review_text"] = df["review_text"].astype("str")
        _has_review_df = df[df["review_text"] != "nan"]
        reviews = _has_review_df["review_text"]
        labels = _has_review_df.apply(labeling_function, axis=1)
        # Storing the Dataframe as an instance variable of the class
        # allows us to share it across all Steps
        # self.df is referred to as a Data Artifact now
        # You can read more about it here https://docs.metaflow.org/metaflow/basics#artifacts
        self.df = pd.DataFrame({"label": labels, **_has_review_df})
        del df
        del _has_review_df

        # split the data 80/20, or by using the flow's split-sz CLI argument
        _df = pd.DataFrame({"review": reviews, "label": labels})
        self.traindf, self.valdf = train_test_split(_df, test_size=self.split_size)
        print(f"num of rows in train set: {self.traindf.shape[0]}")
        print(f"num of rows in validation set: {self.valdf.shape[0]}")

        self.next(self.baseline)

    @step
    def baseline(self):
        "Compute the baseline"
        import numpy as np
        from sklearn import metrics

        val_y = self.valdf["label"].values
        self.valdf["predicted"] = np.ones(len(val_y))
        majority_y = self.valdf["predicted"].values

        self.base_acc = metrics.accuracy_score(val_y, majority_y)
        self.base_rocauc = metrics.roc_auc_score(val_y, majority_y)

        self.next(self.end)

    @card(
        type="corise"
    )  # TODO: after you get the flow working, chain link on the left side nav to open your card!
    @step
    def end(self):
        msg = "Baseline Accuracy: {}\nBaseline AUC: {}"
        print(msg.format(round(self.base_acc, 3), round(self.base_rocauc, 3)))

        current.card.append(Markdown("# Womens Clothing Review Results"))
        current.card.append(Markdown("Baseline model predicting the majority label (1)"))
        current.card.append(Markdown("## Metrics"))
        current.card.append(Table(
            [
                ["Accuracy", Artifact(round(self.base_acc,3))],
                ["ROC-AUC", Artifact(round(self.base_rocauc,3))],
            ]
        ))

        current.card.append(Markdown("## Example Datasets"))
        current.card.append(Markdown("### Full DF"))
        current.card.append(Table.from_dataframe(self.df.head()))
        current.card.append(Markdown("### Train DF"))
        current.card.append(Table.from_dataframe(self.traindf.head()))
        current.card.append(Markdown("### Val DF"))
        current.card.append(Table.from_dataframe(self.valdf.head()))

        current.card.append(Markdown("## Examples of False Positives"))
        fpdf = self.valdf.loc[(self.valdf["predicted"] == 1) & (self.valdf["label"] == 0), ["review"]]
        current.card.append(Table.from_dataframe(fpdf.head()))

        current.card.append(Markdown("## Examples of False Negatives"))
        fndf = self.valdf.loc[(self.valdf["predicted"] == 0) & (self.valdf["label"] == 1), ["review"]]
        current.card.append(Table.from_dataframe(fndf.head()))


if __name__ == "__main__":
    BaselineNLPFlow()
