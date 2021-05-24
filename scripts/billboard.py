import os
import urllib.request

import pandas as pd

m = None
LANG_LABEL_TO_KEEP = "__label__en"

try:
    import fasttext

    if not os.path.exists("model.ft"):
        urllib.request.urlretrieve(
            "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz",
            "model.ft",
        )
    m = fasttext.load_model("model.ft")
except:
    print("Will not detect language")


def run():
    df = pd.read_csv("data/billboardHot100_1999-2019.csv")
    first = df.sort_values(by="Week").groupby(["Artists", "Name"]).first().reset_index()
    first = first.rename({"Artists": "artist", "Lyrics": "text"}, axis=1)
    # Looks like the title of the song (or a part of it is always on the first
    # line of each lyric.
    first["text"] = first["text"].str.split("\n").str[1:].str.join("\n")
    if m is not None:
        first["lang"] = first["text"].apply(
            lambda x: m.predict(" ".join(x.split("\n")))[0][0]
        )
        print("Languages found:")
        print(first["lang"].value_counts())

        print(f"Removing everything but {LANG_LABEL_TO_KEEP}")
        is_lang = first["lang"] == LANG_LABEL_TO_KEEP
        first = first[is_lang]
        print(f"Removed {len(is_lang) - sum(is_lang)} entries")

    first.to_csv("data/billboard_cleaned.csv", index=False)


if __name__ == "__main__":
    run()
