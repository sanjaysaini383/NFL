# Engine vs. Expression — NFL Big Data Bowl 2027

> Does the Combine engine show up on Sundays?

An entry for the [NFL Big Data Bowl 2027](https://www.kaggle.com/competitions/nfl-big-data-bowl-2027), which links 10 Hz NFL Combine tracking to in-game tracking and NFL success.

*Work in progress.*

## Data & license

The competition data is licensed **CC BY-NC 4.0** and is **not** included in this repository. Download it from Kaggle into `data/raw/`:

```bash
kaggle competitions download -c nfl-big-data-bowl-2027 -p data/raw && unzip data/raw/*.zip -d data/raw
```

`src/combine2field/synthetic.py` generates **synthetic** files with the same schema for tests and the app's demo mode. They contain no real measurements.
