# pr_detector
NLP model to detect when a company pivots or rebrands using blurbs (their own description of themselves) taken from their site.

The model normalises the blurb text, and then embeds them into a space where a notion of similarity can be computed. We can then use this similarity measure to identify blurbs that are worth investigating further.

To read more about the model, see:
https://www.nltk.org/_modules/nltk/stem/porter.html,
https://en.wikipedia.org/wiki/Tf%E2%80%93idf, https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html and
https://en.wikipedia.org/wiki/Cosine_similarity.

A natural extension to this repo would be to scrape a set of company's websites regularly, identify their blurb, store it and compare it to the previous scraped blurb using the functionality developed here, flagging when the threshold is breached.

See "example use case.ipynb" for an example of how one would interact with the code and "analysis.ipynb" for a runthrough of the code.

## Environment
The environment is being managed by conda with configuration stored in the 'env.yml' file. To create your environment just use `conda env create --file env.yml` from the base of the repository. To update using a new config use `conda env update --file env.yml --prune`.
For other conda tasks check out the cheat sheet @ https://docs.conda.io/projects/conda/en/latest/_downloads/843d9e0198f2a193a3484886fa28163c/conda-cheatsheet.pdf .

## Documentation

Documented in NumPy style.
