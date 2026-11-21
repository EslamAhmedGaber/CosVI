"""Display names of every method and representation in the CosVI paper: the single source for figures, tables and macros.

Our method is called CosVI (cosine-coordinate variational inference).

* CosVI    = variational inference in whitened coordinates with the two-scale map: a cosine block (J = 16) carrying
             the long memory plus the whitened residual on the time grid, so the prior law is exact on the grid.
             The default, and the row the paper leads with (best held-out likelihood on N1b and on the Nile data).
* CosVI-J  = the same inference with the cosine coordinates only (J of them, e.g. CosVI-32, CosVI-64): the light
             version (smooth, fewer coordinates; needs J above the resolution limit of the observations).
* Baselines keep descriptive names. The competitor keeps its own name, MA-fBm (Daems et al., ICLR 2024); its noise
  law used under our inference is "MA-fBm law, <nodes>", its full system (their Girsanov control network) is "MA-fBm".

Change a name here only; scripts import it (`from fnvi import labels as L`).
"""

METHOD = "CosVI"
MA = "MA-fBm"


def cosvi_j(J):
    """Light version with J cosine coordinates; J may be a number or a symbol such as 'J'."""
    return f"{METHOD}-{J}"


# representation keys used in noise_maps and in the result files -> display name (all under our inference)
REPRESENTATION = {
    "twoscale": METHOD,
    "cosine": cosvi_j(32),          # N1, N1b: J = 32
    "cosine64": cosvi_j(64),        # N4 v2: J = 64
    "grid": "exact grid",
    "markov": f"{MA} law, published nodes",
    "markov_tuned": f"{MA} law, tuned nodes",
    "markov_best": f"{MA} law, best nodes",
    "brownian": "Brownian (H = 1/2)",
}

# the competitor's full system (their inference), by configuration
COMPETITOR = {"published": MA, "best": f"{MA}, best nodes"}
MODE = {"joint": "H learned jointly", "oracle": "H fixed at the truth"}

# file-name tags of run variants -> words added to the name
TAGS = {"_steps10000": "10 000 steps"}

# LaTeX forms (the manuscript defines \newcommand{\CosVI}{CosVI} and uses the macro, so the name changes in one place)
LATEX = {"CosVI": r"\CosVI", "MA-fBm": r"MA-fBm"}

# colours validated with the dataviz checker for the existing figure drafts (light mode); new keys are validated
# when the figure that needs them is drawn
COLOR = {"cosine": "#2a78d6", "markov": "#eb6834", "markov_tuned": "#1baf7a", "reference": "#1f1f1f",
         "riemann_liouville": "#8c8c8c"}


def rep_name(key):
    """Display name for a representation key, with a variant tag if present (e.g. 'grid_steps10000')."""
    for tag, words in TAGS.items():
        if key.endswith(tag):
            return f"{REPRESENTATION.get(key[:-len(tag)], key[:-len(tag)])} ({words})"
    return REPRESENTATION.get(key, key)


def competitor_name(mode, config="published", lr_h=None):
    """Display name for a run of the competitor's full system."""
    name = f"{COMPETITOR.get(config, config)} ({MODE.get(mode, mode)}"
    if lr_h is not None:
        name += f", lr for H {lr_h:g}"
    return name + ")"
