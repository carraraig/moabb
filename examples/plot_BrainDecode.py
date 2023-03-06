"""
=====================================================
Example of usage of BrainDecode with MOABB evaluation
=====================================================
This example shows how to use BrainDecode in combination with MOABB evaluation.
In this example we use the architecture ShallowFBCSPNet.
"""
# Authors: Igor Carrara <igor.carrara@inria.fr>
#          Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)

import matplotlib.pyplot as plt
import mne
import torch
from braindecode import EEGClassifier
from braindecode.models import ShallowFBCSPNet
from sklearn.pipeline import Pipeline
from skorch.callbacks import EarlyStopping, EpochScoring
from skorch.dataset import ValidSplit

from moabb.analysis.plotting import score_plot
from moabb.datasets import BNCI2014001
from moabb.evaluations import CrossSessionEvaluation
from moabb.paradigms import MotorImagery
from moabb.pipelines.utilis_pytorch import Transformer, set_seed


mne.set_log_level(False)

# Print Information PyTorch
print(f"Tensorflow Version: {torch.__version__}")

# Set up GPU if it is there
cuda = torch.cuda.is_available()
device = "cuda" if cuda else "cpu"
print("GPU is", "AVAILABLE" if cuda else "NOT AVAILABLE")


###############################################################################
# In this example, we will use only the dataset ``BNCI2014001``.
#
# Running the benchmark
# ---------------------
#
# This example use the CrossSession evaluation procedure. We focus on the dataset BNCI2014001 and only on 1 subject
# to reduce the computational time.
#
# To keep the computational time low the number of epoch is reduced. In a real situation we suggest to use
# EPOCH = 1000
# PATIENCE = 300
#
# This code is implemented to run on CPU. If you're using a GPU, do not use multithreading
# (i.e. set n_jobs=1)


# Set random seed to be able to reproduce results
set_seed(42)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Hyperparameter
LEARNING_RATE = 0.0625 * 0.01  # result taken from BrainDecode
WEIGHT_DECAY = 0  # result taken from BrainDecode
BATCH_SIZE = 64  # result taken from BrainDecode
EPOCH = 10
PATIENCE = 3
fmin = 4
fmax = 100
tmin = 0
tmax = None

# Load the dataset
dataset = BNCI2014001()
events = ["right_hand", "left_hand"]
paradigm = MotorImagery(
    events=events, n_classes=len(events), fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax
)
subjects = [1]
X, _, _ = paradigm.get_data(dataset=dataset, subjects=subjects)
# Define Transformer of Dataset compatible with Brain Decode
create_dataset = Transformer()

##############################################################################
# Create Pipelines
# ----------------
# In order to create a pipeline we need to load a model from BrainDecode.
# the second step is to define a skorch model using EEGClassifier from BrainDecode
# that allow to convert the PyTorch model in a scikit-learn classifier.

model = ShallowFBCSPNet(
    in_chans=X.shape[1],
    n_classes=len(events),
    input_window_samples=X.shape[2],
    final_conv_length="auto",
)

# Send model to GPU
if cuda:
    model.cuda()

# Define a Skorch classifier
clf = EEGClassifier(
    model,
    criterion=torch.nn.CrossEntropyLoss,
    optimizer=torch.optim.Adam,
    optimizer__lr=LEARNING_RATE,
    batch_size=BATCH_SIZE,
    max_epochs=EPOCH,
    train_split=ValidSplit(0.2),
    device=device,
    callbacks=[
        EarlyStopping(monitor="valid_loss", patience=PATIENCE),
        EpochScoring(
            scoring="accuracy", on_train=True, name="train_acc", lower_is_better=False
        ),
        EpochScoring(
            scoring="accuracy", on_train=False, name="valid_acc", lower_is_better=False
        ),
    ],
    verbose=1,  # Not printing the results foe each epoch
)

# Create the pipelines
pipes = {}
pipes["ShallowFBCSPNet"] = Pipeline(
    [("Braindecode_dataset", create_dataset), ("Net", clf)]
)


##############################################################################
# Evaluation
# ----------
dataset.subject_list = dataset.subject_list[:2]

evaluation = CrossSessionEvaluation(
    paradigm=paradigm,
    datasets=dataset,
    suffix="braindecode_example",
    overwrite=True,
    return_epochs=True,
    n_jobs=1,
)

results = evaluation.process(pipes)

print(results.head())

##############################################################################
# Plot Results
# ----------------
score_plot(results)
plt.show()