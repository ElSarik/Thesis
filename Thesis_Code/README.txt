Tested with Python versions 3.8 & 3.9

This software is part of my bachelor thesis on Optical Character Recognition (OCR), and it is meant to function as a demonstration of how neural network models can be trained by using generated MNIST-like datasets with characters from different alphabets, and afterwards attempt to recognise what they have learnt in user-given pictures.

Features:
- Graphical User Interface (GUI)

- Model creation:
  -> Automatic dataset generation used for training based on user selection.
  -> User specified number of epochs for the training.
  -> Training information for each epoch.
  -> Representation of training information in the form of a graph upon training completion.
  -> Ability of saving the created model in .h5 form.

- Recognition:
  -> Ability of loading a created .h5 model.
  -> Ability of loading a user selected .png or .jpg image file.
  -> Recognition on user defined areas inside the selected image.
  -> Display of user defined areas together with the model's recognition on those areas.


The software is fully funtional, but it can also be considered far from perfect when it comes to its results. However I can say that it has been developed to the best of my knowledge, and at the same time I am also proud of the final outcome.
