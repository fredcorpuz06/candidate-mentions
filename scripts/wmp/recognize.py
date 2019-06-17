'''Recognize text, image and video references to people.

This module contains methods for detecting all text, image and video
references of known people. Text references are found through an string input
of identifiers for a person and regular expressions in the SpaCy package. 
Image and video references are found through an image input of a portrait of
the person and face matching methods in the face_recognition package.

The module structure is the following:

- The ``NameFinder`` class

- The ``FaceRecognizer`` base class

- The ``VideoAnnotator`` class 
'''

# Authors: Frederick Corpuz