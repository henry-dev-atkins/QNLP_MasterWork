# QNLP_MasterWork
Work completed during research for Masters Dissertation on QuantumNLP
# Files:

## DataSet.csv

Data from Mitchel and Lapata's 2-word phrase similarity dataset. Chosen words that are similar as well, such that the first and second words of each phrase are similar. This introduces a lot of bias, as this could be subjective. Has format: Word1a Word1b, Word2a, Word2b, similarity(catagorical High,Med,Low=1,0.5,0). 

## DisCoPyTutorial.ipynb

Word conducted by following: https://docs.discopy.org/en/main/notebooks/qnlp.html

## Henry Atkins Project Outline.pdf

Outline for Dissertation plan. 

## PennyLaneTutorials.ipynb

Following data encoding tutorial from Pennylane website & Lambeq. Also inplementing encoding functions for Amplitude and Angle methods.

## RECREATING_PAPER_The effect of data encoding on the expressive power of variational quantum machine learning models.ipynb

This paper https://arxiv.org/abs/2008.08605 outlines how repeated encoding can allow you to encode more expressive function spaces. 

## ReadingData.ipynb

Main novel (for now) work. Created a class to read the DataSet.csv and BERT encode the phrase, followed by a classical similarity test (cosine).
