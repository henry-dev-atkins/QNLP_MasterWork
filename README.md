# QNLP_MasterWork
Work completed during research for Masters Dissertation on QuantumNLP. 

The goal here is to explore strategies to encode meaning vectors onto Quantum Computers. This will either be classically pre-trained word meaning vectors or corpus data for quantum NLP training. Using the optimum (to be found) method of encoding I will explore whether a quantum computer could learn meaning vectors from a pre-trained model (BERT or FastText) via variational methods. This would be evaluated by a similarity (vector dot-product) calculation between two words, as extracting the learnt vector from a quantum computer would be difficult due to superposition. In practice, phrase similarity (and learning) may be more useful due to the limited quantum resources needed to learn a large corpus. 

# Files:

## DataSet.csv

Data from Mitchel and Lapata's 2-word phrase similarity dataset [1], Table A2. The dataset extracted does not represent the entire data as we plan to conduct phrase and word level similarity comparisons. For this reason, only phrase pairs with similar words have been extracted. 

For example: *development plan, action programme* are similar phrases and the pairwise comparisons of *development, action* and *plan, programme* are also similar. This is a high similarity row so the similarity has been allocated 1 (in the dataset it is labelled as 'High'). The same process gives the phrase pair *security policy, defence minister* a similarity of 0.5 (dataset has similarity = 'Medium'), with word pairs *security, defence* and  *policy, minister* also having 0.5 similarity. This was done manually, with a general method of a) are the words immediately similar? for example most people agree that *world economy, management structure* has no similarity ([1] has it as 'Low' similarity) and importantly the word pair comparisons *world, management* and *economy, structure* have no similarity - it was labelled as 0 accordingly. However there are some words, for example *assistant manager, company director* which have been labelled by [1] as Medium phrase similarity and have 1 similar word pair - in this case *manager, director*. This was kept at a similarity of 0.5. 


| Phrase 1 <br> String Phrase1a Phrase1b  | Phrase 2 <br> String Phrase2a Phrase2b | similarity <br> (High,Med,Low = 1.0,0.5,0.0) |
|:-----|:--------:|------:|
| development plan | action programme | 1.0 |
|security policy | defence minister | 0.5 |
|world economy | management structure | 0.0 |

### DisCoPyTutorial.ipynb

Word conducted by following: https://docs.discopy.org/en/main/notebooks/qnlp.html

### Henry Atkins Project Outline.pdf

Outline for Dissertation plan. 

### PennyLaneTutorials.ipynb

Following data encoding tutorial from Pennylane website & Lambeq. Also implementing encoding functions for Amplitude and Angle methods.

### RECREATING_PAPER_The effect of data encoding on the expressive power of variational quantum machine learning models.ipynb

This paper https://arxiv.org/abs/2008.08605 outlines how repeated encoding can allow you to encode more expressive function spaces. 

### ReadingData.ipynb

Main novel (for now) work. Created a class to read the DataSet.csv and BERT encode the phrase, followed by a classical similarity test (cosine).


# References:
[1] - Mitchel and Lapata, Composition in Distributional Models of Semantics - https://onlinelibrary.wiley.com/doi/10.1111/j.1551-6709.2010.01106.x