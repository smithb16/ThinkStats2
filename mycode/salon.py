## Module import and boilerplate
import numpy as np
import matplotlib.pyplot as plt

## Function definition
def Sim(p, clients, stylists):
    """Performs random simulation of disease transmission
    between stylist and n clients with transmission probability 'p'
    args:
        p: float probability
        clients: int number of clients
        stylists: int number of stylists
    returns:
        1: disease transmission
        0: no disease transmission
    """
    for _ in range(stylists):
        sample = np.random.random_sample(clients)
        if np.any(sample < p):
            return 1

    return 0

if __name__ == '__mai_':
    """Main scripts here"""
    ## Main scripts here
    ps = np.logspace(-5,0,num=10)
    clients = 70
    stylists = 2
    sims = 1000
    fracs = []
    for p in ps:
        total = 0
        for _ in range(sims):
            total += Sim(p, clients, stylists)
        fracs.append(total / sims)

    ## Print the results
    print('Disease transmission p \t % Chance of repeating these results')
    for p, f in zip(ps, fracs):
        print('%0.5f \t\t\t\t %0.1f' %(p,(1-f)*100))
