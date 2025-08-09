from src.dataset.amlsim import loader as amlsim_loader
from src.dataset.berka import loader as berka_loader
from src.dataset.amlworld import loader as amlworld_loader
from src.dataset.rabo import loader as rabo_loader

def main():
    print('Rabo')
    rabo_loader.main()
    print('AMLSim')
    amlsim_loader.main()
    print('Berka')
    berka_loader.main()
    print('AMLworld')
    amlworld_loader.main()

if __name__ == '__main__':
    main()
