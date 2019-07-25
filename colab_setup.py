# quick script to get environment setup in google colab notebook

from IPython import get_ipython
ipython = get_ipython()

def R(cmd):
    ipython.run_line_magic("R", "", cmd)

def bash(cmd):
    ipython.run_cell_magic("bash", "", cmd)


def setup():
    R("install.packages('imputeTS')")
    bash("pip install tensorboardX")


if __name__ == "__main__":
    setup()
