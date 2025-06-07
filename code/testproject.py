import ManualStrategy as ms
import numpy as np
import experiment1 as e1
import experiment2 as e2


def author():
    return 'ssimha31'


def gtid():
    return 903845103


def run_ms(results_file_name):
    ms.run_ms(results_file_name)


def run_exp1(results_file_name):
    e1.run_experiment1(results_file_name)


def run_exp2(results_file_name):
    e2.run_experiment2(results_file_name)


def run():
    np.random.seed(gtid())  # do this only once
    results_file_name = 'p8_results.txt'

    # erase any existing contents
    sf = open(results_file_name, 'w')
    sf.close()

    run_ms(results_file_name)
    run_exp1(results_file_name)
    run_exp2(results_file_name)

    return


if __name__ == "__main__":
    run()
