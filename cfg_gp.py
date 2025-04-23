import argparse


def get_cfg():

    parser = argparse.ArgumentParser(description="")

    # 데이터 생성 관련 파라미터
    parser.add_argument("--n_ships", type=int, default=80, help="number of ships in data")

    parser.add_argument("--pop_size", type=int, default=60, help="population size")
    parser.add_argument("--min_depth", type=int, default=10, help="minimal initial random tree depth")
    parser.add_argument("--max_depth", type=int, default=15, help="maximal initial random tree depth")
    parser.add_argument("--generations", type=int, default=50, help="maximal number of generations to run evolution")
    parser.add_argument("--tournament_size", type=int, default=10, help="size of tournament for tournament selection")
    parser.add_argument("--xo_rate", type=float, default=0.8, help="crossover rate")
    parser.add_argument("--prob_mutation", type=float, default=0.2, help="per-node mutation probability")

    parser.add_argument("--data_dir", type=str, default="./input/gp/v1/28-80/", help="directory where the data for fitness calculation are stored")

    return parser.parse_args()