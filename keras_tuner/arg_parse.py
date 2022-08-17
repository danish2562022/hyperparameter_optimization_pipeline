import argparse

def get_args():

    parser = argparse.ArgumentParser(
        description="Hyper parameter tuning of models.",
        usage="-------",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )


    parser.add_argument('--number_of_layers',
                        action = 'store',
                        default = 5,
                        type = int,
                        help = "Number of layers in architectures"
                        )

    parser.add_argument('--model_type',
                        action = 'store',
                        default = 'r',
                        type= str,
                        help= 'r for regression and c for classification')

    parser.add_argument('--min_units_per_layers',
                        action = 'store',
                        default = 32,
                        type= int,
                        help= 'minimum number of units per layers')

    parser.add_argument('--max_units_per_layers',
                        action = 'store',
                        default = 512,
                        type= int,
                        help= 'minimum number of units per layers')

    parser.parse_args()
    return parser



