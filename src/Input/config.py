"""
Input Experiment Configuration 

Defines the 4 dimensions of input space for idiomaticity detection experiments:
1. Context patterns - which parts of the sentence to include
2. MWE inclusion - whether to include MWE as separate input feature
3. MWE transformation - how to represent the MWE in the context
4. Linguistic features - additional features to include (NER, glosses)

"""

import argparse

#  Configuration options
CONTEXT_OPTIONS = [
    "target",
    "previous_target",
    "target_next",
    "previous_target_next",
]

TRANSFORMATION_OPTIONS = [
    "plain",
    "mask",
    "highlight",
]

FEATURE_OPTIONS = [
    "none",
    "ner",
    "glosses",
    "ner_glosses",
]


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Configure a single preprocessing setup"
    )

    parser.add_argument(
        "--context",
        required=True,
        choices=CONTEXT_OPTIONS,
        help="Context configuration",
    )

    parser.add_argument(
        "--mwe_inclusion",
        required=True,
        choices=["True", "False"],
        help="Include MWE as separate feature",
    )

    parser.add_argument(
        "--transformation",
        required=True,
        choices=TRANSFORMATION_OPTIONS,
        help="MWE transformation type",
    )

    parser.add_argument(
        "--features",
        required=True,
        choices=FEATURE_OPTIONS,
        help="Linguistic features to include",
    )
    args, _ = parser.parse_known_args()
    return args

# Build configuration dictionary from arguments
def get_config():
    args = parse_arguments()
    

    include_mwe = args.mwe_inclusion == "True"

    config = {
        "context": args.context,
        "include_mwe": include_mwe,
        "transformation": args.transformation,
        "features": args.features,
        "name": build_name(
            args.context,
            include_mwe,
            args.transformation,
            args.features,
        ),
    }

    return config


def build_name(context, include_mwe, transformation, features):
    mwe_label = "with_mwe" if include_mwe else "no_mwe"
    return f"{context}_{mwe_label}_{transformation}_{features}"


if __name__ == "__main__":
    config = get_config()
    print("\nSelected Configuration:")
    for k, v in config.items():
        print(f"{k}: {v}")
