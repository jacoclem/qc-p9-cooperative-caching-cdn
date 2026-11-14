#import dei parser
import argparse
from src.parser import (
    CachingInstance,
    load_instance_by_id,
    load_instances,
    load_instances_by_block,
)

#parser per gli argomenti
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Risolutore per il problema del caching cooperativo per i CDN."
    )
    
    # Gruppo mutualmente esclusivo tra id o blocco
    group_loader = parser.add_mutually_exclusive_group()
    group_loader.add_argument(
        "--load-by-id",
        type=str,
        help="Carica una singola istanza tramite il suo ID (es. P09-00).",
    )
    group_loader.add_argument(
        "--load-by-block",
        type=int,
        choices=[1, 2, 3, 4, 5],
        help="Carica tutte le istanze appartenenti a un blocco (1..5).",
    )
    return parser.parse_args()

def main() ->  None:
    args = parse_args()
    if args.load_by_id:
        instances = [load_instance_by_id(args.load_by_id)]
    elif args.load_by_block:
        instances = load_instances_by_block(args.load_by_block)
    else:
        instances = load_instances()

    print(f"Numero istanze caricate: {len(instances)}")


if __name__ == "__main__":
    main()