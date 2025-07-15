from addict import Dict
import torch
from torch.nn import Module
import waitress
from ..data import Transform, full_neighbor_list
from .trainer import DTYPE_MAPPING, _load_state_dict
from .batch import _decorate_batch_input, _to_device, _decorate_batch_output


class Server:
    def __init__(self, config: Dict, model: Module, model_path: str, out_dir: str, transform: Transform):
        self.neighbor_list_type = config.Server.get("neighbor_list", "full")
        self.cuda = config.Simulation.get('cuda', False)
        self.dtype = DTYPE_MAPPING[config.Simulation.get("dtype", "float64")]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and self.cuda else "cpu")
        # single ff simulation
        self.model = model.to(self.device).type(self.dtype)
        _load_state_dict(model, self.device, model_path, inference=True)
        self.model.eval()
        self.calculator = None
        self.out_dir = out_dir
        
    def calculate(self, info):
        features = info.get("features", None)
        if features is None:
            return {}
        if features["N"] is None:
            features["N"] = len(features["Ra"])
        if self.neighbor_list_type == "full":
            idx_i, idx_j = full_neighbor_list(features["N"])
            features["idx_i"] = idx_i
            features["idx_j"] = idx_j
            features["N_pair"] = len(idx_i)
        net_input, _ = _decorate_batch_input(
            batch=[(features, None)],
            device=self.device,
            dtype=self.dtype
        )
        net_input, _ = _to_device((net_input, {}), self.device)
        net_output = self.model(net_input)
        output, _ = _decorate_batch_output(
            output=net_output,
            features=net_input,
            targets=None
        )
        self.transform.inverse_transform(output)
        return output


def run_predict():
    """
    Runs an enerzyme calculation.
    Expects a JSON payload, which can be deserialized directly as kwargs to AIMNet2Calculator, i.e.:
    {
        "data": {
            "Ra": list[list[tuple[float, float, float]]],
            "Za": list[list[int]],
            "Q": list[list[float]],
            "S": list[list[int]],
        }
    }
    Returns JSON with energy and flattened gradient in a.u.
    """
    input = request.get_json()

    # Set the number of torch threads
    nthreads = input.pop('nthreads', 1)
    torch.set_num_threads(nthreads)

    # Get the initialized AIMNet2Calculator
    # Since the object is not thread-safe, we initialize one per server thread
    thread_id = threading.get_ident()
    global calculators
    if thread_id not in calculators:
        calculators[thread_id] = calculator.init(model=model)
    calc = calculators[thread_id]

    # run the calculation
    result = calc(**input)

    # get the output
    energy, gradient = common.process_output(result)

    return jsonify({'energy': energy, 'gradient': gradient})


def run(arglist: list[str]):
    """Start the AIMNet2 calculation server using a specified model file."""
    args = common.cli_parse(arglist, mode=common.RunMode.Server)

    # get the absolute path of the model file as a plain string
    global model
    model = str(args.model_dir / args.model)

    # set up logging
    logger = logging.getLogger('waitress')
    logger.setLevel(logging.DEBUG)

    # start the server
    waitress.serve(app, listen=args.bind, threads=args.nthreads)


def main():
    """Entry point for CLI execution"""
    run(sys.argv[1:])


if __name__ == '__main__':
    main()