import os
import json
import uuid
import subprocess
from ..config.settings import Config

class MiningService:
    @staticmethod
    def run_miner(input_file_path):
        """
        Runs the subgraph miner on the given input file.
        Returns the parsed JSON results.
        """
        out_filename = str(uuid.uuid4()) + '.pkl'
        out_path = os.path.join(Config.RESULTS_FOLDER, out_filename)
        json_path = os.path.join(Config.RESULTS_FOLDER, out_filename.replace('.pkl', '.json'))

        try:
            # Run the miner
            cmd = [
                "python3", "-m", "subgraph_mining.decoder",
                "--dataset={}".format(input_file_path),
                "--n_trials=100", 
                "--node_anchored",
                "--out_path={}".format(out_path)
            ]
            
            print("Running command: {}".format(' '.join(cmd)))
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise Exception("Miner failed: {}".format(result.stderr))

            # Read the results
            if not os.path.exists(json_path):
                 raise Exception('Result file not found')

            with open(json_path, 'r') as f:
                mining_results = json.load(f)

            return mining_results

        finally:
            # Cleanup output files
            if os.path.exists(out_path):
                os.remove(out_path)
            if os.path.exists(json_path):
                os.remove(json_path)
