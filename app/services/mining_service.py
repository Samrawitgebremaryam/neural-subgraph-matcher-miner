import os
import json
import uuid
import subprocess
import shutil
from ..config.settings import Config

class MiningService:
    @staticmethod
    def run_miner(input_file_path, job_id=None, config=None):
        """
        Runs the subgraph miner on the given input file.
        Returns the parsed JSON results and file paths.
        """
        if job_id is None:
            job_id = str(uuid.uuid4())
        
        shared_job_dir = "/shared/output/{}".format(job_id)
        os.makedirs(shared_job_dir, exist_ok=True)
        
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
            
            if config.get('visualize_instances', False):
                cmd.append("--visualize_instances")
            
            print("DEBUG MINING_SERVICE: config={}".format(json.dumps(config)), flush=True)
            print("DEBUG MINING_SERVICE: Running command: {}".format(' '.join(cmd)), flush=True)
            print("Mining started - this may take several minutes...", flush=True)
            print("Job ID: {}".format(job_id), flush=True)
            
            # Use Popen to stream output in real-time
            import os as os_module
            env = os_module.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env
            )
            
            # Stream output line by line
            for line in process.stdout:
                print(line.rstrip(), flush=True)
            
            process.wait()
            
            if process.returncode != 0:
                raise Exception("Miner failed with exit code {}".format(process.returncode))

            # Read the results
            if not os.path.exists(json_path):
                 raise Exception('Result file not found')

            with open(json_path, 'r') as f:
                mining_results = json.load(f)

            shared_results_dir = os.path.join(shared_job_dir, "results")
            shared_plots_dir = os.path.join(shared_job_dir, "plots")
            os.makedirs(shared_results_dir, exist_ok=True)
            os.makedirs(shared_plots_dir, exist_ok=True)
            
            if os.path.exists(out_path):
                shutil.copy(out_path, os.path.join(shared_results_dir, "patterns.pkl"))
            if os.path.exists(json_path):
                shutil.copy(json_path, os.path.join(shared_results_dir, "patterns.json"))
            
            # Recursively copy plots/cluster directory including subdirectories
            plots_cluster_dir = "/app/plots/cluster"
            if os.path.exists(plots_cluster_dir):
                # Remove old plots first to avoid mixing old and new results
                if os.path.exists(shared_plots_dir):
                    shutil.rmtree(shared_plots_dir)
                shutil.copytree(plots_cluster_dir, os.path.join(shared_plots_dir, "cluster"))
            
            print("Results saved to shared volume: {}".format(shared_job_dir), flush=True)
            
            return {
                "motifs": mining_results,
                "job_id": job_id,
                "results_path": "/shared/output/{}/results".format(job_id),
                "plots_path": "/shared/output/{}/plots".format(job_id)
            }

        finally:
            # Cleanup temporary output files
            if os.path.exists(out_path):
                os.remove(out_path)
            if os.path.exists(json_path):
                os.remove(json_path)
