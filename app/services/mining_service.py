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
        
        if config is None:
            config = {}
        
        shared_job_dir = "/shared/output/{}".format(job_id)
        os.makedirs(shared_job_dir, exist_ok=True)
        
        # Clean plots directory to prevent old results from mixing with new ones
        plots_cluster_dir = "/app/plots/cluster"
        if os.path.exists(plots_cluster_dir):
            shutil.rmtree(plots_cluster_dir)
        os.makedirs(plots_cluster_dir, exist_ok=True)
        
        out_filename = str(uuid.uuid4()) + '.pkl'
        out_path = os.path.join(Config.RESULTS_FOLDER, out_filename)
        json_path = os.path.join(Config.RESULTS_FOLDER, out_filename.replace('.pkl', '.json'))

        try:
            # Build command with config parameters
            cmd = [
                "python3", "-m", "subgraph_mining.decoder",
                "--dataset={}".format(input_file_path),
                "--n_trials={}".format(config.get('n_trials', 100)),
                "--min_pattern_size={}".format(config.get('min_pattern_size', 5)),
                "--max_pattern_size={}".format(config.get('max_pattern_size', 10)),
                "--min_neighborhood_size={}".format(config.get('min_neighborhood_size', 5)),
                "--max_neighborhood_size={}".format(config.get('max_neighborhood_size', 10)),
                "--n_neighborhoods={}".format(config.get('n_neighborhoods', 2000)),
                "--radius={}".format(config.get('radius', 3)),
                "--graph_type={}".format(config.get('graph_type', 'undirected')),
                "--search_strategy={}".format(config.get('search_strategy', 'greedy')),
                "--sample_method={}".format(config.get('sample_method', 'tree')),
                "--node_anchored",
                "--out_path={}".format(out_path)
            ]
            
            if config.get('visualize_instances', False):
                cmd.append("--visualize_instances")
            
            print("Running command: {}".format(' '.join(cmd)), flush=True)
            print("Mining started - this may take several minutes...", flush=True)
            print("Job ID: {}".format(job_id), flush=True)
            print("Config: {}".format(json.dumps(config, indent=2)), flush=True)
            
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
            total_chunks = 1
            current_chunk = 0
            
            progress_file = os.path.join(shared_job_dir, "progress.json")
            
            def update_progress(status, progress, message):
                with open(progress_file, 'w') as f:
                    json.dump({
                        "status": status,
                        "progress": min(progress, 99), # Never hit 100 until fully done
                        "message": message
                    }, f)

            # Initialize progress
            update_progress("starting", 0, "Initializing miner...")

            for line in process.stdout:
                line_str = line.rstrip()
                print(line_str, flush=True)
                
                try:
                    # Robust parsing that ignores timestamp prefixes
                    # Example: "[10:00:00] Worker PID 123 finished chunk 1/4"
                    
                    if "started chunk" in line_str:
                         # "... started chunk 1/4"
                        parts = line_str.split("started chunk")[1].strip().split(" ")[0] # "1/4"
                        current, total = map(int, parts.split("/"))
                        total_chunks = total
                        current_chunk = current
                        
                        # Start of a chunk is roughly (chunk-1)/total
                        base_progress = int(((current_chunk - 1) / total_chunks) * 90)
                        update_progress("mining", base_progress, f"Started processing chunk {current_chunk} of {total_chunks}...")

                    elif "still processing chunk" in line_str:
                        # Bump progress slightly to show activity
                        # "... still processing chunk 1/4"
                        parts = line_str.split("still processing chunk")[1].strip().split(" ")[0]
                        current, total = map(int, parts.split("/"))
                        
                        base_progress = int(((current_chunk - 1) / total_chunks) * 90)
                        active_progress = base_progress + int((1 / total_chunks) * 45) # Halfway through chunk
                        update_progress("mining", active_progress, f"Still processing chunk {current_chunk} of {total_chunks}...")

                    elif "finished chunk" in line_str:
                        # "... finished chunk 1/4"
                        parts = line_str.split("finished chunk")[1].strip().split(" ")[0]
                        current, total = map(int, parts.split("/"))
                        
                        # End of chunk is current/total
                        completed_progress = int((current_chunk / total_chunks) * 90)
                        update_progress("mining", completed_progress, f"Finished chunk {current_chunk} of {total_chunks}")
                        
                except Exception as e:
                    # Don't let parsing errors stop the stream
                    print(f"Warning: Failed to parse progress line: {e}", flush=True)

            process.wait()
            
            # Final completion update
            update_progress("completed", 100, "Mining completed successfully!")
            
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
