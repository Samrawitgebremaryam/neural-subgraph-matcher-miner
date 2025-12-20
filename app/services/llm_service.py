
import os
import json
import logging
import google.generativeai as genai
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class LLMService:
    _instance = None
    _patterns_cache = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMService, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize Gemini model and load patterns."""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.warning("GEMINI_API_KEY not found in environment variables.")
        else:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-2.5-flash')
        
        self._load_patterns()

    def _load_patterns(self):
        """Load patterns from the JSON file."""
        try:
            # Try to find the file in potential locations
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            json_path = os.path.join(base_dir, "results", "patterns_all_instances.json")
            
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    self._patterns_cache = json.load(f)
                logger.info(f"Loaded patterns from {json_path}")
            else:
                logger.warning(f"Patterns file not found at {json_path}")
                self._patterns_cache = []
        except Exception as e:
            logger.error(f"Failed to load patterns: {e}")
            self._patterns_cache = []

    def _get_pattern_instances(self, pattern_key: str) -> list:
        """Find instances for a specific pattern key."""
        if not self._patterns_cache:
            self._load_patterns()
        
    
        
        instances = []
        start_idx = 1 if self._patterns_cache and self._patterns_cache[0].get('type') == 'graph_context' else 0
        
        for item in self._patterns_cache[start_idx:]:
            if item.get('metadata', {}).get('pattern_key') == pattern_key:
                # This item represents the pattern group, containing instances
                return item.get('instances', [])
            
        return []

    def _find_pattern_data(self, pattern_key: str) -> Optional[Dict[str, Any]]:
        """Find the full pattern data object for a specific key."""
        if not self._patterns_cache:
            self._load_patterns()
            
        start_idx = 1 if self._patterns_cache and self._patterns_cache[0].get('type') == 'graph_context' else 0
        
        for item in self._patterns_cache[start_idx:]:
            if item.get('metadata', {}).get('pattern_key') == pattern_key:
                return item
        return None

    def analyze_motif(self, graph_data: Dict[str, Any], user_query: str, pattern_key: Optional[str] = None) -> str:
        """
        Analyze a motif using Gemini, integrating graph structure and instance context.
        Focuse is on Network Topology.
        """
        if not hasattr(self, 'model'):
             return "Error: LLM service not initialized. Check server logs for API key issues."

        context_str = ""
        num_instances = "unknown" # Default if pattern data not found
        if pattern_key:
            pattern_data = self._find_pattern_data(pattern_key)
            if pattern_data:
                metadata = pattern_data.get('metadata', {})
                # Use original_count if available (total occurrences including duplicates), otherwise fallback to count (unique)
                num_instances = metadata.get('original_count', metadata.get('count', 0))
                freq_score = metadata.get('frequency_score', 0)
                
                context_str = f"""
                CONTEXT FROM MINING RESULTS:
                - Pattern Key: {pattern_key}
                - Occurrences: {num_instances} instances found in the dataset.
                - Frequency Score: {freq_score}
                - Size: {metadata.get('size')} nodes
                - Rank: {metadata.get('rank')}
                """
                
                # Add info about a few instances to show variability or consistency
                instances = pattern_data.get('instances', [])
                if instances:
                    # Take up to 3 instances as examples
                    examples = instances[:3]
                    context_str += "\nINSTANCE EXAMPLES (for context on node/edge attributes):\n"
                    for i, inst in enumerate(examples):
                        # Extract a brief summary of attributes
                        nodes_attrs = [n.get('label', 'N/A') for n in inst.get('nodes', [])]
                        context_str += f"  Instance {i+1}: Node Labels: {nodes_attrs}\n"

        prompt = f"""
        You are an expert Graph Theory analyst.
        Your task is to interpret the provided graph motif (subgraph pattern) and answer the user's question.
        
        **CRITICAL FOCUS: NETWORK TOPOLOGY**
        **CRITICAL FOCUS: NETWORK TOPOLOGY**
        INSTRUCTION: If the user asks for a general explanation/summary of the pattern, OR asks about its frequency/count:
        Do not start with the frequency. Instead, include the statement 'This pattern occurred {num_instances} times in the sampled data.' (Do not mention the rank) in the middle of your response, for example after describing the structure.
        If the user asks specific questions about nodes/edges (e.g. "what is node 0?"), skip this frequency statement.

        Do not just list the data. Analyze the STRUCTURE based on what you see.
        - **Connectivity**: How are nodes connected? chains, stars, cycles, cliques?
        - **Topology**: Describe the topology based on the visual structure.
        - **Flow**: If directed, how does information flow? Source -> Sink?
        - **Roles**: What functions do these topological positions suggest?
        
        GRAPH DATA:
        {json.dumps(graph_data, indent=2)}
        
        {context_str}
        
        USER QUESTION: "{user_query}"
        
        Provide a concise, insightful answer focusing on the structural implications of this pattern.
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Gemini API call failed: {e}")
            return f"Error processing your request: {e}"
