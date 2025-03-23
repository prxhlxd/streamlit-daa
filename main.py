import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict

# Set page configuration
st.set_page_config(page_title="Maximal Clique Analyzer", layout="wide")

# Initialize session state to keep track of algorithm progress
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.current_step = 0
    st.session_state.all_steps = []
    st.session_state.maximal_cliques = []
    st.session_state.graph = None
    st.session_state.pos = None
    st.session_state.call_stack = []
    st.session_state.page = "reports"  # Default page

# Add navigation in sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Algorithm Reports", "Algorithm Visualization"], index=0)

# Update session state with selected page
if page == "Algorithm Visualization":
    st.session_state.page = "visualization"
else:
    st.session_state.page = "reports"

# Functions for algorithm visualization
def parse_edge_list(edge_text):
    """Parse the edge list from input text."""
    edges = []
    for line in edge_text.strip().split('\n'):
        if line.strip():
            u, v = map(int, line.strip().split())
            edges.append((u, v))
    return edges

def build_graph(edges):
    """Build a NetworkX graph from edge list."""
    G = nx.Graph()
    for u, v in edges:
        G.add_edge(u, v)
    return G

def initialize_algorithm(G):
    """Initialize the algorithm data structures."""
    steps = []
    
    # Initial state
    steps.append({
        'description': "Algorithm starts with an empty clique C = {} and all vertices as candidates.",
        'current_clique': set(),
        'candidates': set(G.nodes()),
        'action': 'initialize',
        'maximality_check': None,
        'maximality_result': None,
        'call_level': 0
    })
    
    return steps

def simulate_update(G, initial_C=None, initial_candidates=None, call_level=0):
    """Simulate the UPDATE function to generate steps."""
    if initial_C is None:
        initial_C = set()
    if initial_candidates is None:
        initial_candidates = set(G.nodes())
        
    steps = []
    
    # Add initial state for this call
    steps.append({
        'description': f"Entering UPDATE with C = {initial_C} and candidates = {initial_candidates}",
        'current_clique': initial_C.copy(),
        'candidates': initial_candidates.copy(),
        'action': 'enter_update',
        'maximality_check': None,
        'maximality_result': None,
        'call_level': call_level
    })
    
    # Empty candidates check
    if not initial_candidates:
        maximality_result = is_maximal(G, initial_C)
        steps.append({
            'description': f"Candidates empty. Testing if C = {initial_C} is maximal.",
            'current_clique': initial_C.copy(),
            'candidates': set(),
            'action': 'maximality_test',
            'maximality_check': initial_C.copy(),
            'maximality_result': maximality_result,
            'call_level': call_level
        })
        
        if maximality_result:
            steps.append({
                'description': f"C = {initial_C} is a maximal clique!",
                'current_clique': initial_C.copy(),
                'candidates': set(),
                'action': 'found_clique',
                'maximality_check': None,
                'maximality_result': None,
                'call_level': call_level
            })
        else:
            steps.append({
                'description': f"C = {initial_C} is not maximal.",
                'current_clique': initial_C.copy(),
                'candidates': set(),
                'action': 'not_maximal',
                'maximality_check': None,
                'maximality_result': None,
                'call_level': call_level
            })
            
        steps.append({
            'description': f"Returning from UPDATE with C = {initial_C}",
            'current_clique': initial_C.copy(),
            'candidates': set(),
            'action': 'return',
            'maximality_check': None,
            'maximality_result': None,
            'call_level': call_level
        })
        return steps
    
    # Process each candidate
    candidates = list(initial_candidates)
    while candidates:
        v = candidates[0]
        candidates.remove(v)
        remaining_candidates = set(candidates)
        
        steps.append({
            'description': f"Selected vertex {v} from candidates.",
            'current_clique': initial_C.copy(),
            'candidates': remaining_candidates.copy(),
            'action': 'select_vertex',
            'selected_vertex': v,
            'maximality_check': None,
            'maximality_result': None,
            'call_level': call_level
        })
        
        # Add v to C
        new_C = initial_C.copy()
        new_C.add(v)
        
        steps.append({
            'description': f"Added vertex {v} to clique C, new C = {new_C}",
            'current_clique': new_C.copy(),
            'candidates': remaining_candidates.copy(),
            'action': 'add_to_clique',
            'selected_vertex': v,
            'maximality_check': None,
            'maximality_result': None,
            'call_level': call_level
        })
        
        # Find new candidates that are adjacent to v
        new_candidates = set()
        for u in remaining_candidates:
            if G.has_edge(v, u):
                new_candidates.add(u)
        
        steps.append({
            'description': f"Filtered candidates to those adjacent to {v}, new candidates = {new_candidates}",
            'current_clique': new_C.copy(),
            'candidates': new_candidates.copy(),
            'action': 'filter_candidates',
            'selected_vertex': v,
            'maximality_check': None,
            'maximality_result': None,
            'call_level': call_level
        })
        
        # Recursive call
        recursive_steps = simulate_update(G, new_C, new_candidates, call_level + 1)
        steps.extend(recursive_steps)
        
    steps.append({
        'description': f"No more candidates to process. Returning from UPDATE with C = {initial_C}",
        'current_clique': initial_C.copy(),
        'candidates': set(),
        'action': 'return',
        'maximality_check': None,
        'maximality_result': None,
        'call_level': call_level
    })
    
    return steps

def is_maximal(G, C):
    """Check if a clique is maximal."""
    if not C:
        return False
    
    # Check if C is actually a clique
    for u in C:
        for v in C:
            if u != v and not G.has_edge(u, v):
                return False
    
    # Check if any vertex outside C can be added
    for y in G.nodes():
        if y not in C:
            can_add = True
            for x in C:
                if not G.has_edge(x, y):
                    can_add = False
                    break
            if can_add:
                return False  # Not maximal
    
    return True  # Is maximal

def find_maximal_cliques(G):
    """Find all maximal cliques in the graph."""
    maximal_cliques = []
    
    def update(C, candidates):
        if not candidates:
            if is_maximal(G, C):
                maximal_cliques.append(C.copy())
            return
        
        candidates_copy = candidates.copy()
        while candidates_copy:
            v = next(iter(candidates_copy))
            candidates_copy.remove(v)
            
            new_C = C.copy()
            new_C.add(v)
            
            new_candidates = set()
            for u in candidates_copy:
                if G.has_edge(v, u):
                    new_candidates.add(u)
            
            update(new_C, new_candidates)
    
    update(set(), set(G.nodes()))
    return maximal_cliques

# Define the Algorithm Visualization Page
def show_visualization_page():
    st.title("Maximal Clique Algorithm Visualization")
    
    # Graph Input
    st.sidebar.header("Graph Input")
    
    # Default edge list
    default_edges = """0 1
1 2
0 2
1 3
1 4
2 4
2 3
3 4
6 7
7 8
8 9
9 6
6 8
7 9"""
    
    edge_list_text = st.sidebar.text_area("Edge List (one edge per line)", value=default_edges, height=200)
    
    if st.sidebar.button("Initialize Algorithm"):
        # Parse edge list
        edges = parse_edge_list(edge_list_text)
        
        # Build graph
        G = build_graph(edges)
        
        # Generate positions for visualization
        pos = nx.spring_layout(G, seed=42)
        
        # Generate all steps
        all_steps = initialize_algorithm(G)
        all_steps.extend(simulate_update(G)[1:])  # Skip the first step which is duplicate
        
        # Find all maximal cliques
        maximal_cliques = find_maximal_cliques(G)
        
        # Save to session state
        st.session_state.initialized = True
        st.session_state.current_step = 0
        st.session_state.all_steps = all_steps
        st.session_state.maximal_cliques = maximal_cliques
        st.session_state.graph = G
        st.session_state.pos = pos
        st.session_state.total_steps = len(all_steps)
    
    # Step controls
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col1:
        if st.session_state.initialized and st.session_state.current_step > 0:
            if st.button("Previous Step"):
                st.session_state.current_step -= 1
    
    with col2:
        if st.session_state.initialized:
            st.progress(st.session_state.current_step / max(1, len(st.session_state.all_steps) - 1))
            step_text = f"Step {st.session_state.current_step + 1}/{len(st.session_state.all_steps)}"
            st.markdown(f"<div style='text-align: center'>{step_text}</div>", unsafe_allow_html=True)
    
    with col3:
        if st.session_state.initialized and st.session_state.current_step < len(st.session_state.all_steps) - 1:
            if st.button("Next Step"):
                st.session_state.current_step += 1
    
    # Visualization area
    if st.session_state.initialized:
        col_graph, col_status = st.columns([2, 1])
        
        with col_graph:
            G = st.session_state.graph
            pos = st.session_state.pos
            step = st.session_state.all_steps[st.session_state.current_step]
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Draw graph
            nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax)
            
            # Get node colors based on current state
            current_clique = step['current_clique']
            candidates = step['candidates']
            
            # Track position of nodes in different sets for label placement
            clique_nodes = []
            candidate_nodes = []
            other_nodes = []
            
            for node in G.nodes():
                if node in current_clique:
                    clique_nodes.append(node)
                elif node in candidates:
                    candidate_nodes.append(node)
                else:
                    other_nodes.append(node)
            
            # Draw nodes with different colors
            nx.draw_networkx_nodes(G, pos, nodelist=clique_nodes, 
                                    node_color='green', node_size=700, alpha=0.8, ax=ax)
            nx.draw_networkx_nodes(G, pos, nodelist=candidate_nodes, 
                                    node_color='yellow', node_size=500, alpha=0.8, ax=ax)
            nx.draw_networkx_nodes(G, pos, nodelist=other_nodes, 
                                    node_color='gray', node_size=500, alpha=0.5, ax=ax)
            
            # Highlight selected vertex if applicable
            if 'selected_vertex' in step and step['selected_vertex'] is not None:
                nx.draw_networkx_nodes(G, pos, nodelist=[step['selected_vertex']], 
                                        node_color='red', node_size=700, alpha=0.8, ax=ax)
            
            # Highlight maximality check if applicable
            if step['maximality_check'] is not None:
                check_edges = []
                check_nodes = list(step['maximality_check'])
                
                # Get all edges in the potential clique
                for i, u in enumerate(check_nodes):
                    for v in check_nodes[i+1:]:
                        if G.has_edge(u, v):
                            check_edges.append((u, v))
                
                nx.draw_networkx_edges(G, pos, edgelist=check_edges, width=3, 
                                        edge_color='green', ax=ax)
                
                # If not maximal, highlight the vertex that could be added
                if not step['maximality_result']:
                    for y in G.nodes():
                        if y not in step['maximality_check']:
                            can_add = True
                            for x in step['maximality_check']:
                                if not G.has_edge(x, y):
                                    can_add = False
                                    break
                            if can_add:
                                nx.draw_networkx_nodes(G, pos, nodelist=[y], 
                                                        node_color='purple', node_size=700, 
                                                        alpha=0.8, ax=ax)
                                # Draw edges to show why it can be added
                                connecting_edges = [(y, x) for x in step['maximality_check'] if G.has_edge(y, x)]
                                nx.draw_networkx_edges(G, pos, edgelist=connecting_edges, 
                                                        width=2.5, edge_color='purple', 
                                                        style='dashed', ax=ax)
                                break
            
            # Draw node labels
            nx.draw_networkx_labels(G, pos, ax=ax)
            
            # Add legend
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=15, label='In Clique C'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', markersize=15, label='Candidates'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=15, label='Selected Vertex'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=15, label='Other Vertices'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=15, label='Could Be Added (Not Maximal)')
            ]
            ax.legend(handles=legend_elements, loc='upper right')
            
            plt.title("Maximal Clique Algorithm Visualization")
            plt.axis('off')
            st.pyplot(fig)
        
        with col_status:
            st.subheader("Algorithm State")
            
            # Display current clique
            st.markdown(f"**Current Clique C:** {{{', '.join(map(str, step['current_clique']))}}}")
            
            # Display candidates
            st.markdown(f"**Candidates:** {{{', '.join(map(str, step['candidates']))}}}")
            
            # Display call level
            st.markdown(f"**Recursion Depth:** {step['call_level']}")
            
            # Display step description
            st.subheader("Current Step")
            st.markdown(step['description'])
            
            # Display action type
            action_types = {
                'initialize': "Initializing algorithm",
                'enter_update': "Entering UPDATE function",
                'maximality_test': "Testing for maximality",
                'found_clique': "Found maximal clique!",
                'not_maximal': "Clique is not maximal",
                'select_vertex': "Selecting a vertex",
                'add_to_clique': "Adding vertex to clique",
                'filter_candidates': "Filtering candidates",
                'return': "Returning from function call"
            }
            
            st.markdown(f"**Action:** {action_types.get(step['action'], step['action'])}")
            
            # Display all found maximal cliques
            st.subheader("Final Results")
            st.markdown(f"**Maximal Cliques Found:** {len(st.session_state.maximal_cliques)}")
            
            clique_df = pd.DataFrame({
                'Clique': [f"{{{', '.join(map(str, clique))}}}" for clique in st.session_state.maximal_cliques]
            })
            st.dataframe(clique_df)
    
    # Display instructions if not initialized
    if not st.session_state.initialized:
        st.info("Enter the edge list in the sidebar and click 'Initialize Algorithm' to begin the visualization.")
        
        st.markdown("""
        ## How to use this visualization:
        
        1. Enter the edge list in the format "u v" (one edge per line) in the sidebar
        2. Click the "Initialize Algorithm" button
        3. Use the "Next Step" and "Previous Step" buttons to navigate through the algorithm execution
        4. The visualization will show:
            - Green nodes: Current clique C
            - Yellow nodes: Current candidates
            - Red node: Currently selected vertex
            - Purple node: Vertex that could be added (indicates clique is not maximal)
            - Green edges: Edges within the clique being tested for maximality
        """)
import plotly.express as px
# Define the Report Page
def show_reports_page():
    st.title("Reports and Analysis")
    
    # Create tabs for the different analyses
    tabs = st.tabs(["Comparative Analysis", "Tomita Algorithm", "Bron-Kerbosch Algorithm", "Chiba-Nishizeki Algorithm"])
    
    with tabs[0]:  # Comparative Analysis
        st.header("Comparative Analysis of the Three Algorithms")
        
        # Dataset information table
        st.subheader("Dataset Statistics")
        data = {
            "Dataset": ["Wiki-Vote", "Email-Enron", "As-Skitter"],
            "Total Maximal Cliques": ["459,002", "226,859", "37,322,355"],
            "Largest Clique Size": ["17", "20", "67"]
        }
        df_stats = pd.DataFrame(data)
        st.table(df_stats)
        
        # Execution time comparison
        st.subheader("Execution Time Comparison")
        execution_data = {
            "Algorithm": ["Tomita", "Bron-Kerbosch", "Chiba-Nishizeki"] * 3,
            "Dataset": ["Wiki-Vote"] * 3 + ["Email-Enron"] * 3 + ["As-Skitter"] * 3,
            "Time (seconds)": [27, 212, 484, 142, 582, 697, 32000, 45750, 90437]
        }
        df_execution = pd.DataFrame(execution_data)
        
        # Convert to numeric for plotting
        df_execution["Time (seconds)"] = pd.to_numeric(df_execution["Time (seconds)"])
        
        fig = px.bar(
            df_execution,
            x="Dataset",
            y="Time (seconds)",
            color="Algorithm",
            barmode="group",
            title="Execution Time Comparison",
            log_y=True  # Using log scale due to large differences
        )
        st.plotly_chart(fig)
        
        # Distribution of different size cliques with actual data
        st.subheader("Distribution of Clique Sizes")
        st.write("Select dataset to view clique size distribution:")
        selected_dataset = st.selectbox("Dataset", ["Wiki-Vote", "Email-Enron", "As-Skitter"], key="clique_dist")
        
        # Actual clique size distribution data
        if selected_dataset == "Wiki-Vote":
            # Wiki-Vote data
            counts = [8655, 13718, 27292, 48416, 68872, 83266, 76732, 54456, 35470, 21736, 
          11640, 5449, 2329, 740, 208, 23]
            # Clique sizes from 2 to 20
            sizes = list(range(2, 2 + len(counts)))
            
        elif selected_dataset == "Email-Enron":
            # Email-Enron data
            counts = [14070, 7077, 13319, 18143, 22715, 25896, 24766, 22884, 21393, 17833, 
                      15181, 11487, 7417, 3157, 1178, 286, 41, 10, 6]
            # Clique sizes from 2 to 20
            sizes = list(range(2, 2 + len(counts)))
            
        else:  # As-Skitter
            # As-Skitter data
            counts = [2319807, 3171609, 1823321, 939336, 684873, 598284, 588889, 608937, 665661, 728098, 
                      798073, 877282, 945194, 980831, 939987, 839330, 729601, 639413, 600192, 611976,
                      640890, 673924, 706753, 753633, 818353, 892719, 955212, 999860, 1034106, 1055653,
                      1017560, 946717, 878552, 809485, 744634, 663650, 583922, 520239, 474301, 420796,
                      367879, 321829, 275995, 222461, 158352, 99522, 62437, 39822, 30011, 25637,
                      17707, 9514, 3737, 2042, 1080, 546, 449, 447, 405, 283, 242, 146, 84, 49, 22, 4]
            # Clique sizes from 2 to 67
            sizes = list(range(2, 2 + len(counts)))
        
        clique_df = pd.DataFrame({"Clique Size": sizes, "Frequency": counts})
        
        # Create a bar chart with the actual data
        fig = px.bar(
            clique_df, 
            x="Clique Size", 
            y="Frequency", 
            title=f"Clique Size Distribution for {selected_dataset}"
        )
        
        # Customize x-axis to show more tick marks appropriately
        if selected_dataset == "As-Skitter":
            # For the larger dataset, show fewer ticks
            fig.update_xaxes(tickmode='array', tickvals=list(range(2, 68, 5)))
        else:
            # For smaller datasets, show all ticks
            fig.update_xaxes(tickmode='linear')
            
        # Apply log scale to y-axis for better visualization
        fig.update_yaxes(type="log")
        
        st.plotly_chart(fig)
        
        # Add efficiency comparison
        st.subheader("Algorithm Efficiency Comparison")
        efficiency_data = {
            "Algorithm": ["Tomita", "Bron-Kerbosch", "Chiba-Nishizeki(per clique)"],
            "Time Complexity": ["O(3^(n/3))", "O(d·n·3^(d/3))", "O(α(G)·m)"],
            "Approach": ["DFS with pruning", "Degeneracy ordering with pivoting", "Vertex elimination with arboricity"]
        }
        efficiency_df = pd.DataFrame(efficiency_data)
        st.table(efficiency_df)
    
    with tabs[1]:  # Tomita Algorithm
        st.header("Tomita Algorithm")
        
        st.markdown("""
        **Time Complexity**: O(3^(n/3))
        
        **Summary**: The Tomita algorithm is a depth-first search algorithm for generating all maximal cliques with efficient pruning methods. Its novelty lies in achieving the optimal worst-case time complexity of O(3^(n/3)), matching the theoretical upper bound on maximal cliques in a graph as proven by Moon and Moser. The algorithm uses a clever pivoting strategy that minimizes the size of the search space by selecting pivots that maximize the intersection with the candidate set.
        """)
        
        # Tomita algorithm performance table
        tomita_data = {
            "Dataset": ["Wiki-Vote", "Email-Enron", "As-Skitter"],
            "Total Maximal Cliques": ["459,002", "226,859", "37,322,355"],
            "Execution Time (seconds)": ["27", "142", "~54000"],
            "Largest Clique Size": ["17", "20", "67"]
        }
        tomita_df = pd.DataFrame(tomita_data)
        st.table(tomita_df)
        
        # Visualization of the algorithm's working
        st.subheader("Algorithm Visualization")
        
        # Simple mermaid diagram to show algorithm flow
        mermaid_code = """
        graph TD
            A[Start] --> B[Initialize Q=∅, SUBG=V, CAND=V, FINI=∅]
            B --> C[Call EXPAND]
            C --> D{Is SUBG empty?}
            D -- Yes --> E[Record maximal clique]
            D -- No --> F[Choose pivot u to maximize |CAND∩Γ(u)|]
            F --> G[Build candidate set CAND-Γ(u)]
            G --> H[For each vertex q in candidate set]
            H --> I[Compute SUBG_q, CAND_q, FINI_q]
            I --> J[Add q to Q]
            J --> K[Recursively call EXPAND]
            K --> L[Remove q from Q, move to FINI]
            L --> H
        """
        st.markdown(f"```mermaid\n{mermaid_code}\n```")
        
        # Time complexity breakdown visualization
        st.subheader("Time Complexity Breakdown")
        complexities = {
            "Operation": ["Choosing a pivot", "Generating candidate set", "Processing each vertex", "Overall complexity"],
            "Time Complexity": ["O(n²)", "O(n)", "O(n-1) per vertex", "O(3^(n/3))"]
        }
        complexity_df = pd.DataFrame(complexities)
        st.table(complexity_df)
    
    with tabs[2]:  # Bron-Kerbosch Algorithm
        st.header("Bron-Kerbosch Algorithm with Degeneracy Ordering")
        
        st.markdown("""
        **Time Complexity**: O(d·n·3^(d/3)), where d is the degeneracy of the graph
        
        **Summary**: The Bron-Kerbosch algorithm with degeneracy ordering is particularly efficient for sparse graphs. Its novelty is in using the degeneracy ordering of vertices (processing lowest-degree vertices first) to reduce the search space and minimize recursive calls. This approach is especially powerful when the graph's degeneracy d is much smaller than the number of vertices n, making it near-optimal for many real-world networks.
        """)
        
        # Bron-Kerbosch algorithm performance table
        bk_data = {
            "Dataset": ["Wiki-Vote", "Email-Enron", "As-Skitter"],
            "Total Maximal Cliques": ["459,002", "226,859", "37,322,355"],
            "Execution Time (seconds)": ["212", "582", "~75750"],
            "Largest Clique Size": ["17", "20", "67"]
        }
        bk_df = pd.DataFrame(bk_data)
        st.table(bk_df)
        
        # Visualization of the algorithm's working
        st.subheader("Algorithm Visualization")
        
        # Simple mermaid diagram to show algorithm flow with degeneracy ordering
        mermaid_code = """
        graph TD
            A[Start] --> B[Compute degeneracy ordering]
            B --> C[Process vertices in order]
            C --> D[For each vertex v]
            D --> E[R = {v}]
            E --> F[P = neighbors after v in ordering]
            F --> G[X = neighbors before v in ordering]
            G --> H[Call BronKerbosch with R, P, X]
            H --> I{Are P and X empty?}
            I -- Yes --> J[Record maximal clique]
            I -- No --> K[Choose pivot to maximize neighbors in P]
            K --> L[For each vertex in P not adjacent to pivot]
            L --> M[Update R, P, X for recursive call]
            M --> N[Recursively call BronKerbosch]
            N --> L
        """
        st.markdown(f"```mermaid\n{mermaid_code}\n```")
        
        # Time complexity components
        st.subheader("Time Complexity Components")
        bk_complexity = {
            "Operation": ["Computing degeneracy ordering", "Processing each vertex", "Recursive calls", "Overall complexity"],
            "Time Complexity": ["O(n+m)", "O(d) per vertex", "O(3^(d/3)) per vertex", "O(d·3^(d/3))"]
        }
        bk_complexity_df = pd.DataFrame(bk_complexity)
        st.table(bk_complexity_df)
    
    with tabs[3]:  # Chiba-Nishizeki Algorithm
        st.header("Chiba-Nishizeki Algorithm")
        
        st.markdown("""
        **Time Complexity**: O(α(G)·m) per maximal clique, where α(G) is the arboricity and m is the number of edges
        
        **Summary**: The Chiba-Nishizeki algorithm leverages graph arboricity to efficiently list all maximal cliques. Its novelty lies in using a vertex elimination approach combined with degree-based vertex sorting to minimize work. The algorithm particularly shines in graphs with low arboricity, which includes many real-world networks. The lexicographical test ensures each clique is enumerated exactly once.
        """)
        
        # Chiba-Nishizeki algorithm performance table
        cn_data = {
            "Dataset": ["Wiki-Vote", "Email-Enron", "As-Skitter"],
            "Total Maximal Cliques": ["459,002", "226,859", "37,322,355"],
            "Execution Time (seconds)": ["484", "697", "~130,437"],
            "Largest Clique Size": ["17", "20", "67"]
        }
        cn_df = pd.DataFrame(cn_data)
        st.table(cn_df)
        
        # Visualization of the algorithm's working
        st.subheader("Algorithm Visualization")
        
        # Simple mermaid diagram to show algorithm flow
        mermaid_code = """
        graph TD
            A[Start] --> B[Sort vertices by degree]
            B --> C[Process vertices in order]
            C --> D[For each vertex v]
            D --> E[Check maximality test]
            E --> F[Check lexicographic test]
            F --> G{Tests passed?}
            G -- Yes --> H[Expand clique with v]
            G -- No --> I[Skip v]
            H --> J[Update neighborhood graph]
            J --> K[Continue recursion]
            K --> C
        """
        st.markdown(f"```mermaid\n{mermaid_code}\n```")
        
        # Time complexity components
        st.subheader("Time Complexity Breakdown")
        cn_complexity = {
            "Operation": ["Reading graph", "Vertex sorting", "Building neighborhood subgraphs", "Recursive enumeration", "Pruning & backtracking", "Overall complexity"],
            "Time Complexity": ["O(m)", "O(n)", "O(m)", "O(α(G)·m)", "O(m)", "O(α(G)·m) per clique"]
        }
        cn_complexity_df = pd.DataFrame(cn_complexity)
        st.table(cn_complexity_df)
        
        # Optimization potential
        st.subheader("Potential Optimizations")
        st.markdown("""
        1. **Early vertex elimination**: Detecting and removing 2-sized cliques during bucket sort (improves by factor β)
        2. **Iterator-based implementation**: Avoiding copy operations during recursive calls
        3. **Bitset representation**: Using bitwise operations for set operations in smaller graphs
        """)
# Update the main flow to choose between pages
if __name__ == "__main__":
    if st.session_state.page == "visualization":
        
    else:
        import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict

# Set page configuration
st.set_page_config(page_title="Maximal Clique Analyzer", layout="wide")

# Initialize session state to keep track of algorithm progress
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.current_step = 0
    st.session_state.all_steps = []
    st.session_state.maximal_cliques = []
    st.session_state.graph = None
    st.session_state.pos = None
    st.session_state.call_stack = []
    st.session_state.page = "reports"  # Default page

# Add navigation in sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Algorithm Reports", "Algorithm Visualization"], index=0)

# Update session state with selected page
if page == "Algorithm Visualization":
    st.session_state.page = "visualization"
else:
    st.session_state.page = "reports"

# Functions for algorithm visualization
def parse_edge_list(edge_text):
    """Parse the edge list from input text."""
    edges = []
    for line in edge_text.strip().split('\n'):
        if line.strip():
            u, v = map(int, line.strip().split())
            edges.append((u, v))
    return edges

def build_graph(edges):
    """Build a NetworkX graph from edge list."""
    G = nx.Graph()
    for u, v in edges:
        G.add_edge(u, v)
    return G

def initialize_algorithm(G):
    """Initialize the algorithm data structures."""
    steps = []
    
    # Initial state
    steps.append({
        'description': "Algorithm starts with an empty clique C = {} and all vertices as candidates.",
        'current_clique': set(),
        'candidates': set(G.nodes()),
        'action': 'initialize',
        'maximality_check': None,
        'maximality_result': None,
        'call_level': 0
    })
    
    return steps

def simulate_update(G, initial_C=None, initial_candidates=None, call_level=0):
    """Simulate the UPDATE function to generate steps."""
    if initial_C is None:
        initial_C = set()
    if initial_candidates is None:
        initial_candidates = set(G.nodes())
        
    steps = []
    
    # Add initial state for this call
    steps.append({
        'description': f"Entering UPDATE with C = {initial_C} and candidates = {initial_candidates}",
        'current_clique': initial_C.copy(),
        'candidates': initial_candidates.copy(),
        'action': 'enter_update',
        'maximality_check': None,
        'maximality_result': None,
        'call_level': call_level
    })
    
    # Empty candidates check
    if not initial_candidates:
        maximality_result = is_maximal(G, initial_C)
        steps.append({
            'description': f"Candidates empty. Testing if C = {initial_C} is maximal.",
            'current_clique': initial_C.copy(),
            'candidates': set(),
            'action': 'maximality_test',
            'maximality_check': initial_C.copy(),
            'maximality_result': maximality_result,
            'call_level': call_level
        })
        
        if maximality_result:
            steps.append({
                'description': f"C = {initial_C} is a maximal clique!",
                'current_clique': initial_C.copy(),
                'candidates': set(),
                'action': 'found_clique',
                'maximality_check': None,
                'maximality_result': None,
                'call_level': call_level
            })
        else:
            steps.append({
                'description': f"C = {initial_C} is not maximal.",
                'current_clique': initial_C.copy(),
                'candidates': set(),
                'action': 'not_maximal',
                'maximality_check': None,
                'maximality_result': None,
                'call_level': call_level
            })
            
        steps.append({
            'description': f"Returning from UPDATE with C = {initial_C}",
            'current_clique': initial_C.copy(),
            'candidates': set(),
            'action': 'return',
            'maximality_check': None,
            'maximality_result': None,
            'call_level': call_level
        })
        return steps
    
    # Process each candidate
    candidates = list(initial_candidates)
    while candidates:
        v = candidates[0]
        candidates.remove(v)
        remaining_candidates = set(candidates)
        
        steps.append({
            'description': f"Selected vertex {v} from candidates.",
            'current_clique': initial_C.copy(),
            'candidates': remaining_candidates.copy(),
            'action': 'select_vertex',
            'selected_vertex': v,
            'maximality_check': None,
            'maximality_result': None,
            'call_level': call_level
        })
        
        # Add v to C
        new_C = initial_C.copy()
        new_C.add(v)
        
        steps.append({
            'description': f"Added vertex {v} to clique C, new C = {new_C}",
            'current_clique': new_C.copy(),
            'candidates': remaining_candidates.copy(),
            'action': 'add_to_clique',
            'selected_vertex': v,
            'maximality_check': None,
            'maximality_result': None,
            'call_level': call_level
        })
        
        # Find new candidates that are adjacent to v
        new_candidates = set()
        for u in remaining_candidates:
            if G.has_edge(v, u):
                new_candidates.add(u)
        
        steps.append({
            'description': f"Filtered candidates to those adjacent to {v}, new candidates = {new_candidates}",
            'current_clique': new_C.copy(),
            'candidates': new_candidates.copy(),
            'action': 'filter_candidates',
            'selected_vertex': v,
            'maximality_check': None,
            'maximality_result': None,
            'call_level': call_level
        })
        
        # Recursive call
        recursive_steps = simulate_update(G, new_C, new_candidates, call_level + 1)
        steps.extend(recursive_steps)
        
    steps.append({
        'description': f"No more candidates to process. Returning from UPDATE with C = {initial_C}",
        'current_clique': initial_C.copy(),
        'candidates': set(),
        'action': 'return',
        'maximality_check': None,
        'maximality_result': None,
        'call_level': call_level
    })
    
    return steps

def is_maximal(G, C):
    """Check if a clique is maximal."""
    if not C:
        return False
    
    # Check if C is actually a clique
    for u in C:
        for v in C:
            if u != v and not G.has_edge(u, v):
                return False
    
    # Check if any vertex outside C can be added
    for y in G.nodes():
        if y not in C:
            can_add = True
            for x in C:
                if not G.has_edge(x, y):
                    can_add = False
                    break
            if can_add:
                return False  # Not maximal
    
    return True  # Is maximal

def find_maximal_cliques(G):
    """Find all maximal cliques in the graph."""
    maximal_cliques = []
    
    def update(C, candidates):
        if not candidates:
            if is_maximal(G, C):
                maximal_cliques.append(C.copy())
            return
        
        candidates_copy = candidates.copy()
        while candidates_copy:
            v = next(iter(candidates_copy))
            candidates_copy.remove(v)
            
            new_C = C.copy()
            new_C.add(v)
            
            new_candidates = set()
            for u in candidates_copy:
                if G.has_edge(v, u):
                    new_candidates.add(u)
            
            update(new_C, new_candidates)
    
    update(set(), set(G.nodes()))
    return maximal_cliques

# Define the Algorithm Visualization Page
def show_visualization_page():
    st.title("Maximal Clique Algorithm Visualization")
    
    # Graph Input
    st.sidebar.header("Graph Input")
    
    # Default edge list
    default_edges = """0 1
1 2
0 2
1 3
1 4
2 4
2 3
3 4
6 7
7 8
8 9
9 6
6 8
7 9"""
    
    edge_list_text = st.sidebar.text_area("Edge List (one edge per line)", value=default_edges, height=200)
    
    if st.sidebar.button("Initialize Algorithm"):
        # Parse edge list
        edges = parse_edge_list(edge_list_text)
        
        # Build graph
        G = build_graph(edges)
        
        # Generate positions for visualization
        pos = nx.spring_layout(G, seed=42)
        
        # Generate all steps
        all_steps = initialize_algorithm(G)
        all_steps.extend(simulate_update(G)[1:])  # Skip the first step which is duplicate
        
        # Find all maximal cliques
        maximal_cliques = find_maximal_cliques(G)
        
        # Save to session state
        st.session_state.initialized = True
        st.session_state.current_step = 0
        st.session_state.all_steps = all_steps
        st.session_state.maximal_cliques = maximal_cliques
        st.session_state.graph = G
        st.session_state.pos = pos
        st.session_state.total_steps = len(all_steps)
    
    # Step controls
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col1:
        if st.session_state.initialized and st.session_state.current_step > 0:
            if st.button("Previous Step"):
                st.session_state.current_step -= 1
    
    with col2:
        if st.session_state.initialized:
            st.progress(st.session_state.current_step / max(1, len(st.session_state.all_steps) - 1))
            step_text = f"Step {st.session_state.current_step + 1}/{len(st.session_state.all_steps)}"
            st.markdown(f"<div style='text-align: center'>{step_text}</div>", unsafe_allow_html=True)
    
    with col3:
        if st.session_state.initialized and st.session_state.current_step < len(st.session_state.all_steps) - 1:
            if st.button("Next Step"):
                st.session_state.current_step += 1
    
    # Visualization area
    if st.session_state.initialized:
        col_graph, col_status = st.columns([2, 1])
        
        with col_graph:
            G = st.session_state.graph
            pos = st.session_state.pos
            step = st.session_state.all_steps[st.session_state.current_step]
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Draw graph
            nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax)
            
            # Get node colors based on current state
            current_clique = step['current_clique']
            candidates = step['candidates']
            
            # Track position of nodes in different sets for label placement
            clique_nodes = []
            candidate_nodes = []
            other_nodes = []
            
            for node in G.nodes():
                if node in current_clique:
                    clique_nodes.append(node)
                elif node in candidates:
                    candidate_nodes.append(node)
                else:
                    other_nodes.append(node)
            
            # Draw nodes with different colors
            nx.draw_networkx_nodes(G, pos, nodelist=clique_nodes, 
                                    node_color='green', node_size=700, alpha=0.8, ax=ax)
            nx.draw_networkx_nodes(G, pos, nodelist=candidate_nodes, 
                                    node_color='yellow', node_size=500, alpha=0.8, ax=ax)
            nx.draw_networkx_nodes(G, pos, nodelist=other_nodes, 
                                    node_color='gray', node_size=500, alpha=0.5, ax=ax)
            
            # Highlight selected vertex if applicable
            if 'selected_vertex' in step and step['selected_vertex'] is not None:
                nx.draw_networkx_nodes(G, pos, nodelist=[step['selected_vertex']], 
                                        node_color='red', node_size=700, alpha=0.8, ax=ax)
            
            # Highlight maximality check if applicable
            if step['maximality_check'] is not None:
                check_edges = []
                check_nodes = list(step['maximality_check'])
                
                # Get all edges in the potential clique
                for i, u in enumerate(check_nodes):
                    for v in check_nodes[i+1:]:
                        if G.has_edge(u, v):
                            check_edges.append((u, v))
                
                nx.draw_networkx_edges(G, pos, edgelist=check_edges, width=3, 
                                        edge_color='green', ax=ax)
                
                # If not maximal, highlight the vertex that could be added
                if not step['maximality_result']:
                    for y in G.nodes():
                        if y not in step['maximality_check']:
                            can_add = True
                            for x in step['maximality_check']:
                                if not G.has_edge(x, y):
                                    can_add = False
                                    break
                            if can_add:
                                nx.draw_networkx_nodes(G, pos, nodelist=[y], 
                                                        node_color='purple', node_size=700, 
                                                        alpha=0.8, ax=ax)
                                # Draw edges to show why it can be added
                                connecting_edges = [(y, x) for x in step['maximality_check'] if G.has_edge(y, x)]
                                nx.draw_networkx_edges(G, pos, edgelist=connecting_edges, 
                                                        width=2.5, edge_color='purple', 
                                                        style='dashed', ax=ax)
                                break
            
            # Draw node labels
            nx.draw_networkx_labels(G, pos, ax=ax)
            
            # Add legend
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=15, label='In Clique C'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', markersize=15, label='Candidates'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=15, label='Selected Vertex'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=15, label='Other Vertices'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=15, label='Could Be Added (Not Maximal)')
            ]
            ax.legend(handles=legend_elements, loc='upper right')
            
            plt.title("Maximal Clique Algorithm Visualization")
            plt.axis('off')
            st.pyplot(fig)
        
        with col_status:
            st.subheader("Algorithm State")
            
            # Display current clique
            st.markdown(f"**Current Clique C:** {{{', '.join(map(str, step['current_clique']))}}}")
            
            # Display candidates
            st.markdown(f"**Candidates:** {{{', '.join(map(str, step['candidates']))}}}")
            
            # Display call level
            st.markdown(f"**Recursion Depth:** {step['call_level']}")
            
            # Display step description
            st.subheader("Current Step")
            st.markdown(step['description'])
            
            # Display action type
            action_types = {
                'initialize': "Initializing algorithm",
                'enter_update': "Entering UPDATE function",
                'maximality_test': "Testing for maximality",
                'found_clique': "Found maximal clique!",
                'not_maximal': "Clique is not maximal",
                'select_vertex': "Selecting a vertex",
                'add_to_clique': "Adding vertex to clique",
                'filter_candidates': "Filtering candidates",
                'return': "Returning from function call"
            }
            
            st.markdown(f"**Action:** {action_types.get(step['action'], step['action'])}")
            
            # Display all found maximal cliques
            st.subheader("Final Results")
            st.markdown(f"**Maximal Cliques Found:** {len(st.session_state.maximal_cliques)}")
            
            clique_df = pd.DataFrame({
                'Clique': [f"{{{', '.join(map(str, clique))}}}" for clique in st.session_state.maximal_cliques]
            })
            st.dataframe(clique_df)
    
    # Display instructions if not initialized
    if not st.session_state.initialized:
        st.info("Enter the edge list in the sidebar and click 'Initialize Algorithm' to begin the visualization.")
        
        st.markdown("""
        ## How to use this visualization:
        
        1. Enter the edge list in the format "u v" (one edge per line) in the sidebar
        2. Click the "Initialize Algorithm" button
        3. Use the "Next Step" and "Previous Step" buttons to navigate through the algorithm execution
        4. The visualization will show:
            - Green nodes: Current clique C
            - Yellow nodes: Current candidates
            - Red node: Currently selected vertex
            - Purple node: Vertex that could be added (indicates clique is not maximal)
            - Green edges: Edges within the clique being tested for maximality
        """)
import plotly.express as px
# Define the Report Page
def show_reports_page():
    st.title("Reports and Analysis")
    
    # Create tabs for the different analyses
    tabs = st.tabs(["Comparative Analysis", "Tomita Algorithm", "Bron-Kerbosch Algorithm", "Chiba-Nishizeki Algorithm"])
    
    with tabs[0]:  # Comparative Analysis
        st.header("Comparative Analysis of the Three Algorithms")
        
        # Dataset information table
        st.subheader("Dataset Statistics")
        data = {
            "Dataset": ["Wiki-Vote", "Email-Enron", "As-Skitter"],
            "Total Maximal Cliques": ["459,002", "226,859", "37,322,355"],
            "Largest Clique Size": ["17", "20", "67"]
        }
        df_stats = pd.DataFrame(data)
        st.table(df_stats)
        
        # Execution time comparison
        st.subheader("Execution Time Comparison")
        execution_data = {
            "Algorithm": ["Tomita", "Bron-Kerbosch", "Chiba-Nishizeki"] * 3,
            "Dataset": ["Wiki-Vote"] * 3 + ["Email-Enron"] * 3 + ["As-Skitter"] * 3,
            "Time (seconds)": [27, 212, 484, 142, 582, 697, 32000, 45750, 90437]
        }
        df_execution = pd.DataFrame(execution_data)
        
        # Convert to numeric for plotting
        df_execution["Time (seconds)"] = pd.to_numeric(df_execution["Time (seconds)"])
        
        fig = px.bar(
            df_execution,
            x="Dataset",
            y="Time (seconds)",
            color="Algorithm",
            barmode="group",
            title="Execution Time Comparison",
            log_y=True  # Using log scale due to large differences
        )
        st.plotly_chart(fig)
        
        # Distribution of different size cliques with actual data
        st.subheader("Distribution of Clique Sizes")
        st.write("Select dataset to view clique size distribution:")
        selected_dataset = st.selectbox("Dataset", ["Wiki-Vote", "Email-Enron", "As-Skitter"], key="clique_dist")
        
        # Actual clique size distribution data
        if selected_dataset == "Wiki-Vote":
            # Wiki-Vote data
            counts = [8655, 13718, 27292, 48416, 68872, 83266, 76732, 54456, 35470, 21736, 
          11640, 5449, 2329, 740, 208, 23]
            # Clique sizes from 2 to 20
            sizes = list(range(2, 2 + len(counts)))
            
        elif selected_dataset == "Email-Enron":
            # Email-Enron data
            counts = [14070, 7077, 13319, 18143, 22715, 25896, 24766, 22884, 21393, 17833, 
                      15181, 11487, 7417, 3157, 1178, 286, 41, 10, 6]
            # Clique sizes from 2 to 20
            sizes = list(range(2, 2 + len(counts)))
            
        else:  # As-Skitter
            # As-Skitter data
            counts = [2319807, 3171609, 1823321, 939336, 684873, 598284, 588889, 608937, 665661, 728098, 
                      798073, 877282, 945194, 980831, 939987, 839330, 729601, 639413, 600192, 611976,
                      640890, 673924, 706753, 753633, 818353, 892719, 955212, 999860, 1034106, 1055653,
                      1017560, 946717, 878552, 809485, 744634, 663650, 583922, 520239, 474301, 420796,
                      367879, 321829, 275995, 222461, 158352, 99522, 62437, 39822, 30011, 25637,
                      17707, 9514, 3737, 2042, 1080, 546, 449, 447, 405, 283, 242, 146, 84, 49, 22, 4]
            # Clique sizes from 2 to 67
            sizes = list(range(2, 2 + len(counts)))
        
        clique_df = pd.DataFrame({"Clique Size": sizes, "Frequency": counts})
        
        # Create a bar chart with the actual data
        fig = px.bar(
            clique_df, 
            x="Clique Size", 
            y="Frequency", 
            title=f"Clique Size Distribution for {selected_dataset}"
        )
        
        # Customize x-axis to show more tick marks appropriately
        if selected_dataset == "As-Skitter":
            # For the larger dataset, show fewer ticks
            fig.update_xaxes(tickmode='array', tickvals=list(range(2, 68, 5)))
        else:
            # For smaller datasets, show all ticks
            fig.update_xaxes(tickmode='linear')
            
        # Apply log scale to y-axis for better visualization
        fig.update_yaxes(type="log")
        
        st.plotly_chart(fig)
        
        # Add efficiency comparison
        st.subheader("Algorithm Efficiency Comparison")
        efficiency_data = {
            "Algorithm": ["Tomita", "Bron-Kerbosch", "Chiba-Nishizeki(per clique)"],
            "Time Complexity": ["O(3^(n/3))", "O(d·n·3^(d/3))", "O(α(G)·m)"],
            "Approach": ["DFS with pruning", "Degeneracy ordering with pivoting", "Vertex elimination with arboricity"]
        }
        efficiency_df = pd.DataFrame(efficiency_data)
        st.table(efficiency_df)
    
    with tabs[1]:  # Tomita Algorithm
        st.header("Tomita Algorithm")
        
        st.markdown("""
        **Time Complexity**: O(3^(n/3))
        
        **Summary**: The Tomita algorithm is a depth-first search algorithm for generating all maximal cliques with efficient pruning methods. Its novelty lies in achieving the optimal worst-case time complexity of O(3^(n/3)), matching the theoretical upper bound on maximal cliques in a graph as proven by Moon and Moser. The algorithm uses a clever pivoting strategy that minimizes the size of the search space by selecting pivots that maximize the intersection with the candidate set.
        """)
        
        # Tomita algorithm performance table
        tomita_data = {
            "Dataset": ["Wiki-Vote", "Email-Enron", "As-Skitter"],
            "Total Maximal Cliques": ["459,002", "226,859", "37,322,355"],
            "Execution Time (seconds)": ["27", "142", "~54000"],
            "Largest Clique Size": ["17", "20", "67"]
        }
        tomita_df = pd.DataFrame(tomita_data)
        st.table(tomita_df)
        
        # Visualization of the algorithm's working
        st.subheader("Algorithm Visualization")
        
        # Simple mermaid diagram to show algorithm flow
        mermaid_code = """
        graph TD
            A[Start] --> B[Initialize Q=∅, SUBG=V, CAND=V, FINI=∅]
            B --> C[Call EXPAND]
            C --> D{Is SUBG empty?}
            D -- Yes --> E[Record maximal clique]
            D -- No --> F[Choose pivot u to maximize |CAND∩Γ(u)|]
            F --> G[Build candidate set CAND-Γ(u)]
            G --> H[For each vertex q in candidate set]
            H --> I[Compute SUBG_q, CAND_q, FINI_q]
            I --> J[Add q to Q]
            J --> K[Recursively call EXPAND]
            K --> L[Remove q from Q, move to FINI]
            L --> H
        """
        st.markdown(f"```mermaid\n{mermaid_code}\n```")
        
        # Time complexity breakdown visualization
        st.subheader("Time Complexity Breakdown")
        complexities = {
            "Operation": ["Choosing a pivot", "Generating candidate set", "Processing each vertex", "Overall complexity"],
            "Time Complexity": ["O(n²)", "O(n)", "O(n-1) per vertex", "O(3^(n/3))"]
        }
        complexity_df = pd.DataFrame(complexities)
        st.table(complexity_df)
    
    with tabs[2]:  # Bron-Kerbosch Algorithm
        st.header("Bron-Kerbosch Algorithm with Degeneracy Ordering")
        
        st.markdown("""
        **Time Complexity**: O(d·n·3^(d/3)), where d is the degeneracy of the graph
        
        **Summary**: The Bron-Kerbosch algorithm with degeneracy ordering is particularly efficient for sparse graphs. Its novelty is in using the degeneracy ordering of vertices (processing lowest-degree vertices first) to reduce the search space and minimize recursive calls. This approach is especially powerful when the graph's degeneracy d is much smaller than the number of vertices n, making it near-optimal for many real-world networks.
        """)
        
        # Bron-Kerbosch algorithm performance table
        bk_data = {
            "Dataset": ["Wiki-Vote", "Email-Enron", "As-Skitter"],
            "Total Maximal Cliques": ["459,002", "226,859", "37,322,355"],
            "Execution Time (seconds)": ["212", "582", "~75750"],
            "Largest Clique Size": ["17", "20", "67"]
        }
        bk_df = pd.DataFrame(bk_data)
        st.table(bk_df)
        
        # Visualization of the algorithm's working
        st.subheader("Algorithm Visualization")
        
        # Simple mermaid diagram to show algorithm flow with degeneracy ordering
        mermaid_code = """
        graph TD
            A[Start] --> B[Compute degeneracy ordering]
            B --> C[Process vertices in order]
            C --> D[For each vertex v]
            D --> E[R = {v}]
            E --> F[P = neighbors after v in ordering]
            F --> G[X = neighbors before v in ordering]
            G --> H[Call BronKerbosch with R, P, X]
            H --> I{Are P and X empty?}
            I -- Yes --> J[Record maximal clique]
            I -- No --> K[Choose pivot to maximize neighbors in P]
            K --> L[For each vertex in P not adjacent to pivot]
            L --> M[Update R, P, X for recursive call]
            M --> N[Recursively call BronKerbosch]
            N --> L
        """
        st.markdown(f"```mermaid\n{mermaid_code}\n```")
        
        # Time complexity components
        st.subheader("Time Complexity Components")
        bk_complexity = {
            "Operation": ["Computing degeneracy ordering", "Processing each vertex", "Recursive calls", "Overall complexity"],
            "Time Complexity": ["O(n+m)", "O(d) per vertex", "O(3^(d/3)) per vertex", "O(d·3^(d/3))"]
        }
        bk_complexity_df = pd.DataFrame(bk_complexity)
        st.table(bk_complexity_df)
    
    with tabs[3]:  # Chiba-Nishizeki Algorithm
        st.header("Chiba-Nishizeki Algorithm")
        
        st.markdown("""
        **Time Complexity**: O(α(G)·m) per maximal clique, where α(G) is the arboricity and m is the number of edges
        
        **Summary**: The Chiba-Nishizeki algorithm leverages graph arboricity to efficiently list all maximal cliques. Its novelty lies in using a vertex elimination approach combined with degree-based vertex sorting to minimize work. The algorithm particularly shines in graphs with low arboricity, which includes many real-world networks. The lexicographical test ensures each clique is enumerated exactly once.
        """)
        
        # Chiba-Nishizeki algorithm performance table
        cn_data = {
            "Dataset": ["Wiki-Vote", "Email-Enron", "As-Skitter"],
            "Total Maximal Cliques": ["459,002", "226,859", "37,322,355"],
            "Execution Time (seconds)": ["484", "697", "~130,437"],
            "Largest Clique Size": ["17", "20", "67"]
        }
        cn_df = pd.DataFrame(cn_data)
        st.table(cn_df)
        
        # Visualization of the algorithm's working
        st.subheader("Algorithm Visualization")
        
        # Simple mermaid diagram to show algorithm flow
        mermaid_code = """
        graph TD
            A[Start] --> B[Sort vertices by degree]
            B --> C[Process vertices in order]
            C --> D[For each vertex v]
            D --> E[Check maximality test]
            E --> F[Check lexicographic test]
            F --> G{Tests passed?}
            G -- Yes --> H[Expand clique with v]
            G -- No --> I[Skip v]
            H --> J[Update neighborhood graph]
            J --> K[Continue recursion]
            K --> C
        """
        st.markdown(f"```mermaid\n{mermaid_code}\n```")
        
        # Time complexity components
        st.subheader("Time Complexity Breakdown")
        cn_complexity = {
            "Operation": ["Reading graph", "Vertex sorting", "Building neighborhood subgraphs", "Recursive enumeration", "Pruning & backtracking", "Overall complexity"],
            "Time Complexity": ["O(m)", "O(n)", "O(m)", "O(α(G)·m)", "O(m)", "O(α(G)·m) per clique"]
        }
        cn_complexity_df = pd.DataFrame(cn_complexity)
        st.table(cn_complexity_df)
        
        # Optimization potential
        st.subheader("Potential Optimizations")
        st.markdown("""
        1. **Early vertex elimination**: Detecting and removing 2-sized cliques during bucket sort (improves by factor β)
        2. **Iterator-based implementation**: Avoiding copy operations during recursive calls
        3. **Bitset representation**: Using bitwise operations for set operations in smaller graphs
        """)
# Update the main flow to choose between pages
if __name__ == "__main__":
    if st.session_state.page == "visualization":
        show_reports_page()
    else:
        show_visualization_page()
        
        
